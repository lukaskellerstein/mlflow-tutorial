"""
L3-2.3 — DeepAgents Integration: Multi-Agent Orchestration with MLflow

Demonstrates multi-agent orchestration patterns inspired by DeepAgents
(LangChain-AI's multi-agent framework) with full MLflow tracing:
  1. Builds an orchestration system with specialist agents
  2. Traces inter-agent communication with nested spans
  3. Tracks handoffs, delegation, and coordination metrics
  4. Compares multi-agent vs single-agent approaches
"""

import json
import time
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# 1. Agent definitions
# ---------------------------------------------------------------------------
@dataclass
class AgentMessage:
    """A message passed between agents in the orchestration."""
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Result produced by a specialist agent."""
    agent_name: str
    output: str
    duration_s: float
    token_estimate: int


class BaseAgent:
    """Base class for all agents in the orchestration system."""

    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model="google/gemma-4-26b-a4b", base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.7)

    @mlflow.trace
    def invoke(self, task: str) -> AgentResult:
        """Run the agent on a task, traced by MLflow."""
        with mlflow.start_span(name=f"agent_{self.name}") as span:
            span.set_inputs({"task": task, "agent": self.name, "role": self.role})
            start = time.time()

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=task),
            ]
            response = self.llm.invoke(messages)
            output = response.content
            elapsed = round(time.time() - start, 2)
            token_est = len(output.split())

            span.set_outputs({
                "output_preview": output[:300],
                "duration_s": elapsed,
                "token_estimate": token_est,
            })

            return AgentResult(
                agent_name=self.name,
                output=output,
                duration_s=elapsed,
                token_estimate=token_est,
            )


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="researcher",
            role="information_gathering",
            system_prompt=(
                "You are a research specialist. Given a topic, provide a concise, "
                "factual summary of the key points. Focus on accuracy and breadth. "
                "Keep your response under 200 words."
            ),
        )


class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="analyst",
            role="data_analysis",
            system_prompt=(
                "You are an analysis specialist. Given research findings, identify "
                "the most important patterns, pros/cons, and implications. "
                "Structure your analysis clearly. Keep your response under 200 words."
            ),
        )


class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="writer",
            role="content_synthesis",
            system_prompt=(
                "You are a writing specialist. Given research and analysis, produce "
                "a clear, well-structured summary suitable for a technical audience. "
                "Use bullet points where appropriate. Keep your response under 200 words."
            ),
        )


# ---------------------------------------------------------------------------
# 2. Orchestrator — decomposes tasks and delegates to specialists
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """Decomposes complex tasks and delegates to specialist agents."""

    def __init__(self):
        self.name = "orchestrator"
        self.llm = ChatOpenAI(model="google/gemma-4-26b-a4b", base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.3)
        self.specialists: dict[str, BaseAgent] = {
            "researcher": ResearchAgent(),
            "analyst": AnalysisAgent(),
            "writer": WriterAgent(),
        }
        self.message_log: list[AgentMessage] = []
        self.handoff_count = 0

    def _log_message(self, sender: str, receiver: str, content: str) -> None:
        """Record an inter-agent message."""
        self.message_log.append(AgentMessage(
            sender=sender, receiver=receiver,
            content=content[:200],
        ))

    @mlflow.trace
    def decompose_task(self, task: str) -> list[dict]:
        """Break a complex task into sub-tasks for specialists."""
        with mlflow.start_span(name="task_decomposition") as span:
            span.set_inputs({"task": task})

            # The orchestrator always follows a fixed pipeline:
            #   research -> analysis -> writing
            subtasks = [
                {"agent": "researcher", "task": f"Research the following topic: {task}"},
                {"agent": "analyst", "task": "Analyze the research findings and identify key patterns."},
                {"agent": "writer", "task": "Write a clear summary based on the research and analysis."},
            ]

            span.set_outputs({"subtasks": [s["agent"] for s in subtasks]})
            return subtasks

    @mlflow.trace
    def run(self, task: str) -> dict:
        """Execute the full orchestration pipeline on a task."""
        with mlflow.start_span(name="orchestration_pipeline") as root_span:
            root_span.set_inputs({"task": task})
            pipeline_start = time.time()
            self.message_log.clear()
            self.handoff_count = 0

            # Step 1: Decompose
            print("    [orchestrator] Decomposing task...")
            subtasks = self.decompose_task(task)

            # Step 2: Execute pipeline — each agent's output feeds the next
            agent_results: list[AgentResult] = []
            accumulated_context = ""

            for step in subtasks:
                agent_name = step["agent"]
                agent = self.specialists[agent_name]
                agent_task = step["task"]

                # Append prior context for analyst and writer
                if accumulated_context:
                    agent_task = f"{agent_task}\n\nContext from previous steps:\n{accumulated_context}"

                # Log the handoff
                self._log_message("orchestrator", agent_name, agent_task)
                self.handoff_count += 1
                print(f"    [orchestrator] -> Delegating to {agent_name}")

                result = agent.invoke(agent_task)
                agent_results.append(result)
                print(f"    [{agent_name}] Completed in {result.duration_s}s")

                # Log the response back
                self._log_message(agent_name, "orchestrator", result.output[:200])
                accumulated_context += f"\n\n{agent_name.upper()} output:\n{result.output}"

            pipeline_duration = round(time.time() - pipeline_start, 2)
            final_output = agent_results[-1].output

            root_span.set_outputs({
                "final_output_preview": final_output[:300],
                "pipeline_duration_s": pipeline_duration,
                "agents_used": len(agent_results),
                "handoffs": self.handoff_count,
            })

            return {
                "final_output": final_output,
                "agent_results": agent_results,
                "message_log": self.message_log,
                "handoff_count": self.handoff_count,
                "pipeline_duration_s": pipeline_duration,
            }


# ---------------------------------------------------------------------------
# 3. Single-agent baseline for comparison
# ---------------------------------------------------------------------------
class SingleAgent(BaseAgent):
    """A single generalist agent that handles the entire task alone."""

    def __init__(self):
        super().__init__(
            name="generalist",
            role="all_in_one",
            system_prompt=(
                "You are a generalist assistant. Research the topic, analyze the key "
                "points, and produce a clear summary. Keep your response under 300 words."
            ),
        )


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L3-2.3 — DeepAgents: Multi-Agent Orchestration + MLflow")
    print("=" * 60)

    task = "Research and summarize the benefits of microservices architecture"

    # ---- Part 1: Multi-agent orchestration --------------------------------
    print("\n--- Part 1: Multi-agent orchestration pipeline ---")
    orchestrator = OrchestratorAgent()

    with mlflow.start_run(run_name="multi_agent_orchestration") as parent_run:
        mlflow.set_tags({
            "approach": "multi_agent",
            "framework_inspiration": "deepagents",
            "model": "google/gemma-4-26b-a4b",
        })

        result = orchestrator.run(task)

        # ---- Part 2: Log inter-agent communication -----------------------
        print("\n--- Part 2: Inter-agent communication log ---")
        for msg in result["message_log"]:
            direction = f"{msg.sender} -> {msg.receiver}"
            print(f"    {direction:30s} | {msg.content[:60]}...")

        # Log message flow as artifact
        comm_data = [
            {"sender": m.sender, "receiver": m.receiver,
             "content": m.content, "timestamp": m.timestamp}
            for m in result["message_log"]
        ]
        comm_path = "/tmp/agent_communication_log.json"
        with open(comm_path, "w") as f:
            json.dump(comm_data, f, indent=2)
        mlflow.log_artifact(comm_path, artifact_path="communication")

        # ---- Part 3: Multi-agent metrics ---------------------------------
        print("\n--- Part 3: Multi-agent coordination metrics ---")
        agent_results = result["agent_results"]
        total_tokens = sum(r.token_estimate for r in agent_results)
        total_duration = result["pipeline_duration_s"]
        agent_durations = {r.agent_name: r.duration_s for r in agent_results}

        # Coordination overhead = total pipeline time minus sum of agent times
        sum_agent_time = sum(r.duration_s for r in agent_results)
        coordination_overhead = round(total_duration - sum_agent_time, 2)

        metrics = {
            "agents_used": len(agent_results),
            "handoffs": result["handoff_count"],
            "total_steps": len(agent_results),
            "total_duration_s": total_duration,
            "total_token_estimate": total_tokens,
            "coordination_overhead_s": max(coordination_overhead, 0.0),
        }
        # Log per-agent durations
        for name, dur in agent_durations.items():
            metrics[f"agent_{name}_duration_s"] = dur

        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "task": task[:250],
            "pipeline": "researcher -> analyst -> writer",
            "num_specialists": len(orchestrator.specialists),
        })

        for k, v in metrics.items():
            print(f"    {k:35s} = {v}")

        multi_run_id = parent_run.info.run_id

    # ---- Part 4: Single-agent baseline ------------------------------------
    print("\n--- Part 4: Single-agent baseline comparison ---")
    single = SingleAgent()

    with mlflow.start_run(run_name="single_agent_baseline"):
        mlflow.set_tags({
            "approach": "single_agent",
            "model": "google/gemma-4-26b-a4b",
        })
        mlflow.log_param("task", task[:250])

        single_start = time.time()
        single_result = single.invoke(task)
        single_duration = round(time.time() - single_start, 2)

        single_metrics = {
            "agents_used": 1,
            "handoffs": 0,
            "total_steps": 1,
            "total_duration_s": single_duration,
            "total_token_estimate": single_result.token_estimate,
            "coordination_overhead_s": 0.0,
        }
        mlflow.log_metrics(single_metrics)

        print(f"    Single agent completed in {single_duration}s")
        print(f"    Token estimate: {single_result.token_estimate}")

    # ---- Part 5: Comparison summary ---------------------------------------
    print("\n--- Part 5: Multi-agent vs single-agent comparison ---")
    comparison = pd.DataFrame([
        {
            "approach": "multi_agent",
            "agents_used": metrics["agents_used"],
            "handoffs": metrics["handoffs"],
            "duration_s": metrics["total_duration_s"],
            "token_estimate": metrics["total_token_estimate"],
            "coordination_overhead_s": metrics["coordination_overhead_s"],
        },
        {
            "approach": "single_agent",
            "agents_used": 1,
            "handoffs": 0,
            "duration_s": single_metrics["total_duration_s"],
            "token_estimate": single_metrics["total_token_estimate"],
            "coordination_overhead_s": 0.0,
        },
    ])
    print(comparison.to_string(index=False))

    # Save comparison as artifact on a summary run
    with mlflow.start_run(run_name="approach_comparison"):
        mlflow.set_tag("run_type", "comparison")
        csv_path = "/tmp/agent_approach_comparison.csv"
        comparison.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="comparison")

        speedup = (
            single_metrics["total_duration_s"] / max(metrics["total_duration_s"], 0.01)
        )
        mlflow.log_metrics({
            "multi_agent_duration_s": metrics["total_duration_s"],
            "single_agent_duration_s": single_metrics["total_duration_s"],
            "duration_ratio": round(speedup, 2),
            "multi_agent_tokens": metrics["total_token_estimate"],
            "single_agent_tokens": single_metrics["total_token_estimate"],
        })

    # ---- Final output -----------------------------------------------------
    print("\n--- Final synthesized output (multi-agent) ---")
    print("-" * 60)
    print(result["final_output"][:600])
    print("-" * 60)

    print(f"\n  Multi-agent run:  {multi_run_id}")
    print(f"  View in MLflow UI: http://127.0.0.1:5000")

    print("\n" + "=" * 60)
    print("Done! Compare multi-agent vs single-agent traces in MLflow.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M2_custom_integrations/3_deepagents")
    main()
