"""
L3-M4.1 — Capstone: Production AI Agent Platform

A mini production agent platform that brings together the full MLflow tutorial:
  - Agent Registry: register and manage multiple agents
  - Evaluation System: automated evaluation with quality gates
  - Tracing & Monitoring: full observability via @mlflow.trace
  - Deployment Pipeline: evaluate -> gate -> approve/reject -> deploy
  - Platform Demo: end-to-end run of the entire platform

Builds on every prior module: tracking, models, evaluation, tracing,
agent observability, and deployment patterns.
"""

import time
from dataclasses import dataclass

import mlflow
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L3/M4_capstones/1_agent_platform")

# ---------------------------------------------------------------------------
# Shared tools available to agents
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "python": "Python is a high-level programming language known for readability "
    "and a vast ecosystem of libraries for data science and AI.",
    "mlflow": "MLflow is an open-source platform for managing the full ML lifecycle "
    "including tracking, model registry, evaluation, and deployment.",
    "docker": "Docker is a containerization platform that packages applications "
    "into portable containers for consistent deployment.",
    "kubernetes": "Kubernetes is a container orchestration system for automating "
    "deployment, scaling, and management of containerized applications.",
    "langchain": "LangChain is a framework for building LLM-powered applications "
    "with abstractions for chains, agents, memory, and tools.",
    "langgraph": "LangGraph builds stateful multi-actor LLM applications using "
    "graph-based workflows with nodes, edges, and state management.",
}


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for factual information on a topic."""
    q = query.lower()
    results = [v for k, v in KNOWLEDGE_BASE.items() if k in q]
    return results[0] if results else f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3' or '10 * 5'."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))  # nosec: reached only for whitelisted arithmetic chars
        return "Invalid expression — only basic arithmetic is supported."
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def summarize_text(text: str) -> str:
    """Return a brief one-sentence summary of the given text."""
    words = text.split()
    if len(words) <= 15:
        return text
    return " ".join(words[:15]) + "..."


# ---------------------------------------------------------------------------
# Agent Registry — register and manage multiple agents
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Configuration for a registered agent."""

    name: str
    description: str
    tools: list
    system_prompt: str
    version: str = "1.0.0"


class AgentRegistry:
    """Registry for managing multiple agents in the platform."""

    def __init__(self) -> None:
        self._agents: dict[str, dict] = {}

    def register(self, config: AgentConfig) -> None:
        """Register an agent with the given configuration."""
        llm = ChatOpenAI(
            model="google/gemma-4-26b-a4b",
            base_url="http://localhost:1234/v1",
            api_key=SecretStr("lm-studio"),
            temperature=0.0,
        )
        agent = create_react_agent(
            model=llm,
            tools=config.tools,
            prompt=config.system_prompt,
        )
        self._agents[config.name] = {
            "agent": agent,
            "config": config,
            "status": "registered",
        }
        print(f"    Registered: {config.name} v{config.version}")

    def get(self, name: str):
        """Return the compiled agent graph for a registered agent."""
        entry = self._agents.get(name)
        if not entry:
            raise KeyError(f"Agent '{name}' not found in registry")
        return entry["agent"]

    def get_config(self, name: str) -> AgentConfig:
        """Return the configuration of a registered agent."""
        entry = self._agents.get(name)
        if not entry:
            raise KeyError(f"Agent '{name}' not found in registry")
        return entry["config"]

    def set_status(self, name: str, status: str) -> None:
        """Update the deployment status of an agent."""
        if name in self._agents:
            self._agents[name]["status"] = status

    def get_status(self, name: str) -> str:
        """Return the current status of an agent."""
        return self._agents.get(name, {}).get("status", "unknown")

    def list_agents(self) -> list[str]:
        """Return a list of all registered agent names."""
        return list(self._agents.keys())


# ---------------------------------------------------------------------------
# Evaluation datasets for each agent
# ---------------------------------------------------------------------------
EVAL_DATASETS: dict[str, pd.DataFrame] = {
    "qa_agent": pd.DataFrame(
        [
            {
                "input": "What is Python?",
                "expected": "programming language",
                "category": "knowledge",
            },
            {
                "input": "What is MLflow used for?",
                "expected": "ML lifecycle",
                "category": "knowledge",
            },
            {"input": "Explain Docker.", "expected": "container", "category": "knowledge"},
            {"input": "What is LangChain?", "expected": "framework", "category": "knowledge"},
        ]
    ),
    "summarizer_agent": pd.DataFrame(
        [
            {
                "input": "Summarize: Python is a high-level programming language "
                "known for its readability and vast ecosystem of libraries.",
                "expected": "python",
                "category": "summarization",
            },
            {
                "input": "Summarize: MLflow is an open-source platform for managing "
                "the full machine learning lifecycle.",
                "expected": "mlflow",
                "category": "summarization",
            },
            {
                "input": "Summarize: Kubernetes automates container deployment, "
                "scaling, and management across clusters.",
                "expected": "container",
                "category": "summarization",
            },
        ]
    ),
    "code_helper_agent": pd.DataFrame(
        [
            {"input": "What is 25 * 4?", "expected": "100", "category": "math"},
            {"input": "Calculate 144 / 12.", "expected": "12", "category": "math"},
            {"input": "What is 7 + 8?", "expected": "15", "category": "math"},
        ]
    ),
}


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
@dataclass
class QualityGates:
    min_accuracy: float = 0.5
    max_avg_latency_s: float = 120.0


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
@mlflow.trace(name="invoke_agent")
def invoke_agent(agent, user_input: str) -> dict:
    """Invoke an agent and capture output and metadata."""
    start = time.time()
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        latency = time.time() - start
        answer = response["messages"][-1].content
        tool_calls = [m.name for m in response["messages"] if hasattr(m, "name") and m.name]
        return {
            "output": answer,
            "latency_s": round(latency, 2),
            "tool_calls": tool_calls,
            "error": None,
        }
    except Exception as e:
        return {
            "output": "",
            "latency_s": round(time.time() - start, 2),
            "tool_calls": [],
            "error": str(e),
        }


def evaluate_agent(
    agent_name: str,
    agent,
    dataset: pd.DataFrame,
    gates: QualityGates,
) -> dict:
    """Run evaluation for a single agent, score results, check quality gates."""
    results = []
    for _, row in dataset.iterrows():
        r = invoke_agent(agent, str(row["input"]))
        r["input"] = row["input"]
        r["expected"] = row["expected"]
        r["category"] = row["category"]
        # Score: does the output contain the expected keyword?
        r["correct"] = str(r["expected"]).lower() in str(r["output"]).lower()
        results.append(r)

    n = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    latencies = [r["latency_s"] for r in results]
    accuracy = round(correct_count / n, 3) if n else 0
    avg_latency = round(sum(latencies) / n, 2) if n else 0
    error_rate = round(sum(1 for r in results if r["error"]) / n, 3) if n else 0

    metrics = {
        "accuracy": accuracy,
        "avg_latency_s": avg_latency,
        "error_rate": error_rate,
        "total_tests": n,
    }

    # Quality gate check
    gate_passed = accuracy >= gates.min_accuracy and avg_latency <= gates.max_avg_latency_s

    return {
        "agent_name": agent_name,
        "metrics": metrics,
        "results": results,
        "gate_passed": gate_passed,
        "gate_details": {
            "accuracy": {
                "actual": accuracy,
                "threshold": gates.min_accuracy,
                "passed": accuracy >= gates.min_accuracy,
            },
            "avg_latency": {
                "actual": avg_latency,
                "threshold": gates.max_avg_latency_s,
                "passed": avg_latency <= gates.max_avg_latency_s,
            },
        },
    }


# ---------------------------------------------------------------------------
# Deployment pipeline
# ---------------------------------------------------------------------------
def deployment_decision(eval_result: dict) -> str:
    """Decide whether an agent should be deployed based on eval results."""
    if eval_result["gate_passed"]:
        return "approved"
    return "rejected"


# ---------------------------------------------------------------------------
# Platform orchestration
# ---------------------------------------------------------------------------
def run_platform() -> None:
    """Run the full agent platform demonstration."""

    # =======================================================================
    # Phase 1: Agent Registration
    # =======================================================================
    print("=" * 60)
    print("  Phase 1: Agent Registration")
    print("=" * 60)

    registry = AgentRegistry()

    agent_configs = [
        AgentConfig(
            name="qa_agent",
            description="General Q&A agent — answers knowledge questions",
            tools=[search_knowledge],
            system_prompt=(
                "You are a helpful Q&A assistant. Use the search_knowledge "
                "tool to find factual information. Answer concisely."
            ),
        ),
        AgentConfig(
            name="summarizer_agent",
            description="Summarizer agent — produces concise summaries",
            tools=[summarize_text],
            system_prompt=(
                "You are a summarization assistant. Use the summarize_text "
                "tool to produce concise summaries. Answer briefly."
            ),
        ),
        AgentConfig(
            name="code_helper_agent",
            description="Code helper agent — does calculations and lookups",
            tools=[calculate, search_knowledge],
            system_prompt=(
                "You are a code helper assistant. Use the calculate tool for "
                "math and search_knowledge for tech topics. Be precise."
            ),
        ),
    ]

    for config in agent_configs:
        registry.register(config)

    print(f"\n    Total agents registered: {len(registry.list_agents())}")

    # =======================================================================
    # Phase 2: Evaluate all agents
    # =======================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Automated Evaluation")
    print("=" * 60)

    gates = QualityGates()
    eval_results: dict[str, dict] = {}

    with mlflow.start_run(run_name="platform_evaluation") as parent_run:
        mlflow.set_tags(
            {
                "platform_version": "1.0.0",
                "phase": "evaluation",
                "num_agents": str(len(registry.list_agents())),
            }
        )

        for agent_name in registry.list_agents():
            print(f"\n  --- Evaluating: {agent_name} ---")
            agent = registry.get(agent_name)
            dataset = EVAL_DATASETS[agent_name]

            with mlflow.start_run(run_name=f"eval_{agent_name}", nested=True):
                config = registry.get_config(agent_name)
                mlflow.log_params(
                    {
                        "agent_name": agent_name,
                        "agent_version": config.version,
                        "agent_description": config.description[:250],
                        "num_tools": len(config.tools),
                        "num_test_cases": len(dataset),
                    }
                )

                result = evaluate_agent(agent_name, agent, dataset, gates)
                eval_results[agent_name] = result

                # Log metrics
                mlflow.log_metrics(result["metrics"])

                # Log gate results
                mlflow.set_tag("quality_gate_passed", str(result["gate_passed"]))
                for gate_name, gate_info in result["gate_details"].items():
                    mlflow.set_tag(
                        f"gate_{gate_name}",
                        "PASS" if gate_info["passed"] else "FAIL",
                    )

                # Print results
                m = result["metrics"]
                print(f"    Accuracy:    {m['accuracy']:.1%}")
                print(f"    Avg latency: {m['avg_latency_s']}s")
                print(f"    Error rate:  {m['error_rate']:.1%}")
                status = "PASSED" if result["gate_passed"] else "FAILED"
                print(f"    Gate status: {status}")

        # Log summary at parent level
        summary_rows = []
        for name, r in eval_results.items():
            summary_rows.append(
                {
                    "agent": name,
                    "accuracy": r["metrics"]["accuracy"],
                    "avg_latency_s": r["metrics"]["avg_latency_s"],
                    "error_rate": r["metrics"]["error_rate"],
                    "gate_passed": r["gate_passed"],
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_path = "/tmp/agent_platform_eval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, "evaluation")

        parent_run_id = parent_run.info.run_id

    # =======================================================================
    # Phase 3: Quality Gate Summary
    # =======================================================================
    print("\n" + "=" * 60)
    print("  Phase 3: Quality Gate Results")
    print("=" * 60)

    print(f"\n  {'Agent':<22} {'Accuracy':>10} {'Latency':>10} {'Gate':>8}")
    print("  " + "-" * 52)
    for name, r in eval_results.items():
        m = r["metrics"]
        gate = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  {name:<22} {m['accuracy']:>9.1%} {m['avg_latency_s']:>9.1f}s {gate:>8}")

    # =======================================================================
    # Phase 4: Deployment Pipeline
    # =======================================================================
    print("\n" + "=" * 60)
    print("  Phase 4: Deployment Pipeline")
    print("=" * 60)

    deployed_agent_name = None

    with mlflow.start_run(run_name="deployment_pipeline"):
        mlflow.set_tag("phase", "deployment")

        # Find the best agent that passed quality gates
        approved = []
        for name, r in eval_results.items():
            decision = deployment_decision(r)
            registry.set_status(name, decision)
            mlflow.set_tag(f"deploy_{name}", decision)
            print(f"\n    {name}: {decision.upper()}")
            if decision == "approved":
                approved.append((name, r["metrics"]["accuracy"]))

        if approved:
            # Deploy the agent with highest accuracy among approved
            approved.sort(key=lambda x: x[1], reverse=True)
            deployed_agent_name = approved[0][0]
            registry.set_status(deployed_agent_name, "deployed")
            mlflow.set_tag("deployed_agent", deployed_agent_name)
            mlflow.log_metric(
                "deployed_agent_accuracy",
                eval_results[deployed_agent_name]["metrics"]["accuracy"],
            )
            print(f"\n    --> Deployed: {deployed_agent_name} (accuracy: {approved[0][1]:.1%})")
        else:
            print("\n    No agents passed quality gates. Deployment skipped.")
            mlflow.set_tag("deployed_agent", "none")

    # =======================================================================
    # Phase 5: Production Inference
    # =======================================================================
    print("\n" + "=" * 60)
    print("  Phase 5: Production Inference Demo")
    print("=" * 60)

    if deployed_agent_name:
        agent = registry.get(deployed_agent_name)
        test_queries = [
            "What is LangGraph?",
            "Explain Kubernetes briefly.",
        ]

        with mlflow.start_run(run_name="production_inference"):
            mlflow.set_tags(
                {
                    "phase": "inference",
                    "deployed_agent": deployed_agent_name,
                }
            )

            for i, query in enumerate(test_queries, 1):
                print(f"\n    Query {i}: {query}")
                result = invoke_agent(agent, query)
                print(f"    Answer: {result['output'][:120]}")
                print(f"    Latency: {result['latency_s']}s")
                mlflow.log_metric(f"inference_latency_q{i}", result["latency_s"])
    else:
        print("\n    No agent deployed — skipping inference demo.")

    # =======================================================================
    # Final summary
    # =======================================================================
    print("\n" + "=" * 60)
    print("  Platform Summary")
    print("=" * 60)
    print(f"\n    Agents registered:  {len(registry.list_agents())}")
    print(f"    Agents evaluated:   {len(eval_results)}")
    passed_count = sum(1 for r in eval_results.values() if r["gate_passed"])
    print(f"    Gates passed:       {passed_count}/{len(eval_results)}")
    print(f"    Deployed agent:     {deployed_agent_name or 'none'}")
    print(f"\n    Evaluation run ID:  {parent_run_id}")
    print("    MLflow UI: http://127.0.0.1:5555")
    print("    Experiment: L3/M4_capstones/1_agent_platform")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  L3-M4.1 — Production AI Agent Platform (Capstone)")
    print("=" * 60)
    run_platform()


if __name__ == "__main__":
    main()
