# L3-2.3 — DeepAgents Integration: Multi-Agent Orchestration

**Level:** Expert
**Duration:** ~45 minutes

## Overview

This lesson demonstrates multi-agent orchestration patterns inspired by DeepAgents (LangChain-AI's multi-agent framework) with full MLflow tracing integration. You will build an orchestrator that decomposes tasks, delegates to specialist agents, and tracks the entire communication flow — then compare it against a single-agent baseline.

## Prerequisites

- Completed: L3-M2.1 (Claude Agent SDK), L3-M2.2 (Codex SDK)
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Multi-Agent Orchestration

Complex tasks often exceed the capability of a single LLM call. Multi-agent systems address this by decomposing work across specialized agents:

- **Orchestrator**: Breaks down the task, assigns sub-tasks, aggregates results.
- **Specialists**: Each agent has a narrow role (research, analysis, writing) with a focused system prompt.
- **Message passing**: Agents communicate through structured handoffs — each agent's output becomes the next agent's context.

### Why Trace Multi-Agent Systems?

Without tracing, multi-agent pipelines are opaque. MLflow tracing lets you:

- See the full delegation chain (who called whom, with what input).
- Measure per-agent latency and identify bottlenecks.
- Track coordination overhead (time spent on orchestration vs. actual agent work).
- Compare multi-agent vs. single-agent approaches on the same task.

### Custom Integration Pattern

DeepAgents (and similar frameworks without native MLflow support) require manual tracing. The pattern is:

1. Decorate agent entry points with `@mlflow.trace`.
2. Use `mlflow.start_span()` inside each agent to capture inputs/outputs.
3. Nest spans to reflect the orchestration hierarchy.
4. Log inter-agent messages as artifacts for post-hoc analysis.

## Step-by-Step

### Step 1: Define Specialist Agents

Each agent extends a `BaseAgent` class with its own system prompt. The `invoke()` method is decorated with `@mlflow.trace` and wraps the LLM call in an MLflow span:

```python
class BaseAgent:
    @mlflow.trace
    def invoke(self, task: str) -> AgentResult:
        with mlflow.start_span(name=f"agent_{self.name}") as span:
            span.set_inputs({"task": task, "agent": self.name})
            response = self.llm.invoke(messages)
            span.set_outputs({"output_preview": response.content[:300]})
```

Three specialists are created: `ResearchAgent`, `AnalysisAgent`, and `WriterAgent`.

### Step 2: Build the Orchestrator

The `OrchestratorAgent` decomposes tasks into a pipeline (research -> analysis -> writing) and delegates to each specialist sequentially. Each handoff is logged as an `AgentMessage`:

```python
class OrchestratorAgent:
    @mlflow.trace
    def run(self, task: str) -> dict:
        with mlflow.start_span(name="orchestration_pipeline") as root_span:
            subtasks = self.decompose_task(task)
            for step in subtasks:
                result = self.specialists[step["agent"]].invoke(agent_task)
                self.handoff_count += 1
```

### Step 3: Track Communication

Every message between agents is recorded with sender, receiver, content, and timestamp. The full log is saved as a JSON artifact:

```python
mlflow.log_artifact("agent_communication_log.json", artifact_path="communication")
```

### Step 4: Run and Measure

The pipeline runs on a complex task ("Research and summarize the benefits of microservices architecture"), then a single generalist agent runs the same task for comparison.

### Step 5: Compare Approaches

Key metrics are logged for both approaches: `agents_used`, `handoffs`, `total_duration_s`, `token_estimate`, and `coordination_overhead_s`. A comparison CSV is saved as an MLflow artifact.

## Running the Lesson

```bash
cd tutorial/level_3/M2_custom_integrations/3_deepagents
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-2.3 — DeepAgents: Multi-Agent Orchestration + MLflow
============================================================

--- Part 1: Multi-agent orchestration pipeline ---
    [orchestrator] Decomposing task...
    [orchestrator] -> Delegating to researcher
    [researcher] Completed in 4.2s
    [orchestrator] -> Delegating to analyst
    [analyst] Completed in 3.8s
    [orchestrator] -> Delegating to writer
    [writer] Completed in 3.5s

--- Part 2: Inter-agent communication log ---
    orchestrator -> researcher        | Research the following topic: Research...
    researcher -> orchestrator        | Microservices architecture involves...
    orchestrator -> analyst           | Analyze the research findings...
    analyst -> orchestrator           | Key patterns identified...
    orchestrator -> writer            | Write a clear summary...
    writer -> orchestrator            | ## Benefits of Microservices...

--- Part 3: Multi-agent coordination metrics ---
    agents_used                         = 3
    handoffs                            = 3
    total_steps                         = 3
    total_duration_s                    = 11.5
    coordination_overhead_s             = 0.0

--- Part 4: Single-agent baseline comparison ---
    Single agent completed in 5.1s

--- Part 5: Multi-agent vs single-agent comparison ---
 approach  agents_used  handoffs  duration_s  ...
multi_agent          3         3       11.5  ...
single_agent         1         0        5.1  ...
```

In the MLflow UI you will see:
- **Traces** with nested spans showing the orchestrator -> specialist hierarchy.
- **Three runs**: `multi_agent_orchestration`, `single_agent_baseline`, and `approach_comparison`.
- **Artifacts**: communication log (JSON) and comparison table (CSV).

## Key Takeaways

- Multi-agent orchestration decomposes complex tasks across specialists with focused prompts, improving output quality at the cost of latency.
- Wrapping agent calls with `@mlflow.trace` and `mlflow.start_span()` gives full visibility into the delegation chain.
- Logging inter-agent messages as artifacts enables post-hoc analysis of coordination patterns.
- Coordination overhead (time spent on orchestration vs. agent work) is a key metric for production multi-agent systems.
- Comparing multi-agent vs. single-agent baselines in MLflow helps justify the added complexity.

## Next Steps

Continue to **L3-M2.4 — Custom Autolog** to learn how to build a reusable autolog plugin that automatically instruments any agent framework with MLflow tracing.
