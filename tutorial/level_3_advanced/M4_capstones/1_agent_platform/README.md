# L3-M4.1 — Capstone: Production AI Agent Platform

**Level:** Expert
**Duration:** 4-6 hours

## Overview

This capstone brings together every concept from the tutorial into a mini production AI agent platform. You will build a system that registers multiple agents, evaluates them automatically, enforces quality gates, runs a deployment pipeline, and executes production inference — all with full MLflow observability.

This lesson integrates: experiment tracking (L1-M1), model management (L1-M2), evaluation (L1-M4, L2-M3), tracing (L1-M5, L2-M4), agent observability (L2-M5), and the evaluation pipeline (L3-M1.5).

## Prerequisites

- Completed: All Level 1 and Level 2 modules, plus L3-M1 (Agent Evaluation)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Architecture

The platform consists of five integrated subsystems:

```text
+-----------------------------------------------------------+
|                   Agent Platform                          |
+-----------------------------------------------------------+
|                                                           |
|  1. Agent Registry                                        |
|     +-- QA Agent (search_knowledge tool)                  |
|     +-- Summarizer Agent (summarize_text tool)            |
|     +-- Code Helper Agent (calculate + search tools)      |
|                                                           |
|  2. Evaluation System                                     |
|     +-- Per-agent evaluation datasets                     |
|     +-- Keyword-match accuracy scoring                    |
|     +-- Quality gate enforcement                          |
|                                                           |
|  3. Tracing & Monitoring                                  |
|     +-- @mlflow.trace on every agent invocation           |
|     +-- Latency, tool calls, error tracking               |
|     +-- Nested MLflow runs per evaluation                 |
|                                                           |
|  4. Deployment Pipeline                                   |
|     +-- Evaluate -> Gate check -> Approve/Reject          |
|     +-- Deploy best approved agent                        |
|     +-- Log all deployment decisions                      |
|                                                           |
|  5. Production Inference                                  |
|     +-- Run queries against deployed agent                |
|     +-- Track inference latency in MLflow                 |
|                                                           |
+-----------------------------------------------------------+
           |               |               |
     MLflow Tracking   MLflow Tracing  MLflow Artifacts
```

### Data flow

```text
Register agents --> Evaluate each --> Score results --> Check quality gates
                                                              |
                                                     Pass?  /   \ Fail?
                                                           /     \
                                                   Approve      Reject
                                                      |
                                              Deploy best agent
                                                      |
                                             Production inference
```

## Concepts

### Agent Registry

The `AgentRegistry` class manages multiple agents with different capabilities. Each agent is a LangGraph ReAct agent (`create_react_agent`) configured with specific tools and a system prompt. The registry tracks agent status through the lifecycle: registered, approved, rejected, or deployed.

### Automated Evaluation

Each agent has a dedicated evaluation dataset — a pandas DataFrame with inputs, expected keywords, and categories. The evaluation system runs every test case through the agent, scores results using keyword matching, and computes aggregate metrics (accuracy, latency, error rate).

### Quality Gates

Quality gates enforce minimum standards before deployment. The platform checks:
- **Accuracy threshold** (default >= 50%) — the agent must answer enough questions correctly
- **Latency threshold** (default <= 120s) — the agent must respond fast enough

Only agents that pass all gates are eligible for deployment.

### Deployment Pipeline

The pipeline automates the deploy decision:
1. Evaluate all registered agents
2. Check quality gates for each
3. Approve agents that pass, reject those that fail
4. Deploy the best approved agent (highest accuracy)
5. Log every decision to MLflow for audit

### Tracing

Every agent invocation is traced with `@mlflow.trace`, capturing inputs, outputs, latency, and tool calls. Evaluation runs use nested MLflow runs so each agent's results appear as child runs under the parent evaluation run.

## Step-by-Step

### Step 1: Agent Registration

Three agents are registered, each with different tools and system prompts:

```python
registry = AgentRegistry()
registry.register(
    AgentConfig(
        name="qa_agent",
        description="General Q&A agent",
        tools=[search_knowledge],
        system_prompt="You are a helpful Q&A assistant...",
    )
)
```

Each call to `register()` creates a LangGraph ReAct agent using `create_react_agent` with `ChatOpenAI(model="google/gemma-4-26b-a4b")`.

### Step 2: Automated Evaluation

Each agent is evaluated against its own dataset:

```python
result = evaluate_agent(agent_name, agent, dataset, gates)
```

The `evaluate_agent` function runs every test case, scores results, and checks quality gates. All metrics are logged to MLflow as a nested run under the parent evaluation run.

### Step 3: Quality Gate Check

After evaluation, the platform prints a summary table showing which agents passed:

```text
Agent                  Accuracy    Latency     Gate
----------------------------------------------------
qa_agent                  75.0%       5.2s     PASS
summarizer_agent          66.7%       4.1s     PASS
code_helper_agent         33.3%      12.3s     FAIL
```

### Step 4: Deployment Pipeline

The pipeline approves agents that passed quality gates and deploys the one with the highest accuracy:

```python
decision = deployment_decision(eval_result)  # "approved" or "rejected"
registry.set_status(name, decision)
```

### Step 5: Production Inference

The deployed agent handles live queries, with each call traced in MLflow:

```python
result = invoke_agent(agent, "What is LangGraph?")
```

## Running the Lesson

```bash
cd tutorial/level_3/M5_capstones/1_agent_platform
uv sync
uv run python main.py
```

## Expected Output

The script runs five phases with clear output for each:

```text
==============================================================
  L3-M4.1 — Production AI Agent Platform (Capstone)
==============================================================
  Phase 1: Agent Registration
    Registered: qa_agent v1.0.0
    Registered: summarizer_agent v1.0.0
    Registered: code_helper_agent v1.0.0

  Phase 2: Automated Evaluation
  --- Evaluating: qa_agent ---
    Accuracy:    75.0%
    Avg latency: 5.2s
    Gate status: PASSED

  Phase 3: Quality Gate Results
  Agent                  Accuracy    Latency     Gate
  ...

  Phase 4: Deployment Pipeline
    qa_agent: APPROVED
    --> Deployed: qa_agent (accuracy: 75.0%)

  Phase 5: Production Inference Demo
    Query 1: What is LangGraph?
    Answer: LangGraph builds stateful multi-actor...
    Latency: 4.3s

  Platform Summary
    Agents registered:  3
    Deployed agent:     qa_agent
```

In the MLflow UI at <http://127.0.0.1:5555>, you will see:
- **Experiment**: `L3/M4_capstones/1_agent_platform`
- **platform_evaluation** run with nested child runs for each agent
- **deployment_pipeline** run with deployment decision tags
- **production_inference** run with inference latency metrics
- Traces for every agent invocation under the Traces tab

## Key Takeaways

- An **agent registry** provides a centralized way to manage multiple agents with different capabilities, making it easy to add, evaluate, and deploy agents.
- **Automated evaluation** with quality gates ensures only agents meeting minimum standards reach production — this is the foundation of reliable AI deployment.
- **Full tracing** with `@mlflow.trace` gives visibility into every agent call, making debugging and performance analysis straightforward.
- A **deployment pipeline** that logs every decision to MLflow creates an auditable record of what was deployed and why.
- MLflow's **nested runs** pattern naturally maps to platform operations: one parent run for the evaluation cycle, with child runs for each agent.

## Next Steps

Continue to L3-M4.2 (Agent Framework Benchmark) to build a standardized benchmark comparing agent frameworks — LangChain/LangGraph, Claude Agent SDK, and custom PyFunc agents — using the evaluation patterns from this capstone.
