# L2-5.2 — LangGraph Agent Observability

**Level:** Practitioner
**Duration:** 2 hours

## Overview

This lesson builds a research assistant agent using LangGraph's `StateGraph` and instruments it with MLflow auto-tracing. You will learn how to observe state transitions, conditional edge decisions, retry loops, and per-node execution timing — the essential debugging toolkit for stateful agent workflows.

## Prerequisites

- Completed: L1-M5 (Tracing basics), L2-5.1 (LangChain Agent Tracking)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### LangGraph State Machines

LangGraph models agents as directed graphs where:

- **Nodes** are functions that transform state
- **Edges** define the execution flow between nodes
- **Conditional edges** route execution based on state values (e.g., quality check pass/fail)
- **State** is a typed dictionary that flows through the graph, accumulating results

This is fundamentally different from simple chain-based LangChain agents. The graph structure makes execution non-linear — loops, branches, and conditional routing are first-class concepts.

### Why Observability Matters for Graph Agents

When an agent can loop, branch, and retry, understanding *what actually happened* during execution becomes critical:

- **Did the quality check trigger a retry?** The trace shows conditional edge decisions.
- **Which node is the bottleneck?** Per-node durations reveal latency hotspots.
- **How many loops occurred?** Span counts on repeated nodes quantify retry behavior.
- **What state did each node see?** Span inputs/outputs capture the evolving state.

### MLflow Auto-Tracing for LangGraph

`mlflow.langchain.autolog()` automatically instruments LangGraph executions. Every node invocation, LLM call, and state transition generates a span. The resulting trace tree mirrors the graph execution path, making it straightforward to reconstruct what happened.

## Step-by-Step

### Step 1: Define the Agent State

The state carries messages, research notes, the current processing step, a quality flag, and a retry counter:

```python
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str
    current_step: str
    quality_pass: bool
    retry_count: int
```

### Step 2: Build Graph Nodes

Four nodes form the research pipeline:

1. **analyze_query** — Uses the LLM to break down the user's question into research topics
2. **search_knowledge** — Searches a local knowledge base (no external APIs needed)
3. **synthesize_answer** — Uses the LLM to combine research notes into a coherent answer
4. **quality_check** — Uses the LLM to judge whether the answer passes a quality bar

### Step 3: Add Conditional Routing

The key pattern is the retry loop. After `quality_check`, a conditional edge decides:

- If `quality_pass` is True or `retry_count >= 2`: route to `END`
- Otherwise: route back to `search_knowledge` for another attempt

```python
graph.add_conditional_edges("quality_check", should_retry)
```

This creates a cycle in the graph — the trace will show repeated node executions when retries occur.

### Step 4: Analyze Traces Programmatically

After each invocation, we retrieve the trace and extract:

- **Node execution order** — which nodes ran and in what sequence
- **Per-node durations** — how long each node took (in milliseconds)
- **Retry loops** — how many times `search_knowledge` was re-entered
- **Total span count** — overall execution complexity

### Step 5: Log Aggregate Metrics

All metrics are logged to an MLflow run for comparison across queries:

- `total_nodes_visited` — sum of all spans across queries
- `total_retry_loops` — total quality-check retries
- `avg_trace_duration_ms` — average execution time

## Running the Lesson

```bash
cd tutorial/level_2/M5_agent_observability/2_langgraph_agents
uv sync
uv run python main.py
```

## Expected Output

The console will show:

1. Each query being processed through the graph nodes
2. Per-query trace analysis with node execution order and durations
3. Retry loop detection (0 or more retries per query)
4. Aggregate metrics across all queries

In the **MLflow UI** (http://127.0.0.1:5000):

- Navigate to the `L2/M5_agent_observability/2_langgraph_agents` experiment
- Open the `langgraph_agent_observability` run
- Click **Traces** to see the full execution graph for each query
- Expand spans to see LLM inputs/outputs at each node
- Look for repeated `search_knowledge` spans indicating retry loops

## Key Takeaways

- `mlflow.langchain.autolog()` instruments LangGraph executions automatically — every node and LLM call generates a span
- Conditional edges create non-linear traces; retry loops appear as repeated spans in the trace tree
- Per-node timing analysis reveals bottlenecks (LLM calls typically dominate)
- Programmatic trace analysis with `mlflow.get_trace()` and `trace.data.spans` enables automated quality monitoring
- Logging aggregate agent metrics (nodes visited, retries, duration) to MLflow runs enables cross-query and cross-agent comparison

## Next Steps

In **L2-5.3 (Multi-Agent Systems)**, you will build a multi-agent graph where multiple specialized agents collaborate, and trace inter-agent communication and handoffs.
