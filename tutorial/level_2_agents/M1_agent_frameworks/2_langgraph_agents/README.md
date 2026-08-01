# L2-M1.2 — LangGraph Agents

**Level:** Practitioner
**Duration:** 1.5 hours

## Overview

LangGraph models agents as directed graphs where data flows through nodes connected by edges -- including conditional edges that route execution based on runtime values. This lesson builds two LangGraph workflows: a simple classifier with conditional routing, and a research agent with a quality-check retry loop. Both are auto-traced by MLflow, and we analyze the resulting traces programmatically.

## Prerequisites

- Completed: L2-M1.1 (LangChain Agents)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` loaded

## Concepts

### LangGraph State Machines

A LangGraph `StateGraph` is a directed graph where:
- **State** is a `TypedDict` shared across all nodes
- **Nodes** are functions that receive state and return partial updates
- **Edges** define execution order; conditional edges route based on state values
- **Cycles** are first-class -- enabling retry loops and iterative refinement

### MLflow Auto-Tracing

`mlflow.langchain.autolog()` hooks into LangGraph's callback system. Every graph invocation produces a trace with nested spans:

| Span Level | What It Records |
|---|---|
| Root span | Graph invocation, total duration, final state |
| Node spans | Each node's inputs/outputs, execution time |
| LLM spans | Model calls -- prompt, response, token counts |

### Retry Loops

When a quality check fails, a conditional edge routes back to an earlier node for another attempt. The trace shows repeated spans for the retried nodes, making retry behavior visible.

## Step-by-Step

### Step 1: Simple Workflow (Part 1)

A four-node graph: `classify_input` decides whether a request is simple or complex, then routes to `process_simple` or `process_complex`:

```python
builder.add_conditional_edges("classify_input", route_by_complexity)
```

A "Hello" greeting routes to `process_simple`; a detailed question routes to `process_complex`.

### Step 2: Research Agent (Part 2)

A more complex graph with a retry loop:

1. **analyze_query** -- break down the question into research topics
2. **search_knowledge** -- search a local knowledge base
3. **synthesize_answer** -- combine findings into a coherent answer
4. **quality_check** -- LLM judges if the answer passes

After `quality_check`, a conditional edge either ends or loops back to `search_knowledge`:

```python
def should_retry(state) -> Literal["search_knowledge", "__end__"]:
    if state["quality_pass"] or state["retry_count"] >= 2:
        return "__end__"
    return "search_knowledge"
```

### Step 3: Trace Analysis

After each invocation, retrieve the trace and analyze spans:

```python
trace = mlflow.get_trace(trace_id, flush=True)
for span in trace.data.spans:
    duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
```

### Step 4: Aggregate Metrics

Log total spans, retry loops, and average duration to an MLflow run for cross-query comparison.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/2_langgraph_agents
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Simple Workflow with Conditional Routing
============================================================
  --- Input 1: "Hello, how are you?" ---
  Classified as: simple
  Response: Hello! I'm doing great...
  Time: 3.21s

  --- Input 2: "Explain the difference between REST and GraphQL APIs." ---
  Classified as: complex
  Response: Great question! Here are the key differences...
  Time: 6.54s

============================================================
Part 2: Research Agent with Quality-Check Retry Loop
============================================================
  --- Query 1: What is MLflow tracing and how does it help... ---
  Answer: MLflow tracing captures the full execution path...

  ==================================================
  Trace: Query 1  |  ID: tr-abc123...
  Duration: 8500 ms  |  Spans: 12
    1. analyze_query                  (2100 ms)
    2. search_knowledge               (50 ms)
    3. synthesize_answer              (2800 ms)
    4. quality_check                  (1900 ms)

============================================================
  Aggregate Metrics
============================================================
  Total spans: 24
  Retry loops: 0
  Avg duration: 8250.0 ms
```

In the MLflow UI, navigate to the experiment and open the Traces tab to see the full span tree for each invocation, including which conditional branch was taken and any retry loops.

## Key Takeaways

- `mlflow.langchain.autolog()` traces LangGraph workflows automatically -- every node and LLM call generates a span
- Conditional edges create non-linear traces; the spans reveal which path was taken at runtime
- Retry loops appear as repeated spans in the trace tree, making quality-check cycles visible
- Per-node timing analysis reveals bottlenecks (LLM calls typically dominate)
- Programmatic trace analysis with `mlflow.get_trace()` enables automated quality monitoring
- Aggregate metrics (spans, retries, duration) logged to MLflow runs enable cross-query comparison

## Next Steps

In L2-M1.3 (Multi-Agent Systems), you will build a multi-agent graph where specialized agents collaborate, and trace inter-agent communication and handoffs.
