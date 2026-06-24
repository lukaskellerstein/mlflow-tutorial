# L2-4.1 — Tracing LangGraph State Machines

**Level:** Practitioner
**Duration:** 1.5 hours

## Overview

LangGraph workflows are stateful graphs where data flows through nodes connected by edges — including conditional edges that route execution based on runtime values. Understanding *which* path a request took and *how long* each node spent is critical for debugging and optimizing these workflows. This lesson shows how MLflow auto-tracing captures every node execution, state transition, and conditional routing decision in a LangGraph `StateGraph`, and how to retrieve and analyze that trace data programmatically.

## Prerequisites

- Completed: L1-M5.1 (Auto Tracing), L1-M5.2 (Manual Tracing)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### LangGraph State Machines

A LangGraph `StateGraph` is a directed graph where:

- **State** is a `TypedDict` shared across all nodes.
- **Nodes** are Python functions that receive the current state and return a partial state update.
- **Edges** define the execution order. *Conditional edges* use a routing function to pick the next node at runtime.

### MLflow Auto-Tracing for LangGraph

When you call `mlflow.langchain.autolog()`, MLflow hooks into the LangChain/LangGraph callback system. Every invocation of the compiled graph produces a **trace** containing a tree of **spans** — one for the overall graph run, and nested spans for each node execution. This happens automatically; no code changes to the graph are required.

### What the Traces Capture

| Span level | What it records |
|---|---|
| Root span | Graph invocation, total duration, final state |
| Node spans | Each node's inputs and outputs, execution time |
| LLM spans | Model calls inside nodes — prompt, response, token counts |

Conditional edge decisions are visible through *which* node spans appear in a trace: if the classifier routes to `process_simple`, the trace will contain that span but not `process_complex`.

## Step-by-Step

### Step 1: Define the State and Nodes

We define a `GraphState` TypedDict with fields that accumulate as the workflow proceeds: `input_text`, `complexity`, `processed_text`, and `final_response`. Each node is a plain function that reads what it needs from the state and returns a partial update.

```python
class GraphState(TypedDict):
    messages: list
    input_text: str
    complexity: str        # "simple" or "complex"
    processed_text: str
    final_response: str
```

Four nodes form the processing pipeline:

1. **classify_input** — asks the LLM to label the input as SIMPLE or COMPLEX.
2. **process_simple** — generates a brief, direct answer.
3. **process_complex** — generates a thorough, structured answer.
4. **generate_response** — polishes the processed text into a final reply.

### Step 2: Wire Conditional Edges

After classification, a routing function inspects `state["complexity"]` and returns the name of the next node:

```python
def route_by_complexity(state) -> Literal["process_simple", "process_complex"]:
    if state["complexity"] == "simple":
        return "process_simple"
    return "process_complex"

builder.add_conditional_edges("classify_input", route_by_complexity)
```

This means the trace for a "Hello" greeting will show `classify_input -> process_simple -> generate_response`, while a request for a detailed explanation will show `classify_input -> process_complex -> generate_response`.

### Step 3: Invoke and Trace

With `mlflow.langchain.autolog()` enabled, every call to `graph.invoke(...)` is automatically traced. We invoke the graph with three different inputs to exercise both branches:

```python
test_inputs = [
    "Hello, how are you?",                                  # simple
    "Explain the difference between REST and GraphQL APIs.", # complex
    "What is 2 + 2?",                                       # simple
]
```

### Step 4: Retrieve and Analyze Traces

After execution we use `mlflow.search_traces()` to pull every trace for the experiment and inspect:

- **Span tree** — which nodes ran and in what order.
- **Conditional branch taken** — simple vs. complex path.
- **Span durations** — wall-clock time per node, highlighting bottlenecks.

```python
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    return_type="list",
)
for trace in traces:
    for span in trace.data.spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
        print(f"{span.name}: {duration_ms:.1f} ms")
```

## Running the Lesson

```bash
cd tutorial/level_2/M4_advanced_tracing/1_langgraph_tracing
uv sync
uv run python main.py
```

## Expected Output

In the terminal you will see three invocations with their classifications and responses, followed by a trace analysis section:

```
Step 2: Invoking the graph with different inputs

  --- Input 1: "Hello, how are you?" ---
  Classified as: simple
  Response preview: Hello! I'm doing great, thank you for asking...
  Wall-clock time: 3.21s

  --- Input 2: "Explain the difference between REST and GraphQL APIs." ---
  Classified as: complex
  Response preview: Great question! Here are the key differences...
  Wall-clock time: 6.54s

  --- Input 3: "What is 2 + 2?" ---
  Classified as: simple
  Response preview: The answer is 4...
  Wall-clock time: 2.89s

Step 3: Analyzing traces from MLflow

  Found 3 trace(s) in the experiment.
  ...
    Visited nodes: classify_input -> process_simple -> generate_response
    Conditional edge: classify_input --> process_simple  (SIMPLE path)
```

In the **MLflow UI** (http://127.0.0.1:5000), navigate to the experiment and open the Traces tab. Click on any trace to see the full span tree with:

- Inputs and outputs for every node
- LLM call details (prompt and response) nested under each node
- Timing waterfall showing where time was spent

## Key Takeaways

- `mlflow.langchain.autolog()` traces LangGraph workflows automatically — no instrumentation code needed inside nodes.
- Each graph invocation produces a trace with nested spans for every visited node and its internal LLM calls.
- Conditional edge decisions are visible by inspecting which node spans appear in the trace.
- `mlflow.search_traces()` with `return_type="list"` gives you `Trace` objects whose `data.spans` list supports programmatic analysis of execution paths and durations.
- Span duration analysis reveals bottlenecks — in LLM-heavy graphs, the model call inside a node typically dominates.

## Next Steps

In **L2-4.2 (Tracing Temporal.io Workflows)** we move beyond single-process graphs to distributed workflow orchestration, integrating MLflow tracing with Temporal.io activities and tracking long-running, retriable workflows.
