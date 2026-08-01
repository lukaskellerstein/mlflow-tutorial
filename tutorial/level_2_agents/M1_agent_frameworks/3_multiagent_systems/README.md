# L2-5.3 — Multi-Agent Systems

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

This lesson builds a multi-agent system using LangGraph's supervisor pattern and instruments it with MLflow tracing. You will see how three specialized agents (Researcher, Writer, Reviewer) coordinate through a supervisor, how inter-agent handoffs appear in traces, and how to extract per-agent and pipeline-level metrics.

## Prerequisites

- Completed: L1-M5.1 (Auto Tracing), L2-M5.2 (LangGraph Agents)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Multi-Agent Patterns

Multi-agent systems split complex tasks across specialized agents. Common patterns include:

| Pattern | Description | When to use |
|---------|-------------|-------------|
| **Collaboration** | Agents work together as peers, passing messages freely | Creative brainstorming, consensus tasks |
| **Supervision** | A supervisor routes work to specialists and decides next steps | Structured pipelines, quality gates |
| **Swarm** | Agents dynamically hand off to each other based on context | Customer support, triage workflows |

This lesson uses the **supervision** pattern: a supervisor (implemented as a conditional edge) decides whether the pipeline is complete or needs revision.

### The Pipeline

```
START -> Researcher -> Writer -> Reviewer --(pass)--> END
                                    |
                                    +--(fail)--> Writer (revision loop, max 2)
```

- **Researcher**: Takes a topic, produces 3-5 bullet points of key facts
- **Writer**: Takes research notes (or feedback), writes a 2-3 paragraph summary
- **Reviewer**: Reviews the draft, returns PASS or FAIL with feedback
- **Supervisor** (conditional edge): Routes back to Writer on failure, up to 2 revision cycles

### Tracing Multi-Agent Handoffs

When `mlflow.langchain.autolog()` is enabled, LangGraph graph invocations are automatically traced. Each agent node produces spans within the trace, showing:

- The full execution flow through the graph
- Input/output at each node
- Duration of each agent call
- The revision loop (if the reviewer rejects a draft)

## Step-by-Step

### Step 1: Define the Shared State

All agents share a `PipelineState` that carries the topic, research notes, draft, review feedback, and metrics:

```python
class PipelineState(TypedDict):
    topic: str
    research_notes: str
    draft: str
    review_feedback: str
    review_passed: bool
    revision_count: int
    agent_durations: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]
```

The `Annotated[list, operator.add]` fields accumulate values across nodes (each node appends, rather than overwrites).

### Step 2: Build Specialized Agent Nodes

Each agent is a function that receives the state, calls an LLM chain, and returns updated state fields. For example, the Researcher:

```python
def researcher_node(state: PipelineState) -> dict:
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="google/gemma-4-e4b", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([...])
    chain = prompt | llm | StrOutputParser()
    notes = chain.invoke({"topic": state["topic"]})
    return {"research_notes": notes, ...}
```

### Step 3: Wire the Supervisor Graph

The supervisor pattern uses `add_conditional_edges` to route after the Reviewer:

```python
workflow = StateGraph(PipelineState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges("reviewer", supervisor_router)
```

The `supervisor_router` function returns `"writer"` to loop back for revision or `"__end__"` to finish.

### Step 4: Analyze Traces

After running the pipeline, we query MLflow traces to see the full execution flow:

```python
traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    max_results=5,
    return_type="list",
    flush=True,
)
```

Each trace shows spans for every node traversal, including repeated visits to the Writer node during revision loops.

### Step 5: Log Pipeline Metrics

We log per-topic and aggregated metrics to an MLflow run:

- Per-agent duration
- Revision count and handoff count
- Review pass rate
- Average pipeline duration

## Running the Lesson

```bash
cd tutorial/level_2/M5_agent_observability/3_multiagent_systems
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Enabling LangChain auto-tracing: mlflow.langchain.autolog()
============================================================

============================================================
Topic: The impact of large language models on software development
============================================================

  [Researcher] Researching topic...
  [Researcher] Done in 3.2s

  [Writer] Writing...
  [Writer] Done in 4.1s

  [Reviewer] Reviewing draft...
  [Reviewer] Verdict: PASS (2.0s)

  Pipeline completed in 9.3s
  Revisions: 1
  Review passed: True

...

============================================================
Analyzing Traces
============================================================
  Found 2 trace(s)

  --- Trace 1 ---
  Trace ID:  tr-abc123...
  Duration:  9300 ms (9.3s)
  Spans (8):
    (root)   LangGraph  [CHAIN]  9300ms
             researcher  [CHAIN]  3200ms
             ChatOpenAI  [LLM]   3100ms
             writer      [CHAIN]  4100ms
             ChatOpenAI  [LLM]   4000ms
             reviewer    [CHAIN]  2000ms
             ChatOpenAI  [LLM]   1900ms

============================================================
Logging Multi-Agent Metrics
============================================================
  [topic_1] duration=9.3s, revisions=1, handoffs=2, passed=True
  [topic_2] duration=15.1s, revisions=2, handoffs=4, passed=True

  Averages: duration=12.2s, revisions=1.5, pass_rate=100%
```

In the MLflow UI Traces tab, you will see a trace for each topic with nested spans showing the full agent pipeline, including any revision loops.

## Key Takeaways

- **Supervisor pattern**: A conditional edge after the Reviewer acts as the supervisor, deciding whether to loop back for revision or finish.
- **Shared state**: LangGraph's `TypedDict` state with `Annotated[list, operator.add]` lets agents accumulate data without overwriting each other.
- **Automatic tracing**: `mlflow.langchain.autolog()` traces every node invocation in the graph, capturing inputs, outputs, and timing.
- **Handoff visibility**: Traces show the complete flow between agents, making it easy to spot bottlenecks or excessive revision loops.
- **Pipeline metrics**: Logging per-agent durations, handoff counts, and pass rates helps monitor multi-agent system quality over time.

## Next Steps

In Level 3 (L3-M1), you will build on these patterns to implement advanced agent evaluation, including custom scorers for agent-specific behaviors, architecture comparison across frameworks, and production evaluation pipelines.
