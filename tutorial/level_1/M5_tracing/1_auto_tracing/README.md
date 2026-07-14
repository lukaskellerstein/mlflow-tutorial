# L1-5.1 — Automatic Tracing

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Tracing captures the full execution flow of an LLM chain or agent — every prompt template rendering, LLM call, and output parsing step — as a structured tree of **spans**. MLflow's auto-tracing makes this zero-effort: enable it once, and every LangChain invocation is recorded automatically.

## Prerequisites

- Completed: L1-M1 (Tracking), L1-M3.2 (LLM/GenAI Autologging)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### What is a Trace?

A **trace** is an end-to-end record of one operation — for example, a single `chain.invoke()` call. It captures what happened, how long each step took, and what data flowed through each component.

### What is a Span?

A **span** is a single step within a trace. Each component in a LangChain chain (prompt template, LLM, output parser) produces its own span. Spans have:

- **Name**: identifies the component (e.g., `ChatOpenAI`, `StrOutputParser`)
- **Type**: the category (`CHAIN`, `LLM`, `RETRIEVER`, etc.)
- **Inputs/Outputs**: the data that entered and left the component
- **Timing**: start time, end time, and duration
- **Parent**: spans nest in a tree — the chain is the root, and each step is a child

### Auto-Tracing

`mlflow.langchain.autolog()` patches LangChain so that every chain, agent, or retriever invocation automatically generates a trace. No decorators, no manual span creation — it just works.

## Step-by-Step

### Step 1: Enable Auto-Tracing

One line enables tracing for all LangChain operations:

```python
mlflow.langchain.autolog()
```

### Step 2: Simple Chain Tracing

Build a basic chain and invoke it. A trace is created automatically:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers to one sentence."),
    ("human", "{question}"),
])
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1", api_key="lm-studio",
    model="google/gemma-4-e4b", temperature=0.7,
)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"question": "What is MLflow?"})
```

The resulting trace contains spans for each step: prompt rendering, LLM call, and output parsing.

### Step 3: Multi-Step Chain

When you invoke multiple chains in sequence, each invocation creates its own trace. Within each trace, spans form a parent-child tree:

```
Root (RunnableSequence)
  ├── ChatPromptTemplate
  ├── ChatOpenAI
  └── StrOutputParser
```

### Step 4: Search Traces

Query traces programmatically with `mlflow.search_traces()`:

```python
traces = mlflow.search_traces(
    experiment_ids=[exp_id],
    max_results=5,
    return_type="list",
)

for trace in traces:
    print(trace.info.trace_id, trace.info.execution_duration_ms, "ms")
    for span in trace.data.spans:
        print(f"  {span.name} [{span.span_type}]")
```

## Running the Lesson

```bash
cd tutorial/level_1/M5_tracing/1_auto_tracing
uv sync
uv run python main.py
```

## Expected Output

In the terminal you will see:
- Part 1: A one-sentence answer to "What is MLflow?" and confirmation a trace was created
- Part 2: A summary and quiz questions, with two traces created
- Part 3: A listing of trace IDs, durations, and span trees

In the MLflow UI at http://127.0.0.1:5000:
- Navigate to the **Traces** tab
- Each trace shows the full span tree with timing
- Click a span to see its inputs and outputs

## Key Takeaways

- `mlflow.langchain.autolog()` enables zero-code tracing for LangChain
- A **trace** captures one end-to-end invocation; **spans** are the individual steps
- Spans nest in a parent-child tree matching the chain structure
- `mlflow.search_traces()` lets you query traces programmatically
- The MLflow UI Traces tab provides a visual timeline of all spans

## Next Steps

In **L1-5.2 — Manual Tracing**, you will learn how to create traces and spans by hand using `@mlflow.trace` and `mlflow.start_span()` — useful for custom code that is not part of a LangChain chain. In Level 2, we will explore advanced tracing with LangGraph state transitions and OpenTelemetry export.
