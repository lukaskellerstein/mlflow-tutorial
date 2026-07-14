# L1-5.2 — Manual Tracing

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

MLflow provides two manual tracing APIs that give you fine-grained control over how your code is instrumented: the `@mlflow.trace` decorator and the `mlflow.start_span()` context manager. This lesson shows how to use both, and how to combine them with LangChain's automatic tracing for a unified view.

## Prerequisites

- Completed: L1-5.1 (Auto Tracing)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why Manual Tracing?

Auto-tracing (L1-5.1) captures LangChain and framework calls automatically. But what about your own code — data processing, validation, orchestration logic? Manual tracing lets you instrument any Python function so the full picture appears in MLflow's trace viewer.

### Two APIs for Manual Tracing

1. **`@mlflow.trace` decorator** — the simplest approach. Decorate a function and MLflow automatically captures its inputs, outputs, and timing as a span. Nested decorated functions create parent-child spans automatically.

2. **`mlflow.start_span()` context manager** — gives full control. You explicitly set inputs, outputs, and custom attributes on each span. Useful when you need to instrument a block of code that is not a standalone function, or when you want to attach metadata.

### Key Span Methods

When using `start_span()`, the span object exposes:
- `span.set_inputs(dict)` — record what went into this step
- `span.set_outputs(value)` — record what came out
- `span.set_attributes(dict)` — attach custom key-value metadata
- `span.set_attribute(key, value)` — attach a single attribute

### Combining Auto and Manual Tracing

You can mix both approaches in a single trace. For example, wrap your orchestration function with `@mlflow.trace`, and inside it call a LangChain chain that is auto-traced. The result is a single trace tree with both your manual spans and the auto-generated LangChain spans.

## Step-by-Step

### Step 1: @mlflow.trace Decorator

Decorate functions to create spans automatically. When one decorated function calls another, MLflow nests the spans:

```python
@mlflow.trace(name="validate_text")
def validate_text(text: str) -> str:
    return text.strip()

@mlflow.trace(name="process_pipeline")
def process_pipeline(text: str) -> dict:
    validated = validate_text(text)  # child span
    return {"result": validated}
```

The `process_pipeline` span will contain `validate_text` as a child span.

### Step 2: mlflow.start_span() Context Manager

Use `start_span()` for explicit control over span inputs, outputs, and attributes:

```python
with mlflow.start_span(name="batch_analysis") as root_span:
    root_span.set_inputs({"texts": texts})

    for i, text in enumerate(texts):
        with mlflow.start_span(name=f"analyze_item_{i}") as child:
            child.set_inputs({"text": text})
            child.set_attributes({"position": i})
            result = analyze(text)
            child.set_outputs(result)

    root_span.set_outputs(summary)
```

Nested `start_span()` calls automatically create parent-child relationships.

### Step 3: Combining Auto + Manual Tracing

Enable LangChain autolog, then call a LangChain chain inside a manually traced function:

```python
mlflow.langchain.autolog()

@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    chain = prompt | llm
    response = chain.invoke({"text": text})  # auto-traced
    return response.content
```

The resulting trace shows `summarize_with_llm` as the parent span, with auto-generated LangChain spans nested inside.

## Running the Lesson

```bash
cd tutorial/level_1/M5_tracing/2_manual_tracing
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: @mlflow.trace decorator
============================================================
  Pipeline result: {'original': 'Hello world from MLflow tracing', 'word_count': 5, ...}
  Trace ID: <trace-id>

============================================================
Part 2: mlflow.start_span() context manager
============================================================
  Analyzed 3 texts, 14 total words
    - "MLflow makes tracking easy" -> 4 words, avg len 5.5
    - "Tracing shows execution flow" -> 4 words, avg len 5.75
    - "Spans capture details" -> 3 words, avg len 6.0
  Trace ID: <trace-id>

============================================================
Part 3: Combining auto + manual tracing
============================================================
  LLM summary: <one-sentence summary from the LLM>
  Trace ID: <trace-id>
```

In the MLflow UI, navigate to the experiment and open the Traces tab. You will see three traces, each with a tree of spans showing the execution hierarchy.

## Key Takeaways

- **`@mlflow.trace`** is the easiest way to instrument your own functions — just add the decorator.
- **`mlflow.start_span()`** gives you full control to set inputs, outputs, and custom attributes on each span.
- Nested calls (both decorated functions and nested context managers) automatically create parent-child span relationships.
- You can combine manual tracing with auto-tracing — both appear in a single unified trace tree.
- Every span records timing automatically, so you get latency breakdowns for free.

## Next Steps

Continue to **L1-6.1 (Prompt Registry)** to learn how MLflow manages and versions prompt templates.
