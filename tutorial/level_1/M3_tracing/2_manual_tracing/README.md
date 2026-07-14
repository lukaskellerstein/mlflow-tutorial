# L1-M3.2 — Manual Tracing

**Level:** Essentials
**Duration:** 30 min

## Overview

Autologging (L1-M3.1) captures framework calls automatically. But your own code — data processing, validation, orchestration logic — needs manual instrumentation. This lesson shows two manual tracing APIs and how to combine them with autologging for a unified view.

## Prerequisites

- Completed: L1-M3.1 (Autologging and Auto-Tracing)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Two APIs for Manual Tracing

1. **`@mlflow.trace` decorator** — the simplest approach. Decorate a function and MLflow automatically captures its inputs, outputs, and timing as a span. Nested decorated functions create parent-child spans automatically.

2. **`mlflow.start_span()` context manager** — gives full control. You explicitly set inputs, outputs, and custom attributes on each span. Useful for instrumenting code blocks that aren't standalone functions, or when you want to attach metadata.

### Key Span Methods

When using `start_span()`, the span object exposes:
- `span.set_inputs(dict)` — record what went into this step
- `span.set_outputs(value)` — record what came out
- `span.set_attributes(dict)` — attach custom key-value metadata

### Combining Auto and Manual Tracing

You can mix both approaches in a single trace. Wrap your orchestration function with `@mlflow.trace`, and inside it make an OpenAI SDK call that is auto-traced. The result is a single trace tree with both your manual spans and the auto-generated LLM call spans.

## Step-by-Step

### Step 1: @mlflow.trace Decorator

Decorate functions to create spans automatically. When one decorated function calls another, MLflow nests the spans:

```python
@mlflow.trace(name="validate_text")
def validate_text(text: str) -> str:
    return text.strip()

@mlflow.trace(name="process_pipeline")
def process_pipeline(text: str) -> dict:
    validated = validate_text(text)  # becomes a child span
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

### Step 3: Combining Manual + Auto Tracing

Enable OpenAI autolog, then make an LLM call inside a manually traced function:

```python
mlflow.openai.autolog()

@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    with mlflow.start_span(name="prepare_prompt") as span:
        messages = [...]
        span.set_inputs({"text_length": len(text)})

    # This call is auto-traced by mlflow.openai.autolog()
    response = client.chat.completions.create(model=..., messages=messages)
    return response.choices[0].message.content
```

The resulting trace shows `summarize_with_llm` as the parent span, with both the manual `prepare_prompt` span and the auto-generated `chat.completions` span nested inside.

## Running the Lesson

```bash
cd tutorial/level_1/M3_tracing/2_manual_tracing
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: @mlflow.trace decorator
============================================================
  Pipeline result: {'original': 'Hello world from MLflow tracing', ...}
  Trace ID: <trace-id>

============================================================
Part 2: mlflow.start_span() context manager
============================================================
  Analyzed 3 texts, 14 total words
    - "MLflow makes tracking easy" -> 4 words, avg len 5.5
    ...

============================================================
Part 3: Combining manual tracing with autolog
============================================================
  LLM summary: <one-sentence summary>
  Trace ID: <trace-id>
```

In the MLflow UI, navigate to the experiment and open the Traces tab. You will see three traces, each with a tree of spans showing the execution hierarchy.

## Key Takeaways

- **`@mlflow.trace`** is the easiest way to instrument your own functions — just add the decorator.
- **`mlflow.start_span()`** gives full control to set inputs, outputs, and custom attributes on each span.
- Nested calls (both decorated functions and nested context managers) automatically create parent-child span relationships.
- You can combine manual tracing with auto-tracing — both appear in a single unified trace tree.
- Every span records timing automatically, so you get latency breakdowns for free.

## Next Steps

Continue to **L1-M4.1 (LLM Evaluation Basics)** to learn how to use `mlflow.evaluate()` to assess LLM output quality with built-in and GenAI metrics.
