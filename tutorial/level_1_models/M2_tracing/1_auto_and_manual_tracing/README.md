# L1-M2.1 -- Auto-Tracing and Manual Tracing

**Level:** Essentials
**Duration:** 30m

## Overview

MLflow provides two complementary approaches to tracing GenAI applications: automatic tracing (autologging) that captures framework calls with zero code changes, and manual tracing that instruments your own business logic. This lesson demonstrates both sides and shows how to combine them into a unified trace tree.

## Prerequisites

- Completed: L1-M1 (Tracking)
- MLflow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Auto-Tracing (Autologging)

GenAI autologging captures traces -- the full input/output flow of every LLM call, tool invocation, and agent step. Enable it with a single line:

- `mlflow.openai.autolog()` -- traces OpenAI SDK calls (also works with OpenAI-compatible servers like LMStudio)
- `mlflow.langchain.autolog()` -- traces LangChain agents and LangGraph graphs
- `mlflow.autolog()` -- enables all 16+ GenAI integrations at once

Each traced call produces a trace in the MLflow UI showing inputs, outputs, token usage, latency, and execution structure (parent-child spans).

### Manual Tracing

Autologging covers framework calls, but your own code -- data processing, validation, orchestration -- needs manual instrumentation. Two APIs:

1. **`@mlflow.trace` decorator** -- decorate a function and MLflow captures its inputs, outputs, and timing as a span. Nested decorated functions create parent-child spans automatically.

2. **`mlflow.start_span()` context manager** -- gives full control. You explicitly set inputs, outputs, and custom attributes on each span.

### Combining Both

You can mix auto and manual tracing in a single trace tree. Wrap your orchestration function with `@mlflow.trace`, make LLM calls inside it that are auto-traced, and the result is one unified tree with both manual spans and auto-generated LLM call spans.

## Step-by-Step

### Step 1: OpenAI SDK Autologging

Trace direct OpenAI SDK calls with a single line:

```python
mlflow.openai.autolog()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "What is MLflow?"}],
    max_tokens=1024,
)
```

### Step 2: LangChain Agent Autologging

Trace a LangChain agent including tool calls and state transitions:

```python
mlflow.langchain.autolog()

agent = create_agent(model=llm, tools=[get_current_time], system_prompt="...")
result = agent.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
```

### Step 3: @mlflow.trace Decorator

Decorate functions to create spans automatically. Nested decorated functions become child spans:

```python
@mlflow.trace(name="validate_text")
def validate_text(text: str) -> str:
    return text.strip()


@mlflow.trace(name="process_pipeline")
def process_pipeline(text: str) -> dict:
    validated = validate_text(text)  # becomes a child span
    return {"result": validated}
```

### Step 4: mlflow.start_span() Context Manager

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

### Step 5: Combining Auto + Manual Tracing

Enable autolog, then make LLM calls inside a manually traced function:

```python
mlflow.openai.autolog()


@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    with mlflow.start_span(name="prepare_prompt") as span:
        messages = [...]
        span.set_inputs({"text_length": len(text)})
    response = client.chat.completions.create(model=..., messages=messages)
    return response.choices[0].message.content
```

The resulting trace shows `summarize_with_llm` as the root, with both the manual `prepare_prompt` span and the auto-generated `chat.completions` span nested inside.

## Running the Lesson

```bash
cd tutorial/level_1_models/M2_tracing/1_auto_and_manual_tracing
uv sync
uv run python main.py
```

## Expected Output

You will see five sections in the terminal:

1. **Part 1** -- OpenAI SDK auto-traced response
2. **Part 2** -- LangChain agent auto-traced with tool calls
3. **Part 3** -- Decorator-traced pipeline with nested spans
4. **Part 4** -- Batch analysis with explicit span control and custom attributes
5. **Part 5** -- Combined auto + manual trace in one unified tree

In the MLflow UI at http://127.0.0.1:5555, navigate to experiment **L1/M2_tracing/1_auto_and_manual_tracing** and open the Traces tab. Click any trace to see the span tree.

## Key Takeaways

- `mlflow.openai.autolog()` traces OpenAI SDK calls automatically -- works with any OpenAI-compatible server.
- `mlflow.langchain.autolog()` traces LangChain agents including tool calls and state transitions.
- `@mlflow.trace` is the easiest way to instrument your own functions -- just add the decorator.
- `mlflow.start_span()` gives full control to set inputs, outputs, and custom attributes on each span.
- You can combine auto and manual tracing -- both appear in a single unified trace tree.
- Every span records timing automatically, so you get latency breakdowns for free.

## Next Steps

Continue to **L1-M3.1 (Models, Flavors, and Signatures)** to learn how MLflow packages models into a portable, self-describing format with different flavors and input/output signatures.
