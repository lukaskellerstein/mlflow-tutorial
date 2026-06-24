# L1-3.2 — LLM and GenAI Autologging

**Level:** Essentials
**Duration:** 30 min

## Overview

MLflow can automatically trace LLM and GenAI calls with a single line of code. This lesson shows how `mlflow.langchain.autolog()` captures inputs, outputs, latency, and the full execution graph for LangChain-based LLM interactions — no manual logging required.

## Prerequisites

- Completed: L1-3.1 (Traditional ML Autologging)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled (`ollama pull gemma4:e2b`)

## Concepts

### What is GenAI Autologging?

In the previous lesson, you saw how `mlflow.autolog()` captures parameters, metrics, and models for traditional ML frameworks. GenAI autologging works differently — instead of logging training metrics, it captures **traces**: the full input/output flow of every LLM call, chain step, and tool invocation.

When you call `mlflow.langchain.autolog()`, MLflow installs a callback that intercepts every LangChain operation. Each `invoke()` call produces a trace visible in the MLflow UI's **Traces** tab, showing:

- **Inputs**: the prompt or message sent to the LLM
- **Outputs**: the model's response
- **Latency**: how long each step took
- **Chain structure**: parent-child relationships between steps (prompt rendering, LLM call, output parsing)

### Supported Frameworks

MLflow provides autologging integrations for multiple LLM providers:

| Integration | What It Traces |
|---|---|
| `mlflow.langchain.autolog()` | LangChain chains, agents, LangGraph graphs |
| `mlflow.openai.autolog()` | OpenAI API calls (chat, completions, embeddings) |
| `mlflow.anthropic.autolog()` | Anthropic/Claude API calls |
| `mlflow.transformers.autolog()` | Hugging Face model inference |

Additional integrations exist for Mistral, Gemini, Bedrock, Groq, and LiteLLM. In this lesson, we use `mlflow.langchain.autolog()` with a local Ollama model via LangChain.

## Step-by-Step

### Step 1: Enable Autologging

A single line enables tracing for all LangChain operations:

```python
mlflow.langchain.autolog()
```

This installs an MLflow tracer callback that runs automatically whenever you call `.invoke()`, `.ainvoke()`, `.stream()`, or `.batch()` on any LangChain runnable.

### Step 2: Simple LLM Call

The most basic case — call the LLM directly:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:e2b", temperature=0.7)
response = llm.invoke("What are the three laws of thermodynamics?")
```

This produces a single trace with the input message, output content, model name, and execution time.

### Step 3: Chain with Prompt Template

Build a three-step chain (prompt template, LLM, output parser) and invoke it:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise science educator."),
    ("human", "Explain {topic} in exactly two sentences."),
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "black holes"})
```

The trace now shows three child spans: prompt rendering, LLM call, and string parsing. You can see the rendered prompt (with variables substituted) and the final parsed output.

### Step 4: Multi-Step Chain

Two sequential chain invocations — first generate a summary, then create a title from it:

```python
summary = summarize_chain.invoke({"topic": "quantum computing"})
title = title_chain.invoke({"summary": summary})
```

Each invocation creates its own trace. In the MLflow UI, you can compare their latencies and inspect how the output of the first chain becomes the input of the second.

## Running the Lesson

```bash
cd tutorial/level_1/M3_autologging/2_llm_genai
uv sync
uv run python main.py
```

## Expected Output

In the terminal you will see the LLM responses for each part, along with notes about what autologging captured.

In the MLflow UI at http://127.0.0.1:5000:

1. Navigate to the experiment **L1/M3_autologging/2_llm_genai**
2. Click the **Traces** tab
3. You should see 4 traces (one for the simple call, one for the prompt chain, and two for the multi-step chain)
4. Click any trace to expand it and see the span tree with inputs, outputs, and latencies

## Key Takeaways

- `mlflow.langchain.autolog()` is a single line that enables full tracing for all LangChain operations.
- Traces capture inputs, outputs, latency, and the full execution structure of chains.
- Each chain component (prompt template, LLM, parser) appears as a separate span in the trace.
- Multiple LLM providers are supported — LangChain, OpenAI, Anthropic, Hugging Face, and more.
- Unlike traditional ML autologging (which logs params/metrics/models), GenAI autologging focuses on **traces** — the request/response flow.

## Next Steps

In **L1-4.1 (Traditional ML Evaluation)**, you will learn how to use `mlflow.evaluate()` to assess model quality with built-in metrics. Later in **L1-5.1 (Auto Tracing)**, you will explore tracing in more depth, including how to interpret trace structure and use it for debugging.
