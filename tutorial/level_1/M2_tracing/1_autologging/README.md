# L1-M2.1 — Autologging and Auto-Tracing

**Level:** Essentials
**Duration:** 30 min

## Overview

MLflow can automatically trace GenAI calls with a single line of code. This lesson demonstrates three approaches — `mlflow.openai.autolog()` for direct OpenAI SDK calls, `mlflow.langchain.autolog()` for LangChain agents, and `mlflow.autolog()` as a universal switch that enables everything at once.

## Prerequisites

- Completed: L1-M1 (Tracking)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### What is GenAI Autologging?

GenAI autologging captures **traces**: the full input/output flow of every LLM call, tool invocation, and agent step. Unlike traditional ML autologging (which logs parameters, metrics, and models), GenAI autologging focuses on the request/response flow.

Each traced call produces a trace visible in the MLflow UI's **Traces** tab, showing:

- **Inputs**: the prompt or messages sent to the LLM
- **Outputs**: the model's response
- **Latency**: how long each step took
- **Token usage**: prompt, completion, and total tokens
- **Execution structure**: parent-child span relationships (agent → model call → tool call)

### Supported Frameworks

MLflow provides autologging for 16+ GenAI frameworks:

| Integration | What It Traces |
|---|---|
| `mlflow.openai.autolog()` | OpenAI SDK calls (chat, completions, embeddings) |
| `mlflow.langchain.autolog()` | LangChain agents, LangGraph graphs |
| `mlflow.anthropic.autolog()` | Anthropic/Claude API calls |
| `mlflow.gemini.autolog()` | Google Gemini |
| `mlflow.mistral.autolog()` | Mistral |
| `mlflow.bedrock.autolog()` | Amazon Bedrock |
| `mlflow.groq.autolog()` | Groq |
| `mlflow.litellm.autolog()` | LiteLLM |
| `mlflow.crewai.autolog()` | CrewAI agents |
| `mlflow.dspy.autolog()` | DSPy |
| `mlflow.llama_index.autolog()` | LlamaIndex |
| `mlflow.pydantic_ai.autolog()` | Pydantic AI |
| `mlflow.autogen.autolog()` | AutoGen |
| `mlflow.smolagents.autolog()` | HF smolagents |
| `mlflow.haystack.autolog()` | Haystack |
| `mlflow.strands.autolog()` | Strands Agents |

Or use `mlflow.autolog()` to enable them all at once.

## Step-by-Step

### Step 1: OpenAI SDK Autologging

Trace direct OpenAI SDK calls (which also works with OpenAI-compatible servers like LMStudio):

```python
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "What is MLflow?"}],
    max_tokens=1024,
)
```

This produces a trace with input messages, output content, token usage, and latency. No LangChain needed.

### Step 2: LangChain Agent Autologging

Trace a LangChain agent built with `create_agent`:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

mlflow.langchain.autolog()

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    max_tokens=1024,
)

agent = create_agent(
    model=llm,
    tools=[get_current_time],
    system_prompt="You are a helpful assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
```

The trace shows the full agent execution: model calls, tool selection, tool execution, and the final response — each as a nested span.

### Step 3: Universal Switch

`mlflow.autolog()` enables autologging for every supported framework at once:

```python
mlflow.autolog()
```

Both OpenAI SDK and LangChain calls are traced without needing separate `autolog()` calls. Use this when you want comprehensive tracing across your entire application.

### Step 4: Inspecting Traces Programmatically

Search and inspect traces via the Python API:

```python
traces = mlflow.search_traces(
    locations=[experiment_id],
    max_results=5,
    return_type="list",
    flush=True,
)

for trace in traces:
    print(trace.info.trace_id, trace.info.execution_duration)
    for span in trace.data.spans:
        print(f"  {span.name} [{span.span_type}]")
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_tracing/1_autologging
uv sync
uv run python main.py
```

## Expected Output

In the terminal you will see LLM responses for each part, along with notes about what autologging captured.

In the MLflow UI at http://127.0.0.1:5000:

1. Navigate to the experiment **L1/M2_tracing/1_autologging**
2. Click the **Traces** tab
3. You should see traces from all parts — OpenAI SDK, LangChain agent, and universal autolog
4. Click any trace to expand it and see the span tree with inputs, outputs, and latencies
5. Compare the agent trace (with tool call spans) to the simple OpenAI SDK trace

## Key Takeaways

- `mlflow.openai.autolog()` traces direct OpenAI SDK calls — works with any OpenAI-compatible server (like LMStudio).
- `mlflow.langchain.autolog()` traces LangChain agents and LangGraph graphs, including tool calls and state transitions.
- `mlflow.autolog()` enables all 16+ GenAI integrations at once — the simplest way to get full observability.
- Traces capture inputs, outputs, token usage, latency, and execution structure.
- Use `mlflow.search_traces()` to inspect traces programmatically.

## Next Steps

In **L1-M2.2 (Manual Tracing)**, you will learn how to add custom tracing with `@mlflow.trace` and `mlflow.start_span()` for business logic that autologging doesn't cover.
