# L2-5.1 — LangChain Agent Tracking

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

This lesson demonstrates how to build a LangChain ReAct agent using `langchain.agents.create_agent`, instrument it with MLflow's LangChain autolog, and then analyse the captured traces to understand the agent's reasoning and tool-usage decisions. By the end, you will know how to observe every step an agent takes -- from receiving a user query to selecting tools and producing a final answer.

## Prerequisites

- Completed: L1-M5.1 (Auto Tracing), L1-M3.2 (LLM/GenAI Autologging)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`)

## Concepts

### The ReAct Pattern

ReAct (Reasoning + Acting) is the dominant pattern for tool-using LLM agents. The agent loops through a cycle:

1. **Reason** -- the LLM decides what to do next based on the user query and any prior observations.
2. **Act** -- the LLM calls a tool (calculator, search, code executor, etc.).
3. **Observe** -- the tool result is fed back to the LLM.
4. **Repeat** or **Finish** -- the LLM decides whether it has enough information to answer.

Each iteration produces messages that MLflow captures as spans within a trace, giving you full visibility into the agent's decision chain.

### MLflow LangChain Autolog

`mlflow.langchain.autolog()` patches LangChain's callback system so that every invocation of a chain, agent, or tool is recorded as a trace with nested spans. No manual instrumentation is required -- just call `autolog()` before you invoke the agent.

### Why Track Agent Behaviour?

- **Debugging** -- see exactly which tools the agent chose and why.
- **Cost control** -- count tokens and iterations to spot runaway agents.
- **Quality assurance** -- compare tool-selection accuracy across model versions or prompt variants.
- **Reproducibility** -- replay the same reasoning chain for regression testing.

## Step-by-Step

### Step 1: Define Custom Tools

We create three deterministic tools using LangChain's `@tool` decorator. Each tool has a clear docstring so the LLM can decide when to use it.

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    ...

@tool
def string_reverser(text: str) -> str:
    """Reverse the characters in a given string."""
    ...

@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text."""
    ...
```

### Step 2: Build the ReAct Agent

We use `langchain.agents.create_agent`, which is the current API replacing the deprecated `AgentExecutor` pattern. It builds a LangGraph state machine under the hood.

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
agent = create_agent(model=llm, tools=[calculator, string_reverser, word_counter])
```

### Step 3: Enable Autolog and Run Tasks

Call `mlflow.langchain.autolog()` once before agent invocations. Then run the agent on several tasks that require different tools.

```python
mlflow.langchain.autolog(log_traces=True)
response = agent.invoke({"messages": [{"role": "user", "content": "What is 15 * 23?"}]})
```

Each invocation is automatically captured as a trace. We also log per-task metrics (latency, tool calls, steps) using nested MLflow runs.

### Step 4: Analyse Traces

After execution, use `mlflow.search_traces()` to retrieve all captured traces and walk their spans programmatically.

```python
traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    return_type="list",
)
for trace in traces:
    for span in trace.data.spans:
        print(f"[{span.span_type}] {span.name} -- {span.status}")
```

This shows the full decision chain: LLM calls, tool selections, tool results, and the final answer.

## Running the Lesson

```bash
cd tutorial/level_2/M5_agent_observability/1_langchain_agents
uv sync
uv run python main.py
```

## Expected Output

Terminal output will show each task being processed with the agent's answer, latency, and tool-call count, followed by a summary table and trace analysis:

```
============================================================
Task 1: What is 15 * 23?
============================================================
Answer : 345
Latency: 3.14s

...

============================================================
Results Summary
============================================================
Task   Tool Calls   Steps    Latency    Answer
--------------------------------------------------------------------------------
1      1            4        3.140      345
2      1            4        2.870      wolfLM
3      1            4        2.950      8

============================================================
Part 4: Trace Analysis
============================================================
  Found 6 trace(s).

  --- Trace 1 (ID: tr-abc123...) ---
  Number of spans: 15
    [CHAIN] LangGraph — SpanStatus(...)
    [CHAIN] agent — SpanStatus(...)
    [CHAIN] call_model — SpanStatus(...)
    [CHAT_MODEL] ChatOllama — SpanStatus(...)
    [CHAIN] should_continue — SpanStatus(...)
    [CHAIN] tools — SpanStatus(...)
    [TOOL] calculator — SpanStatus(...)
      Inputs : {'expression': '15 * 23'}
      Outputs: {'content': 'Result: 345', ...}
    ...
```

Note: Latency depends on your hardware and Ollama load. The `gemma4:e2b` model is small but still requires GPU inference time.

In the MLflow UI at http://127.0.0.1:5000, navigate to the experiment **L2/M5_agent_observability/1_langchain_agents** to view:
- The parent run with aggregate metrics
- Nested child runs for each task
- Full traces with span trees showing the ReAct loop

## Key Takeaways

- `langchain.agents.create_agent` is the current replacement for the deprecated `AgentExecutor` pattern.
- `mlflow.langchain.autolog()` captures every LLM call, tool invocation, and chain execution as spans within a trace -- zero manual instrumentation needed.
- Traces can be searched and analysed programmatically via `mlflow.search_traces()`, giving you access to span names, types, inputs, outputs, and status.
- Logging per-task metrics (tool calls, latency, steps) alongside traces gives you both the "what happened" (traces) and the "how well" (metrics) views.
- Nested MLflow runs are useful for organising multi-task agent sessions.

## Next Steps

In L2-5.2 (LangGraph Agents), we will build a more complex agent using LangGraph's `StateGraph` directly, with custom state management and conditional routing between nodes, and track the full state-transition graph in MLflow.
