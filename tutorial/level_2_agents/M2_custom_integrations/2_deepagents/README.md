# L2-M2.2 — DeepAgents + MLflow

**Level:** AI Agents
**Duration:** 90 min

## Overview

This lesson uses the real DeepAgents library — LangChain-AI's opinionated agent harness built on top of LangGraph. You'll create agents with `create_deep_agent()`, delegate work to sub-agents via the `task` tool, and see how MLflow auto-traces the entire orchestration because DeepAgents is built on the LangChain/LangGraph stack.

## Prerequisites

- Completed: L2-M2.1 (Claude Agent SDK + MLflow)
- MLFlow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` loaded (context length ≥ 16384)

## Concepts

### DeepAgents Architecture

DeepAgents sits on top of LangChain and LangGraph:

```
DeepAgents      opinionated harness: defaults, middleware, backends, profiles
LangChain       agent abstraction: model + tools + middleware -> agent loop
LangGraph       runtime: state, checkpoints, streaming, interrupts
```

The core entry point is `create_deep_agent()`, which returns a standard LangGraph `CompiledStateGraph`. This means you invoke it with `.invoke()`, `.stream()`, `.ainvoke()`, or `.astream()` — exactly like any LangGraph graph.

### Built-in Tools

Every deep agent comes with these tools automatically:

| Tool | Purpose |
|------|---------|
| `write_todos` | Planning and todo list management |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | Virtual filesystem (StateBackend by default) |
| `task` | Sub-agent delegation — spawns isolated context windows |

Custom tools passed via `tools=` are **additive** — they never replace built-ins.

### Sub-agents and the `task` Tool

Sub-agents are declared as plain dicts and registered via `subagents=`. The orchestrator delegates work by calling the `task` tool, which spawns a sub-agent with its **own isolated context window** — the orchestrator only sees the final answer, not intermediate tool calls. This is the key architectural difference from LangGraph's multi-agent patterns, where agents share state.

### MLflow Integration

Because DeepAgents returns a LangGraph graph, `mlflow.langchain.autolog()` captures everything automatically — no custom tracing code needed. Every agent step, tool call, and sub-agent invocation appears in the MLflow trace.

## Step-by-Step

### Step 1: Basic Deep Agent with Custom Tools

Create a deep agent with domain-specific tools added to the built-in suite:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=llm,
    tools=[search_knowledge_base, get_industry_stats],
    system_prompt="You are a research assistant...",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Research microservices..."}]},
    config={"recursion_limit": 50},
)
```

The agent uses `write_todos` to plan, custom tools to gather data, and `write_file` to save results — all auto-traced by MLflow.

### Step 2: Sub-agent Orchestration

Define specialist sub-agents as dicts and create an orchestrator:

```python
researcher = {
    "name": "researcher",
    "description": "Researches ONE topic using the knowledge base.",
    "system_prompt": "You are a research specialist...",
    "tools": [search_knowledge_base, get_industry_stats],
}

analyst = {
    "name": "analyst",
    "description": "Analyzes research findings.",
    "system_prompt": "You are an analysis specialist...",
}

agent = create_deep_agent(
    model=llm,
    subagents=[researcher, analyst],
    system_prompt="You are an orchestrator. Delegate to sub-agents via task...",
)
```

The orchestrator calls `task(subagent_type="researcher", ...)` to delegate research, then `task(subagent_type="analyst", ...)` to analyze findings. Each sub-agent runs in its own context.

### Step 3: Metrics and Comparison

Track orchestration metrics (steps, handoffs, duration) and compare multi-agent vs single-agent approaches on the same task.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_custom_integrations/2_deepagents
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-M2.2 — DeepAgents + MLflow
============================================================

DeepAgents is built on LangGraph, so mlflow.langchain.autolog()
captures all agent steps, tool calls, and sub-agent traces.

============================================================
Part 1: Basic Deep Agent with Custom Tools
============================================================

  Conversation:
    [HumanMessage] Use write_todos to plan your work, then: research...
    [AIMessage] -> write_todos({"todos": [...]})
    [ToolResult] Todos updated
    [AIMessage] -> search_knowledge_base({"query": "microservices"})
    [ToolResult] Microservices architecture decomposes...
    [AIMessage] -> get_industry_stats({"topic": "microservices"})
    [ToolResult] Adoption: 85% of enterprises...
    [AIMessage] -> write_file({"path": "/research.md", ...})
    [ToolResult] File written
    [AIMessage] I've completed the research...

  Todos:
    [completed] Research microservices using knowledge base
    [completed] Get industry statistics
    [completed] Save summary to /research.md

  Files (StateBackend — ephemeral, in agent state):
    /research.md (450 chars)

  Duration: 15.2s | Steps: 11 | Tool calls: {...}

============================================================
Part 2: Sub-agent Orchestration (task tool)
============================================================

  Streaming orchestration steps:
    [task] -> sub-agent 'researcher': Research microservices vs monolith...
    [result] Key findings: ...
    [task] -> sub-agent 'analyst': Analyze comparison findings...
    [result] Analysis: ...
    [tool] -> write_file({"path": "/analysis.md", ...})

  Duration: 25.3s | Steps: 9
  Sub-agent handoffs (task calls): 2

============================================================
Part 3: Single Agent vs Multi-Agent Comparison
============================================================

  Comparison:
                              approach  duration_s  total_steps  tool_calls  subagent_handoffs
  multi_agent (orchestrator + subagents)       25.3            9           5                  2
                          single_agent       12.1           11           4                  0
```

In the MLflow UI:
- **Traces** show nested spans for orchestrator → sub-agent → tool calls
- **Four runs**: `basic_deep_agent`, `subagent_orchestration`, `single_agent_baseline`, `approach_comparison`
- Sub-agent traces are isolated — you can see each sub-agent's internal reasoning separately

## Key Takeaways

- DeepAgents wraps LangChain/LangGraph with opinionated defaults: built-in tools (filesystem, planning, sub-agents), middleware, and backends.
- `create_deep_agent()` returns a standard LangGraph graph — `mlflow.langchain.autolog()` captures everything automatically.
- Sub-agents run in **isolated context windows** — the orchestrator sees only the final answer. This differs from LangGraph's shared-state multi-agent patterns (L2-M1.3).
- The `StateBackend` (default) keeps files in ephemeral agent state. `FilesystemBackend` writes to disk for persistence.
- Cap `recursion_limit` (default 9999) when using local models to fail fast on loops.

## Next Steps

Continue to **L2-M3.1 — Agent Testing Framework** to learn how to systematically test and evaluate agents using `mlflow.genai.agent_tester` and conversation simulators.
