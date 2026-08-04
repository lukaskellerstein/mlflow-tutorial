# L2-M1.2 — DeepAgents

**Level:** AI Agents
**Duration:** 90 min

## Overview

DeepAgents is LangChain-AI's opinionated agent harness. `create_deep_agent()`
wraps `create_agent()` and hands the model a built-in toolkit it would otherwise
need hand-written: planning, a virtual filesystem, and sub-agent delegation. It
compiles to a LangGraph graph, so `mlflow.langchain.autolog()` traces all of it
with no custom integration — including work done inside sub-agents.

## Prerequisites

- Completed: L2-M1.1 (LangChain + LangGraph Agents)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in the environment

## Concepts

### Where DeepAgents sits

```text
DeepAgents      opinionated harness: built-in tools, sub-agents, backends
LangChain       agent abstraction: model + tools + middleware -> agent loop
LangGraph       runtime: state, checkpoints, streaming, interrupts
```

`create_deep_agent()` returns a standard compiled `StateGraph`, so `.invoke()`,
`.stream()` and their async twins work exactly as in L2-M1.1.

### The built-in toolkit

Your custom tools are *added* to this set, never replacing it:

| Tool | Purpose |
|:--|:--|
| `write_todos` | the agent's own plan, kept in state |
| `ls` `read_file` `write_file` `edit_file` `glob` `grep` | a virtual filesystem |
| `task` | delegate to a sub-agent |

### Sub-agents and context isolation

`task` is the whole point of sub-agents. The orchestrator sees only a sub-agent's
*final answer* — not its intermediate tool calls — so a long research subtask
does not consume the orchestrator's context window. In the trace this shows up as
a nested graph under the `task` span.

### Backends decide what a "file" is

The filesystem tools are backend-agnostic, and the backend decides what survives:

| Backend | A file is | Survives the run? |
|:--|:--|:--|
| `StateBackend` (default) | a key in agent state | no |
| `FilesystemBackend` | a real file on disk | yes |
| `StoreBackend` | a LangGraph store entry | yes, across conversations |
| `CompositeBackend` | routes path prefixes to other backends | depends on route |

Part 3 mounts a `CompositeBackend` whose default is the real `./workspace`
directory and whose `/memories/` prefix goes to a store — then starts a *fresh*
conversation with no checkpointer and no shared messages, so anything the agent
still knows came from the store and nowhere else.

## Step-by-Step

### Step 1: Trace everything with one call

```python
mlflow.langchain.autolog()
```

### Step 2: Part 1 — the built-in toolkit

```python
agent = create_deep_agent(
    model=get_llm(),
    tools=[search_knowledge_base, get_industry_stats],
    system_prompt=(
        "You are a technology research assistant. Use the available tools "
        "to gather facts and statistics, then use write_file to save your "
        "findings. Plan your work with write_todos first."
    ),
)
```

### Step 3: Part 2 — delegate to sub-agents

Sub-agents are plain dicts. Omitting `tools` gives the sub-agent the built-in set
only; omitting `model` inherits the parent's.

```python
researcher: SubAgent = {
    "name": "researcher",
    "description": "Researches ONE topic using the knowledge base and statistics tools.",
    "system_prompt": "You are a research specialist. ...",
    "tools": [search_knowledge_base, get_industry_stats],
}

agent = create_deep_agent(
    model=get_llm(),
    subagents=[researcher, analyst],
    system_prompt="You are an orchestrator. You NEVER research or analyze yourself. ...",
)
```

### Step 4: Part 3 — route paths to different backends

```python
backend = CompositeBackend(
    default=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
    routes={"/memories/": StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("memories",))},
)
```

### Step 5: Cap the recursion limit

DeepAgents defaults to `recursion_limit=9999`, sized for frontier models. Cap it
so a confused run fails fast instead of looping for an hour:

```python
RUN_CONFIG = {"recursion_limit": 50}
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/2_deepagents
uv sync
uv run python main.py
```

Expect roughly 3–5 minutes end to end — Part 2 alone runs three sub-agent
handoffs.

## Expected Output

Part 1 plans, researches and writes a summary into the state filesystem:

```text
    [AIMessage] -> write_file({'file_path': '/todos.md', ...)
    [AIMessage] -> search_knowledge_base({'query': 'microservices architecture'})
    [AIMessage] -> get_industry_stats({'topic': 'microservices architecture'})
    [AIMessage] -> write_file({'content': "# Microservices Architecture Research Summary...)

  Todos (write_todos tool -> agent state):
    (empty — this model planned into a file instead; see Files below)

  Files (StateBackend — ephemeral, lives in agent state):
    /todos.md (224 chars)
    /research.md (1851 chars)

  Duration: 25.17s | Steps: 10 | Tools: {'write_file': 2, 'search_knowledge_base': 1, 'get_industry_stats': 1}
```

**On the empty todos list:** `write_todos` is offered, not forced, and gemma-4
usually plans by writing a plain `/todos.md` with `write_file` instead of calling
the dedicated tool. The planning still happened — it just landed in the
filesystem channel rather than the todos channel. A stronger model picks the
dedicated tool more often. This is left visible rather than papered over, because
"the agent had the tool and chose something else" is exactly the kind of thing
you are meant to catch in a trace.

Part 2 shows the handoffs, including a retry when the first delegation arrives
without its context:

```text
    [task] -> sub-agent 'researcher': Research the key characteristics ...
    [task] -> sub-agent 'analyst': Analyze the following research findings ...
    [result] Please provide the [RESEARCHER'S REPORT] you would like me to analyze.
    [task] -> sub-agent 'analyst': Analyze the following research findings ...
    [tool] -> write_file({'content': '### Architectural Analysis ...)

  Duration: 100.38s | Steps: 10 | Handoffs: 3
```

Three handoffs for two sub-agents: the orchestrator called `analyst` once without
pasting the findings, got asked for them, and called again with the text. That is
a real orchestration failure mode, visible in the trace and counted in the
`subagent_handoffs` metric.

Part 3 writes a real file and then recalls from the store in a fresh
conversation:

```text
  Real files now on disk in ./workspace:
    greeting.txt: "Hello Lukas! It's great to meet you."

  Fresh conversation, answered from the store:
    Based on your `/memories/` folder, here is what I know about you:
    * **Name:** Lukas
    * **Favorite Programming Language:** Python
```

Part 4 compares the two approaches:

```text
  approach                                 steps    tools    handoffs   duration
  multi_agent (orchestrator + subagents)   10       4        3          100.38s
  single_agent                             6        2        0          45.09s
```

Delegation cost roughly 2x the wall-clock and 2x the steps here. That is the
honest result on a task small enough for one agent — sub-agents pay off when a
subtask would otherwise flood the orchestrator's context, not on short work.

In the MLflow UI under **L2/M1_agent_frameworks/2_deepagents**: five runs, a
`comparison.json` table, and traces where sub-agent execution nests under `task`.

## Key Takeaways

- `create_deep_agent()` is `create_agent()` plus a built-in toolkit; it is still
  a LangGraph graph, so autolog covers it.
- Sub-agents buy context isolation, not speed — the orchestrator sees only their
  final answers.
- The backend, not the tool, decides whether a file outlives the run.
- Offering a tool does not mean the model will use it; the trace is where you
  find that out.
- Cap `recursion_limit` — the 9999 default is not sized for smaller models.

## Next Steps

**L2-M1.3 — Claude Agent SDK** drops to a framework with no MLflow autolog at
all, and builds the tracing integration by hand.
