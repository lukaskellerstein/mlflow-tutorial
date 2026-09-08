# L2-M1.1.2 — DeepAgents

**Level:** AI Agents
**Duration:** 90 min

## Overview

DeepAgents is LangChain-AI's opinionated agent harness. `create_deep_agent()`
wraps `create_agent()` and hands the model a built-in toolkit it would otherwise
need hand-written: planning, a virtual filesystem, and sub-agent delegation. It
compiles to a LangGraph graph, so `mlflow.langchain.autolog()` traces all of it
with no custom integration — including work done inside sub-agents.

This is a **turn** lesson. Every task starts from an empty message list, and
nothing is carried from one task to the next.

## Prerequisites

- Completed: L2-M1.1.1 (LangChain + LangGraph Agents)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>
- An `UNSLOTH_API_KEY` in the environment — the only key this stack needs

## Concepts

### Where DeepAgents sits

```text
DeepAgents      opinionated harness: built-in tools, sub-agents, backends
LangChain       agent abstraction: model + tools + middleware -> agent loop
LangGraph       runtime: state, checkpoints, streaming, interrupts
```

`create_deep_agent()` returns a standard compiled `StateGraph`, so `.invoke()`,
`.stream()` and their async twins work exactly as in L2-M1.1.1.

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
a nested graph under the `task` span, and Part 3 counts those spans: on the
delegating turn, 16 of 32 spans ran inside a sub-agent.

Tools and sub-agents live on **one** agent. `create_deep_agent()` takes `tools=`
and `subagents=` together, and the model picks per task: do it myself, or hand it
over. Part 1 gives one agent both, then sends three different tasks through it.

### Backends decide what a "file" is

The filesystem tools are backend-agnostic. The backend decides where the bytes go:

| Backend | A file is | Outlives the turn? | Covered in |
|:--|:--|:--|:--|
| `StateBackend` (default) | a key in the returned state | no | here, Part 2 |
| `FilesystemBackend` | a real file on disk | yes | here, Part 2 |
| `StoreBackend` | a LangGraph store entry | yes, across conversations | `2_conversation/2_deepagents` |
| `CompositeBackend` | routes path prefixes to other backends | depends on route | `2_conversation/2_deepagents` |

Only the first two belong in a turn lesson. The other two exist to carry files
*between* turns, which is memory — and memory is what the conversation lessons
are for.

The difference between the two shown here is visible after a **single** turn, and
it decides how MLflow can log the result:

- `FilesystemBackend` gives you a real path, so `mlflow.log_artifact(path)` works.
- `StateBackend` gives you no path, so you must pull the text out of the returned
  state and use `mlflow.log_text()`.

## Step-by-Step

### Step 1: Trace everything with one call

```python
mlflow.langchain.autolog()
```

### Step 2: Part 1 — one agent with tools AND sub-agents

Sub-agents are plain dicts. Omitting `tools` gives the sub-agent the built-in set
only; omitting `model` inherits the parent's.

```python
RESEARCHER: SubAgent = {
    "name": "researcher",
    "description": "Researches ONE topic using the knowledge base and statistics tools.",
    "system_prompt": "You are a research specialist. ...",
    "tools": CUSTOM_TOOLS,
}

agent = create_deep_agent(
    model=get_llm(),
    tools=CUSTOM_TOOLS,
    subagents=[RESEARCHER, ANALYST],
    system_prompt=(
        "You are a technology research assistant. You have your own tools and two "
        "sub-agents ('researcher', 'analyst') reachable with the task tool. ..."
    ),
)
```

### Step 3: Part 1 — run a list of tasks through it

One parent run, one nested run per task, exactly as in L2-M1.1.1. Each task gets
a fresh message list, so the three are independent:

```python
for idx, (name, task) in enumerate(TASKS, start=1):
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config=RUN_CONFIG)
    with mlflow.start_run(run_name=f"task_{idx}_{name}", nested=True):
        mlflow.log_metrics({"subagent_handoffs": count_tool_calls(result["messages"]).get("task", 0), ...})
```

The three tasks pick different parts of the toolkit: plan and research, delegate
to both sub-agents, then write a file and `grep` it.

### Step 4: Part 2 — the same task against two backends

```python
backends = [
    ("StateBackend", StateBackend()),
    ("FilesystemBackend", FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True)),
]
```

Both agents get the same prompt and make the same `write_file` call. Afterwards,
one has a file in `result["files"]` and nothing on disk; the other has the
opposite.

### Step 5: Part 3 — read the traces back

Part 3 runs no agents, and takes no arguments. It asks the tracking server for
the traces the parts above produced.

For that to work, each turn names its own trace. `autolog` opens and closes a
trace around every `invoke`, so the tag goes on afterwards:

```python
def tag_turn(label: str) -> None:
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        return
    mlflow.set_trace_tag(trace_id, "turn", label)
    mlflow.set_trace_tag(trace_id, "session", SESSION)
```

`SESSION` is a timestamp made once when the file starts. It matters because the
experiment also holds the traces of every previous run — filtering on it is what
makes "the traces this execution produced" an exact query instead of "the most
recent few":

```python
traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    filter_string=f"tags.session = '{SESSION}'",
    return_type="list",
    flush=True,
)
label = trace.info.tags["turn"]
usage = trace.info.token_usage  # {'input_tokens': ..., 'output_tokens': ..., 'total_tokens': ...}
```

`flush=True` waits for the async exporter, so a trace written seconds ago is
already there.

A span counts as **hidden** when one of its ancestors is a `task` tool call. That
is work a sub-agent did, and work the caller never saw:

```python
def inside_subagent(span, by_id) -> bool:
    while span.parent_id and span.parent_id in by_id:
        span = by_id[span.parent_id]
        if span.name == "task":
            return True
    return False
```

### Step 6: Cap the recursion limit

DeepAgents defaults to `recursion_limit=9999`, sized for frontier models. Cap it
so a confused run fails fast instead of looping for an hour:

```python
RUN_CONFIG = {"recursion_limit": 50}
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/1_turn/2_deepagents
uv sync
uv run python main.py
```

Measured end to end: under a minute. Task 2 is the delegating one and takes most
of that on its own. Part 3 runs no agents at all — it only reads traces.

## Expected Output

**Part 1, task 2** — one agent, but this task is delegated:

```text
    [HumanMessage] Use the task tool to delegate. Ask the 'researcher' sub-agent for facts about monolith ...
    [AIMessage] -> task -> sub-agent 'researcher'
    [ToolResult] ### Monolith Architecture Research Report ...
    [AIMessage] -> task -> sub-agent 'analyst'
    [ToolResult] ### Key Findings * Lifecycle Pattern: Monolithic architecture is the standard ...
    [AIMessage] -> write_file({'content': '### Key Findings\n* **Lifecycle Pattern:** Monolithic arc)
    [ToolResult] Updated file /analysis.md

    18.01s | steps 8 | tools 3 | handoffs 2
```

The three tasks summarised:

```text
  task  name               steps  tools  handoffs  files  duration
  ----  -----------------  -----  -----  --------  -----  --------
  1     plan_and_research  10     4      0         2      3.28
  2     delegate           8      3      2         1      18.01
  3     write_then_grep    10     4      0         1      2.22
```

**On the empty todos list:** `write_todos` is offered, not forced, and gemma-4
usually plans by writing a plain `/todos.txt` with `write_file` instead of calling
the dedicated tool. The planning still happened — it just landed in the filesystem
channel rather than the todos channel. This is left visible rather than papered
over, because "the agent had the tool and chose something else" is exactly the
kind of thing you are meant to catch in a trace.

**On task 3:** the model calls `grep` twice — once with the default output mode,
which returns only the file name, then again with `output_mode: 'content'` to get
the matching line. A tool call that half-worked is easy to miss in the answer and
obvious in the trace.

**Part 2** — the same task, the same tool call, one turn each:

```text
  StateBackend: 1.42s
    files in the returned state : ['/research.md']
    files in ./workspace        : (none)

  FilesystemBackend: 0.95s
    files in the returned state : (none)
    files in ./workspace        : ['research.md']

  After ONE turn — the agent did the same thing, the file did not:
  backend            in state  on disk  MLflow logs it with
  -----------------  --------  -------  -------------------------
  StateBackend       1         0        mlflow.log_text(state)
  FilesystemBackend  0         1        mlflow.log_artifact(path)
```

Check the two runs in the UI: `backend_StateBackend` carries
`from_state/research.md` logged as text, `backend_FilesystemBackend` carries
`research.md` logged as a real artifact.

**Part 3** — one row per turn, read back from the traces:

```text
  5 trace(s) tagged session=20260831-001842

  One row per turn. 'hidden' spans ran inside a sub-agent:
  turn               spans  hidden  input_tok  output_tok  ms
  -----------------  -----  ------  ---------  ----------  -----
  plan_and_research  20     0       12868      1120        3251
  delegate           32     16      23054      4727        18010
  write_then_grep    20     0       12436      564         2220
  StateBackend       12     0       7019       354         1423
  FilesystemBackend  12     0       7019       354         945
```

The two backend turns are the same size in every column, which is the point Part
2 makes, now measured: the backend changed where the file went, not what the agent
did.

The delegating turn is the interesting one, so its span tree is printed in full:

```text
  Span tree of 'delegate' — 16 of 32 spans ran inside a sub-agent:
      [CHAIN] LangGraph (18011ms)
        [CHAIN] model (671ms)
          [CHAT_MODEL] ChatOpenAI (669ms)
        [CHAIN] tools (3539ms)
          [TOOL] task (3536ms)
            [CHAIN] researcher (3535ms)
              [CHAIN] model (1731ms)
                [CHAT_MODEL] ChatOpenAI (1725ms)
              [CHAIN] tools (2ms)
                [TOOL] search_knowledge_base (1ms)
              [CHAIN] model (649ms)
                [CHAT_MODEL] ChatOpenAI (646ms)
              [CHAIN] tools (5ms)
                [TOOL] get_industry_stats (1ms)
              [CHAIN] model (1143ms)
                [CHAT_MODEL] ChatOpenAI (1140ms)
        [CHAIN] model (1637ms)
          [CHAT_MODEL] ChatOpenAI (1635ms)
        [CHAIN] tools (8739ms)
          [TOOL] task (8737ms)
            [CHAIN] analyst (8736ms)
              [CHAIN] model (8735ms)
                [CHAT_MODEL] ChatOpenAI (8731ms)
```

This is the claim about context isolation, made concrete. The `researcher`
sub-agent ran three model calls and two tool calls. Not one of them entered the
orchestrator's message list — the orchestrator got a single `ToolMessage` holding
the final report. Half the spans in the trace are work the caller never saw.

It is also the honest cost: those 16 hidden spans are real tokens on the bill.
Delegation buys context, and pays for it in tokens and wall-clock.

In the MLflow UI under **L2/M1_agent_frameworks/1_turn/2_deepagents**: 8 runs (one
parent with three nested tasks, two backend runs, one backend comparison, one
trace analysis), the tables `tasks.json`, `backends.json` and `traces.json`, and
traces where sub-agent execution nests under `task`.

## Key Takeaways

- `create_deep_agent()` is `create_agent()` plus a built-in toolkit; it is still
  a LangGraph graph, so autolog covers it.
- One agent takes `tools=` and `subagents=` together. The split between "uses a
  tool" and "delegates" is a decision the model makes per task, not a decision you
  make per agent.
- Sub-agents buy context isolation, not speed. Count the spans under `[TOOL] task`
  and you have the number: that is work the caller paid for and never saw.
- The backend, not the tool, decides whether `write_file` produced a real file —
  and that decides whether MLflow can log it with `log_artifact` or only with
  `log_text`.
- Offering a tool does not mean the model will use it, or call it correctly; the
  trace is where you find that out.
- `trace.info.token_usage` gives the whole bill for a turn, sub-agents included.
  It needs no instrumentation of your own — autolog already recorded it.
- Tag a trace when you make it, and you can query it by name later. Without a tag
  the only handle you have on a past trace is "the most recent few".
- Cap `recursion_limit` — the 9999 default is not sized for smaller models.

## Next Steps

**L2-M1.1.3 — Claude Agent SDK** drops to a framework with no MLflow autolog at
all, and builds the tracing integration by hand.

For `StoreBackend`, `CompositeBackend` and files that survive a turn boundary, go
to **L2-M1.2.2 — Multi-Turn Conversations with DeepAgents**.
