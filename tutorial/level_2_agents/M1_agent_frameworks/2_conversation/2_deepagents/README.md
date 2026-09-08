# L2-M1.2.2 — Multi-Turn Conversations with DeepAgents

**Level:** AI Agents
**Duration:** 45 min

## Overview

L2-M1.2.1 gave a LangChain agent memory with a checkpointer. A deep agent takes
the same argument, but it has more to remember: alongside the messages it keeps
a todo list and a virtual filesystem in the same graph state. This lesson runs a
four-turn conversation where the file written in turn 1 must still be there in
turn 3, then shows two backend choices that cross the thread boundary. One does
it by accident: `FilesystemBackend` leaks a file from one conversation into
another. One does it on purpose: `CompositeBackend` routes `/memories/` to a
`StoreBackend`, so that one prefix is shared and everything else stays private.

Like L2-M1.2.1, this lesson is **two scripts that do the same job**, because
MLflow has two APIs for stamping a session:

| File | What it is |
|:--|:--|
| `main_decorator.py` | approach 1 — `@mlflow.trace` + `mlflow.update_current_trace()` |
| `main_context.py` | approach 2 — a `mlflow.tracing.context()` block |
| `conversation.py` | everything the two hold fixed — imported by both, run by neither |

Repeating the split is deliberate. The two APIs belong to MLflow, not to the
agent framework, so they land the same way on a deep agent as they did on a
plain LangChain one. Part 3 is where that stops being a copy — see
[Step 7](#step-7-part-3-changes-where-the-context-block-goes).

## Prerequisites

- Completed: L2-M1.2.1 (Multi-Turn Conversations with LangChain + LangGraph)
- Completed: L2-M1.1.2 (DeepAgents) — the built-in toolkit and the backends
- MLflow server running at <http://127.0.0.1:5555>, version 3.15 or later
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`), with Unsloth
  Studio serving `gemma-4-26B-A4B-it-qat` behind the `gemma-agent` alias

## Concepts

### A deep agent has three things to remember

| What | Where it lives | Keyed by |
|:--|:--|:--|
| messages | graph state | `thread_id` |
| todos | graph state | `thread_id` |
| files (`StateBackend`, the default) | graph state | `thread_id` |
| files (`FilesystemBackend`) | **the disk** | the path, and nothing else |
| files (`StoreBackend`) | **a LangGraph store** | a namespace **you** choose |

The first three rows behave the same, because they are the same thing: entries
in the checkpointed state. One `checkpointer=` argument carries all of them.

The last two rows are the odd ones, and they are the point of this lesson. Both
cross the thread boundary. Only one of them was asked to.

### Why the disk row is a trap

A checkpointer is a set of lockers, one per `thread_id`. It can scope anything
it stores. It cannot scope something it does not store.

`FilesystemBackend` writes real bytes to a real directory. That directory has no
idea a thread exists. So two conversations pointed at the same `root_dir` share
one filesystem, and the second one can read what the first one wrote. In a
product with more than one user, that is a data leak, not an inconvenience.

The fix is not a different checkpointer. It is one of:

- keep files in `StateBackend`, where the thread already scopes them, or
- give each session its own `root_dir`, so the isolation is in the path, or
- route the paths that are *meant* to be shared to a store, and leave the rest
  in state — which is the store row.

### Why the store row is not

`StoreBackend` keeps files in a LangGraph `BaseStore`, under a namespace tuple
that you pass in. That tuple is the whole scope. `("memories",)` has no thread
in it, so every thread reads the same files. `(user_id, "memories")` makes them
per-user. The sharing is exactly as wide as the key you chose.

`CompositeBackend` is what keeps it from being the disk trap all over again. It
routes by path prefix: `/memories/` goes to the store, every other path goes to
`StateBackend`. One agent, two scopes, and the path decides which one applies.
Thread B can read what thread A left under `/memories/` and cannot read what it
left at `/` — the same test as the trap, with the verdict chosen per file.

### The session id has not changed

Everything from L2-M1.2.1 still applies: `thread_id` is what the agent
remembers, `session_id` is what groups the traces, and setting both to the same
value is what lets you follow one conversation end to end.

## Step-by-Step

### Step 1: One argument gives the deep agent memory

```python
agent = create_deep_agent(
    model=get_llm(),
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),  # the only change from L2-M1.1.2
)
```

### Step 2: Run the conversation, and watch the state grow

Each turn sends only the new message and reports what the thread is holding:

```python
state = agent.invoke(
    {"messages": [{"role": "user", "content": text}]},
    config={"recursion_limit": 50, "configurable": {"thread_id": session_id}},
)
files = {path: data.get("content", "") for path, data in state.get("files", {}).items()}
```

The `files` channel is read straight out of the returned state. That is stronger
evidence than asking the model whether it remembers — you are looking at the
state itself.

### Step 3: The control

Ask turn 2's question on a thread that has never been used. Same agent, same
checkpointer, so the only variable is the thread:

```text
On the thread  : The very first line of `/research.md` is:
RESEARCH-NOTE-7413
On a new thread: The file /research.md does not exist.
Files on the new thread : none
```

The question ends with "If the file does not exist, say so", and that clause is
load-bearing. Without it, gemma-4 on a fresh thread does not accept that the
file is gone: it hunts with `ls`, `glob` and `read_file` — thirteen calls in one
measured run — and then starts a model call that never ends. A control that
cannot finish proves nothing, so the prompt gives the agent a way to stop.

### Step 4: Swap the backend, and the isolation disappears

Nothing changes except one argument:

```python
agent = create_deep_agent(
    model=get_llm(),
    system_prompt=SYSTEM_PROMPT,
    backend=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
    checkpointer=InMemorySaver(),  # still there, and it does not help
)
```

Thread A writes the file. Thread B — a different `thread_id`, a different MLflow
session, which never wrote anything — reads it back.

`virtual_mode=True` keeps the agent's paths rooted at `/`, so the tool calls
look identical to Part 1. Only the bytes land somewhere else.

### Step 5: Route one prefix to a store, and share on purpose

Again nothing changes but the backend — and this time the agent also gets the
store the backend writes to:

```python
store = InMemoryStore()
backend = CompositeBackend(
    default=StateBackend(),
    routes={"/memories/": StoreBackend(store=store, namespace=lambda _runtime: ("memories",))},
)
agent = create_deep_agent(
    model=get_llm(),
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
    checkpointer=InMemorySaver(),
    store=store,
)
```

Thread A writes `/memories/prefs.md` and `/scratch.md`, each with its own marker
line. Thread B — a new `thread_id`, a new session — is asked for both. It quotes
the first and reports the second as missing. Both outcomes are logged as
metrics: `shared_memory_read` and `cross_thread_read`, the latter the same name
Step 4 used, so the two runs read side by side in the UI.

Two details the code depends on:

- `store=` is passed twice. The backend gets it so the lesson can read the store
  back after the turn; the graph gets it because that is where LangGraph expects
  it. Leave the first out and `StoreBackend` fetches it from the running graph,
  which works inside a turn and fails the moment you try to inspect it outside.
- The route prefix is stripped before the key reaches the store. Ask the store
  for `("memories",)` and you get `/prefs.md`, not `/memories/prefs.md`.

### Step 6: Read the conversations back

```python
sessions = mlflow.search_sessions(locations=[experiment.experiment_id], include_spans=False)
```

Six sessions come back from one run: the real conversation with four traces,
and five one-trace sessions from the control, the two disk threads and the two
store threads.

### Step 7: Part 3 changes where the context block goes

This is the one place where the two approaches stop being interchangeable, and
it is worth the whole second script.

`@mlflow.trace` stamps **one trace**. The session id is a function argument, so
one runner can stamp two different sessions:

```python
@mlflow.trace(name="turn", span_type="AGENT")
def run_turn(text: str, session_id: str) -> dict:
    mlflow.update_current_trace(session_id=session_id, user=USER)
    return plain(text, session_id)
```

`mlflow.tracing.context()` stamps a **region**, and a region cannot hold two
session ids. Parts 1 and 2 are one session each, so one block wraps the whole
part:

```python
with mlflow.tracing.context(session_id=state_session, user=USER):
    rows = run_conversation(make_plain_runner(agent), state_session, label)
```

Both halves of Part 3 write on thread A and read on thread B — two threads, two
sessions, inside one function. There is no single region to wrap, so the block
has to move down to one turn:

```python
def run_turn(text: str, session_id: str) -> dict:
    with mlflow.tracing.context(session_id=session_id, user=USER):
        return plain(text, session_id)
```

Rule of thumb: **stamp per trace when the session changes with the call, stamp
per region when it does not.** The decorator never has to care; the block does.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/2_conversation/2_deepagents
uv sync
uv run python main_decorator.py
uv run python main_context.py
```

Run both, then compare the Part 4 previews below and the Sessions tab in the
UI. `conversation.py` is imported by both and is not runnable on its own.

Measured end to end: under two minutes per script. Turn 3 is the delegating one
and takes most of Part 1 on its own.

## Expected Output

The file grows across the turns and is still there at the end:

```text
  Turn 1  user : Use write_todos to plan your work, then research microservices architecture with search_
          agent: I have completed the research on microservices architecture. I've gathered information on its core p
          files: ['/research.md', '/todos.txt'] (886 chars) | todos 0
          4 tool call(s), 0 handoff(s) | 10 messages | 8.87s

  Turn 2  user : Read /research.md and tell me its very first line, word for word. If the file does not e
          agent: The very first line of `/research.md` is:
RESEARCH-NOTE-7413
          files: ['/research.md', '/todos.txt'] (886 chars) | todos 0
          1 tool call(s), 0 handoff(s) | 14 messages | 2.07s

  Turn 3  user : Ask the 'analyst' sub-agent to turn the findings in /research.md into a recommendation,
          agent: I have analyzed the research findings using the 'analyst' sub-agent and saved the resulting recommen
          files: ['/analysis.md', '/research.md', '/todos.txt'] (3540 chars) | todos 0
          3 tool call(s), 1 handoff(s) | 22 messages | 19.09s

  Turn 4  user : What was the very first thing I asked you to do?
          agent: The very first thing you asked me to do was to use `write_todos` to plan my work, research microserv
          files: ['/analysis.md', '/research.md', '/todos.txt'] (3540 chars) | todos 0
          0 tool call(s), 0 handoff(s) | 24 messages | 1.60s
```

Part 3a is the leak:

```text
  Thread A (cnv-dec-disk-a-29b4516e) — researched and wrote the file
    answer: I have completed the research on microservices architecture. I've saved the findings, including bene

  Thread B (cnv-dec-disk-b-da70a152) — never wrote anything
    answer: The first line of /research.md is `RESEARCH-NOTE-7413`.

  Files on disk: ['research.md', 'todos.txt']
  Thread B quoted thread A's RESEARCH-NOTE-7413: True
```

Part 3b is the same test, answered per path:

```text
  Thread A (cnv-dec-memory-a-dd2df829) — wrote one file into each scope
    answer      : I have created the requested files at /memories/prefs.md and /scratch.md.
    in the state: ['/scratch.md']   (scoped by thread)
    in the store: ['/prefs.md']   (scoped by namespace ('memories',))

  Thread B (cnv-dec-memory-b-80568188) — never wrote anything
    answer      : The first line of `/memories/prefs.md` is "MEMORY-NOTE-2291", and the file `/scratch.md` does not exist.
    quoted MEMORY-NOTE-2291 from /memories/prefs.md: True
    quoted SCRATCH-NOTE-5806 from /scratch.md: False
```

Thread A's state holds only `/scratch.md`; the store holds only `/prefs.md`,
with the route prefix stripped. Thread B quotes the store file and reports the
state file as missing — one agent, two scopes, decided by the path.

`cross_thread_read` is logged as a metric by both halves, so the leak is a
number you can assert on in a test rather than a paragraph in a README: 1 in
`*_filesystem_leak`, 0 in `*_store_share`. The share logs its own,
`shared_memory_read`, as 1.

Both scripts produce those results identically. The difference shows up in
Part 4, in what the trace previews say. `main_decorator.py`:

```text
  Session cnv-dec-state-1ee36f58 — 4 traces
    1. {"text": "Use write_todos to plan your work, then research microservices
       -> {"answer": "I have completed the research on microservices architecture.
```

`main_context.py`, same conversation:

```text
  Session cnv-ctx-state-c1e9f10f — 4 traces
    1. Use write_todos to plan your work, then research microservices architect
       -> {"messages": [{"content": "Use write_todos to plan your work, then resea
    2. Read /research.md and tell me its very first line, word for word. If the
       -> {"messages": [{"content": "Use write_todos to plan your work, then resea
```

The decorator's root span is `run_turn`, so the preview is that function's
arguments and its return value — including the file summary. The context
script's root span is autolog's, so the preview is the raw agent input and the
whole message list back. The response preview is where it hurts most: every
turn looks the same, because the message list always starts with turn 1.

> **Note on todos.** `write_todos` is offered to the model, not forced. The
> gemma-4 behind `gemma-agent` usually plans straight into a `/todos.txt` file
> instead, so the todo count stays 0 while the file list shows the plan. The
> state channel works — this model just does not reach for it. A stronger model
> picks the dedicated tool more often.
>
> **Note on the ceiling.** `get_llm()` sets `max_completion_tokens=4096` and a
> five-minute timeout. Gemma 4 reasons before it answers, the reasoning is
> charged against the ceiling, and on some prompts at temperature 0 it never
> stops. The server's own ceiling is the whole 262k context, which was measured
> here as one forty-minute turn. With the cap a runaway turn fails in about a
> minute, and the client gives up instead of retrying a twenty-minute gateway
> timeout three times over.

## Key Takeaways

- One `checkpointer=` argument carries messages, todos and `StateBackend` files
  together, because all three are entries in the same graph state.
- A checkpointer scopes what it stores. `FilesystemBackend` writes to disk,
  which it does not store, so the thread boundary does not apply.
- Cross-session file leaks are a backend choice, not a bug. Isolate with
  `StateBackend` or with a per-session `root_dir`.
- `StoreBackend` crosses threads too, but by a namespace you chose, and
  `CompositeBackend` limits it to a path prefix. The leak in 3a and the share in
  3b are the same mechanism. The difference is that one of them was designed.
- Read the state channel (`state["files"]`) to verify persistence. Asking the
  model whether it remembers tests the model, not the plumbing.
- `mlflow.search_sessions()` returns each thread as its own conversation, so the
  leak in Part 3 is visible as two separate one-trace sessions holding the same
  content.
- The choice between `@mlflow.trace` and `mlflow.tracing.context()` is an MLflow
  choice, not a framework one. It applies unchanged to a deep agent, and it will
  apply unchanged to the Claude Agent SDK in L2-M1.2.3.
- `@mlflow.trace` stamps a trace; `context()` stamps a region. When the session
  id changes from call to call — as it does in Part 3 — the decorator needs no
  change and the block has to move inside the turn.
- Owning the root span costs a wrapper function and buys readable previews. On a
  deep agent that is worth more than it was in L2-M1.2.1, because the thing you
  most want to see per turn is which files the thread was holding.

## Next Steps

**L2-M1.2.3 — Claude Agent SDK Conversations** moves to a framework that owns
its own session id, so the job changes: instead of inventing an id and pushing
it down, you take the id the SDK already has and stamp MLflow with it.
