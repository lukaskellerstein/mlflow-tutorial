# L2-M1.2.1 — Multi-Turn Conversations with LangChain + LangGraph

**Level:** AI Agents
**Duration:** 45 min

## Overview

L2-M1.1.1 ran three unrelated tasks through one agent. This lesson runs one
conversation of four turns, where turn 2 and turn 4 cannot be answered without
remembering turn 1. You give the agent memory with a LangGraph checkpointer, you
tie the traces together with an MLflow session id, and you read the whole
discussion back with `mlflow.search_sessions()`.

MLflow has **two** APIs for stamping a session, so this lesson is two scripts
that do the same job:

| File | What it is |
|:--|:--|
| `main_decorator.py` | approach 1 — `@mlflow.trace` + `mlflow.update_current_trace()` |
| `main_context.py` | approach 2 — a `mlflow.tracing.context()` block |
| `conversation.py` | everything the two hold fixed — imported by both, run by neither |

Read the two scripts side by side. They are short, and the diff between them is
the lesson.

## Prerequisites

- Completed: L2-M1.1.1 (LangChain + LangGraph Agents) — the agent built here is
  the same one, with a checkpointer added
- Completed: L1-M2 (Tracing)
- MLflow server running at <http://127.0.0.1:5555>, version 3.15 or later
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`), with Unsloth
  Studio serving `gemma-4-26B-A4B-it-qat` behind the `gemma-agent` alias

## Concepts

### Two keys, one conversation

A conversation needs two separate things to work, and they are owned by
different systems.

| Key | Owner | What it decides |
|:--|:--|:--|
| `thread_id` | LangGraph | what the **agent** remembers |
| `session_id` | MLflow | which traces belong to **one conversation** |

They are independent. Set only `thread_id` and the agent answers correctly, but
MLflow shows you four unrelated traces. Set only `session_id` and MLflow groups
them neatly, but the agent has amnesia. Both scripts set both, to the same
value, which is what you want in production: one id you can follow from the
user's browser through the agent and into the trace store.

### What a checkpointer actually does

A checkpointer is a saved copy of the graph's state, keyed by `thread_id`.
Think of a locker at a train station. The `thread_id` is the locker number.
Each turn the agent opens the locker, adds the new messages, and closes it
again.

That is why the code only ever sends the **new** message:

```python
state = agent.invoke(
    {"messages": [{"role": "user", "content": text}]},  # just this turn
    config={"configurable": {"thread_id": session_id}},  # the locker number
)
```

`state["messages"]` comes back holding the whole history, because
`add_messages` appended to what the checkpointer restored.

`InMemorySaver` keeps the lockers in RAM, so they vanish when the script ends.
Swap it for `SqliteSaver` or `PostgresSaver` and the conversation survives a
restart. Nothing else in either script changes.

### Two ways to stamp the session

| API | Where it goes | Covers | Root span is |
|:--|:--|:--|:--|
| `mlflow.update_current_trace(session_id=...)` | inside a `@mlflow.trace` function | the one trace it runs in | yours, and you named it |
| `mlflow.tracing.context(session_id=...)` | a `with` block around the calls | every trace opened in the block | autolog's |

This lesson pins MLflow 3.15 or later. The `session_id` and `user` arguments
themselves arrived in 3.12.0 — on anything older you write the metadata key by
hand: `mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})`.

The second is less code. The first gives you a clean trace, and the read-back at
the end of each script shows why that matters.

### Why the shared module exists

`conversation.py` holds the agent, the tools, the four turns, the run-level
logging, the control and the read-back. Neither script owns a copy.

That is not tidiness. The claim this lesson makes is *"the stamping API is the
only variable"*. If each script carried its own copy of the agent and the
logging, the two could drift, and a difference in the MLflow UI would stop being
evidence about the API — it could just be two files doing different work. One
shared module is what makes the comparison hold.

## Step-by-Step

### Step 1: Give the agent memory

One argument. Everything else is the L2-M1.1.1 agent.

```python
create_agent(model=get_llm(), tools=TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
```

Without it the agent is stateless and every `invoke` starts from an empty
message list. The checkpointer is a LangGraph feature, not a LangChain one —
`create_agent` hands it to `compile()` for you.

### Step 2: The turn, with no MLflow in it at all

`conversation.py` owns the call itself. The agent is bound in the closure, and
`seen` remembers how long the history was after the previous turn, so each turn
counts only its own tool calls.

```python
def make_plain_runner(agent) -> TurnFn:
    seen = 0

    def run_turn(text: str, session_id: str) -> dict:
        nonlocal seen
        state = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": session_id}},
        )
        ...

    return run_turn
```

Both scripts start from this function. What each does to it is the difference.

### Step 3: Approach 1 — own the trace

`main_decorator.py` wraps the plain runner, so the turn becomes the root of its
own trace and stamps itself:

```python
def make_turn_runner(agent) -> TurnFn:
    plain = make_plain_runner(agent)

    @mlflow.trace(name="turn", span_type="AGENT")
    def run_turn(text: str, session_id: str) -> dict:
        mlflow.update_current_trace(session_id=session_id, user=USER)
        return plain(text, session_id)

    return run_turn
```

`@mlflow.trace` records **every argument** as the span's input. Pass the agent in
and every trace preview reads
`{"agent": "<CompiledStateGraph object at 0x117034e50>", ...}`. Binding it in the
closure keeps the trace inputs to the two values a reader cares about.

### Step 4: Approach 2 — own nothing

`main_context.py` adds no wrapper. The plain runner goes in unchanged, and one
`with` block does the stamping:

```python
with mlflow.tracing.context(session_id=conversation_session, user=USER):
    rows = run_conversation(make_plain_runner(agent), conversation_session, label)
```

Every trace opened inside the block gets the session id, including the ones
LangChain autolog creates on its own.

### Step 5: The control — prove what carries the memory

Both scripts ask turn 2's question again with the same agent and the same
checkpointer, but a thread id that has never been used:

```text
Question       : Now double that number.
On the thread  : 690 is the result.
On a new thread: Could you please specify which number you would like me to double?
```

One variable changed. The thread id is the memory.

### Step 6: Read the conversation back

```python
sessions = mlflow.search_sessions(locations=[experiment.experiment_id], include_spans=False)
for session in sessions:
    for trace in session:  # already oldest first
        print(trace.info.request_preview, "->", trace.info.response_preview)
```

`search_sessions` returns one `Session` per conversation, not one row per trace.
A `Session` supports `len()`, indexing and iteration, and its traces arrive
already sorted oldest first — so iterating it replays the discussion in order.

## Running the Lesson

Run both. They log to the same experiment, so the four sessions land side by
side in the UI.

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/2_conversation/1_langchain_langgraph
uv sync
uv run python main_decorator.py
uv run python main_context.py
```

## Expected Output

Four turns, each building on the last — identical in both scripts:

```text
  Turn 1  user : What is 15 * 23?
          agent: 15 * 23 is 345.
          1 tool call(s), history now 4 messages, 0.73s

  Turn 2  user : Now double that number.
          agent: 690 is the result.
          1 tool call(s), history now 8 messages, 0.70s

  Turn 4  user : What was the very first thing I asked you?
          agent: The first thing you asked was "What is 15 * 23?".
          0 tool call(s), history now 14 messages, 0.37s
```

The history grows by 4 messages a turn (user, assistant-with-tool-call, tool
result, assistant) and by 2 on turn 4, which needs no tool.

The read-back is where the two scripts part company. `main_decorator.py`:

```text
  Session cnv-decorator-cd05098a — 4 traces
    1. {"text": "What is 15 * 23?", "session_id": "cnv-decorator-cd05098a"}
       -> {"answer": "15 * 23 is 345.", "tool_calls": 1, "history_messages": 4}
    2. {"text": "Now double that number.", "session_id": "cnv-decorator-cd05098
       -> {"answer": "690 is the result.", "tool_calls": 1, "history_messages": 8}

  Session cnv-decorator-control-fed230b5 — 1 trace
    1. {"text": "Now double that number.", "session_id": "cnv-decorator-control
       -> {"answer": "Could you please specify which number you would like me to d
```

`main_context.py`, same four turns, same answers:

```text
  Session cnv-context-6463fb70 — 4 traces
    1. What is 15 * 23?
       -> {"messages": [{"content": "What is 15 * 23?", "additional_kwargs": {}, "
    2. Now double that number.
       -> {"messages": [{"content": "What is 15 * 23?", "additional_kwargs": {}, "
```

Both scripts logged the same params, the same three metrics and the same
`conversation.json` table, and both drove the same agent through the same four
turns — `conversation.py` guarantees it. The trace is the only thing that came
out different: approach 2's is rooted in autolog's span, so the response preview
is the raw LangChain message list, far harder to read. That is the cost of the
shorter API, and because nothing else was allowed to vary, it is the only cost.

One number is not comparable. `avg_latency` can be many times higher in whichever
script you run first, because that run waits for Unsloth to load the model. Run
both twice and the averages converge — on a warm model they land within a few
tenths of a second of each other.

In the MLflow UI, the left nav has a **Sessions** tab
(`#/experiments/<id>/chat-sessions`). It lists one row per conversation with its
first request and last response, and opening a row replays the turns in order.
The **Traces** tab has a **Group by session** button that does the same grouping
in place. Every trace also carries the user.

## Key Takeaways

- `thread_id` gives the agent memory, `session_id` groups the traces. They are
  different systems and both are needed. Use the same value for both.
- A checkpointer means you send only the new message; the framework restores the
  rest. `InMemorySaver` for a demo, `SqliteSaver` or `PostgresSaver` for real.
- `@mlflow.trace` serializes every argument into the trace. Close over the big
  objects instead of passing them.
- `mlflow.tracing.context()` is the short way to stamp a session, but the root
  span stays autolog's and the previews get noisy.
- Hold everything else fixed when you compare two approaches. Both scripts import
  the same module, so the trace is the only difference you can see — and
  therefore the only one you have to explain.
- `mlflow.search_sessions()` returns whole conversations, already in order —
  that is what you evaluate in L2-M2.1.2.

## Next Steps

**L2-M1.2.2 — DeepAgents Conversations** keeps the same checkpointer idea but
adds what makes DeepAgents different: the todo list and the virtual filesystem
also survive the turn boundary, so state carries, not just messages.
