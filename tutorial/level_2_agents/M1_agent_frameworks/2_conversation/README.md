# L2-M1.2 — Conversational Agent Frameworks

**Many turns, one session.** The same three frameworks as
[`1_turn/`](../1_turn/), now carrying state across a turn boundary and grouped
in MLflow so you can read the whole discussion back.

Each lesson runs the same four-turn conversation, so the frameworks stay
directly comparable:

```text
   turn 1   "What is 15 * 23?"                    -> 345
   turn 2   "Now double that number."             -> 690      needs turn 1
   turn 3   "Reverse the word 'MLflow'."          -> wolfLM
   turn 4   "What was the very first thing I asked you?"      needs turn 1
```

Turns 2 and 4 are the test. A stateless agent answers turn 1 and turn 3 exactly
as well and fails the other two, so the conversation proves the memory rather
than assuming it.

## The three lessons

| Lesson | Memory mechanism | The specific thing it teaches |
|:--|:--|:--|
| [`1_langchain_langgraph`](1_langchain_langgraph/) | `InMemorySaver` keyed by `thread_id` | the two keys, and two ways to stamp a session |
| [`2_deepagents`](2_deepagents/) | the same checkpointer | files and todos carry too — one backend leaks them across sessions, another shares one path prefix on purpose |
| [`3_claude_agent_sdk`](3_claude_agent_sdk/) | the open client, plus `resume` | the framework picks the session id, so you read it out instead of pushing it in |

## Two ways to stamp a session, in two of the lessons

MLflow has two APIs for tying traces to a conversation, and the first two
lessons each ship **two scripts** that differ only in which one they use:

| File | Approach |
|:--|:--|
| `main_decorator.py` | `@mlflow.trace` + `mlflow.update_current_trace()` — you own the root span |
| `main_context.py` | a `mlflow.tracing.context()` block — you own no span |
| `conversation.py` | everything the two hold fixed, imported by both |

The repetition is the lesson. These APIs belong to MLflow, so the same choice
lands the same way on a LangChain agent and on a deep agent, and the shared
module is what proves the difference in the UI comes from the stamping and not
from two files doing different work.

`2_deepagents` then pushes it one step further. `@mlflow.trace` stamps a
**trace** and `context()` stamps a **region**, which look interchangeable until
a region has to hold two session ids — exactly what that lesson's Part 3 does
when it writes on one thread and reads on another.

`3_claude_agent_sdk` stays at one script, because there the session id is not
yours to choose and the interesting problem moves elsewhere.

## Every lesson has the same four parts

This shape is deliberate. Compare the same part across lessons and the framework
differences stand out on their own.

| Part | What it does |
|:--|:--|
| 1 | run the conversation, one trace per turn, all stamped with one session id |
| 2 | **the control** — repeat one turn without the memory, and watch it fail |
| 3 | the framework-specific twist |
| 4 | read the conversation back with `mlflow.search_sessions()` |

Part 2 is the one to not skip. It is easy to write a lesson where the agent
answers correctly for the wrong reason; the control changes exactly one variable
and shows what was carrying the state.

## Seeing a conversation in the UI

The left nav carries a **Sessions** tab, at
`#/experiments/<id>/chat-sessions`. It lists one row per conversation — the
session id, the first request, the last response — and opening a row replays the
turns in order. The **Traces** tab has a **Group by session** button that applies
the same grouping without leaving the list.

Both read the `mlflow.trace.session` metadata key, which is what `session_id`
sets. A trace without it never appears in either view.

## Reading a conversation back in code

```python
sessions = mlflow.search_sessions(locations=[experiment.experiment_id], include_spans=False)
for session in sessions:
    for trace in session:  # already oldest first
        print(trace.info.request_preview, "->", trace.info.response_preview)
```

`search_sessions` returns one `Session` per conversation rather than one row per
trace. A `Session` supports `len()`, indexing and iteration, and its traces
arrive sorted oldest first — so iterating one replays the discussion in order.

**Keep the traced function's arguments small.** `@mlflow.trace` records every
argument as the span's input, so passing the agent or the client in makes every
preview read `<CompiledStateGraph object at 0x...>`. All three lessons bind
those in a closure instead, which is what makes the read-back above legible.

## Requirements

MLflow 3.15 or later, which is what these lessons pin. The `session_id` and
`user` arguments — on both `mlflow.update_current_trace()` and
`mlflow.tracing.context()` — arrived in 3.12.0. On anything older you must write
the metadata key by hand:

```python
mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
```
