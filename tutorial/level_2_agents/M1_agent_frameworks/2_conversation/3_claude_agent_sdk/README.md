# L2-M1.2.3 — Multi-Turn Conversations with the Claude Agent SDK

**Level:** AI Agents
**Duration:** 45 min

## Overview

The other two lessons in this group make you supply the session key: you invent
a `thread_id` and hand it to a checkpointer. The Claude Agent SDK works the
other way round — it owns the conversation and assigns the id itself, then tells
you what it was after the turn finished. This lesson reads that id out and
stamps MLflow with it, then uses `resume` to continue the same conversation from
a brand-new client.

## Prerequisites

- Completed: L2-M1.2.1 and L2-M1.2.2 — the `thread_id` / `session_id` split
- Completed: L2-M1.1.3 (Claude Agent SDK) — hand-built tracing and in-process
  MCP tools
- MLflow server running at <http://127.0.0.1:5555>, version 3.15 or later
- A working Claude Code CLI login. **This lesson does not use the gateway or any
  local model**, and it spends real subscription usage — see Cost below.

## Concepts

### The id goes the other way

| Framework | Who picks the session key | When you learn it |
|:--|:--|:--|
| LangGraph / DeepAgents | you do | before the turn |
| Claude Agent SDK | the SDK does | after the turn, in `ResultMessage` |

That changes the shape of the code. A LangGraph turn takes `session_id` as an
argument. An SDK turn cannot, because the value does not exist yet:

```python
@mlflow.trace(name="turn", span_type="AGENT")
async def run_turn(text: str) -> dict:      # no session_id parameter
    ...
    elif isinstance(message, ResultMessage):
        result.session_id = message.session_id   # it arrives here, at the end
    ...
    mlflow.update_current_trace(session_id=result.session_id, user=USER)
```

Stamping on the way out is legal. `update_current_trace` only has to run before
the trace closes, not before it opens.

### The client is the conversation

There is no checkpointer here and no `thread_id`. What keeps the context is the
open `ClaudeSDKClient`:

```python
async with ClaudeSDKClient(options=build_options()) as client:
    for text in CONVERSATION:
        await run_turn(text)  # same client, same conversation
```

Open a new client and you get a new conversation with a new id. That is exactly
what L2-M1.1.3 did for every single query — one fresh client per call — which is
why that lesson had no memory. It was not an oversight; it was the shape of the
code.

### Resume is the payoff

The SDK persists sessions on its own. So a completely new client, in a new
process, on a later day, picks the conversation back up with one option:

```python
ClaudeAgentOptions(..., resume=session_id)
```

No checkpoint store, no database, no `SqliteSaver`. This is the one thing this
framework hands you that the LangGraph-based ones make you build.

## Step-by-Step

### Step 1: Three in-process tools

The same calculator, string reverser and word counter as the sibling lessons, so
the three frameworks stay comparable. They run in this process, so no MCP
subprocess and no settle delay:

```python
INPROC_SERVER = create_sdk_mcp_server(
    name="inproc",
    version="1.0.0",
    tools=[calculator, string_reverser, word_counter],
)
```

Every prompt names its tool. Without that Claude does the arithmetic itself and
the tool spans stay empty.

### Step 2: One client, four turns

The client is opened once, outside the loop. The turn runner closes over it
rather than taking it as an argument, because `@mlflow.trace` serializes every
argument into the span's input.

### Step 3: Read the id out, stamp the trace

The count is the assertion: four turns, one distinct session id.

```text
Distinct session ids across 4 turns: 1
```

It is logged as the `distinct_session_ids` metric, so a test can assert on it.

### Step 4: The control

The same question from a client that has never been used:

```text
On the session  : 690.
On a new client : Which number are you referring to — I don't see one from earlier ...
```

### Step 5: Resume, and check the id survived

```python
async with ClaudeSDKClient(options=build_options(resume=session_id)) as client:
    result = await make_turn_runner(client)(RESUMED_QUESTION)
```

`session_id_preserved` is logged as a metric. It is 1 when the resumed client
reports the same id, which means MLflow files the resumed turn into the original
conversation with no extra work.

## Cost

Every turn is a real, billed Claude call. The lesson runs six of them: four in
Part 1, one control, one resumed. `max_budget_usd = 1.0` is set on the options
as a stop, not a target — a normal run costs far less. There is no local
fallback: if the Claude Code CLI is not logged in, the lesson fails rather than
quietly using another model.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/2_conversation/3_claude_agent_sdk
uv sync
uv run python main.py
```

## Expected Output

```text
  Turn 3  user : Use the string_reverser tool on the word 'MLflow'.
          agent: wolfLM
          tools: ['string_reverser'] | session 3cb0abb9… | 3277ms

  Turn 4  user : What was the very first thing I asked you?
          agent: You asked me to use the calculator tool to compute 15 * 23.
          tools: none | session 3cb0abb9… | 1615ms

  Distinct session ids across 4 turns: 1
```

Part 3 continues that same conversation from a new client:

```text
  Resumed from   : 3cb0abb9-178d-4c47-ad78-8441abdbcf74
  Question       : What did the string_reverser tool return earlier?
  Answer         : It returned "wolfLM".
  Same session id: True
```

And Part 4 shows why that matters — the resumed turn is filed into the original
conversation, so the session holds **five** traces, not four:

```text
  Session 3cb0abb9-178d-4c47-ad78-8441abdbcf74 — 5 traces
    1. {"text": "Use the calculator tool: what is 15 * 23?"}
    ...
    5. {"text": "What did the string_reverser tool return earlier?"}
       -> {"answer": "It returned \"wolfLM\".", ...
```

## Key Takeaways

- The Claude Agent SDK assigns the session id; you read it from
  `ResultMessage.session_id` and stamp the trace on the way out.
- An open `ClaudeSDKClient` is the conversation. A new client is a new session,
  which is why L2-M1.1.3 had no memory.
- `resume=<session_id>` continues a conversation from any new client, with no
  checkpoint store to run — the SDK persists it for you.
- A resumed turn keeps the same session id, so MLflow files it into the original
  conversation automatically.
- Assert on `distinct_session_ids` and `session_id_preserved`. A conversation
  that silently split into four sessions still prints plausible answers.

## Next Steps

That closes L2-M1. You have built the same conversation three ways, and grouped
all three in MLflow with a session id.

**L2-M2 — Agent Evaluation** now judges what you have been tracing. Its
`1_instruments/2_conversation/` group scores whole sessions rather than single
turns, and it reads them with the same `mlflow.search_sessions()` used here.
