# L2-M2.1.2.3 — Judging a Whole Conversation

**Level:** AI Agents
**Duration:** 90 min

## Overview

The two lessons before this one produce conversations and grade nothing. This is
the one that turns a conversation into a verdict, and it does so two ways: in
**words** with `discover_issues()`, and in **numbers** with session-level
scorers. Both take the whole session, which is what lets them ask questions no
turn-level scorer can.

## Prerequisites

- Completed: [`1_conversation_simulation`](../1_conversation_simulation/),
  [`../../1_turn/3_quality_metrics`](../../1_turn/3_quality_metrics/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Three failures that only exist at conversation scope

- The agent answers every question adequately and never finishes the task.
- The agent forgets on turn 4 what the user told it on turn 1.
- The user is visibly getting more frustrated with every reply.

None of those is a function of `(inputs, outputs)`. Session scorers take
`session=list[Trace]` instead.

The scripted conversation ends with **"Remind me which day my order was
arriving?"** — deliberately, without restating the day. That is the only way to
see whether the agent retained turn 1.

### All seven built-ins

| Scorer | Good answer |
|:--|:--|
| `ConversationCompleteness` | `yes` |
| `KnowledgeRetention` | `yes` |
| `ConversationalToolCallEfficiency` | `yes` |
| `ConversationalRoleAdherence` | `yes` |
| `ConversationalSafety` | `yes` |
| `ConversationalGuidelines` | `yes` |
| `UserFrustration` | **`none`** |

Every one takes `model=`, so every one is a **judge**.

### Writing your own needs no LLM

A `@scorer` function becomes session-level purely because its parameter is named
`session`. That is the whole opt-in:

```python
@scorer
def turn_count(session: list[Trace]) -> Feedback:
    return Feedback(value=float(len(session)), rationale=f"{len(session)} turns.")
```

It counts, so it is a scorer and not a judge. Session scope does not imply LLM.

## Step-by-Step

### Step 1: Own the root span and stamp the session id

```python
@mlflow.trace(name="conversation_turn")
def run_turn(history, session_id):
    mlflow.update_current_trace(session_id=session_id)
    return AGENT.invoke({"messages": history})
```

Collecting traces is not enough. Without the id the scorer raises
`All traces in 'session' must have a session_id`.

### Step 2: Flush before reading traces back

```python
mlflow.flush_trace_async_logging()
```

Traces export asynchronously. Fetching one straight after the turn that produced
it races the exporter.

### Step 3: Score in numbers

```python
feedback = session_scorer(session=traces)  # not inputs=/outputs=
```

### Step 4: Score in words

```python
from mlflow.genai.discovery.pipeline import discover_issues

result = discover_issues(traces=traces, model=SESSION_MODEL, max_issues=5)
```

> [!note]
> `discover_issues` is **not** in `mlflow.genai.discovery`'s `__all__`.
> `test_agent()` imports it exactly this way, and it is the supported entry
> point. It is called directly here so you meet stage 4 before meeting the
> wrapper.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/2_conversation/3_judging_conversations
uv sync
uv run python main.py
```

> [!note]
> Expect 10–15 minutes. Four agent turns, then eight scorers each reading the
> whole conversation, then issue discovery.

## Expected Output

```text
    scorer                               value    rationale
    ------------------------------------ -------- --------------------------------
    conversation_completeness            ERROR    Failed to parse response from ...
    user_frustration                     none     The user's questions are direct ...
    knowledge_retention                  yes      Knowledge retention across 4 turn(s)
    conversational_tool_call_efficiency  yes      The agent used tools efficiently ...
    conversational_role_adherence        yes      The assistant consistently acts ...
    conversational_safety                yes      The assistant's responses are ...
    conversational_guidelines            no       The assistant failed to comply ...
    turn_count                           4.0      4 turns in this session.
```

Three things in that output are worth reading closely.

**`conversational_guidelines` said `no`, and it was right.** The guideline was
"always cite the policy reference code". The agent answered "Return shipping is
free for faulty items" without citing P-330. A real violation, caught.

**`conversation_completeness` errored.** A small local model returned prose where
the judge wanted structured output. Note what the lesson does **not** do: it logs
no metric for that scorer. A judge that failed to answer is not a score of zero,
and recording it as one would quietly drag the average down.

**`turn_count` returned `4.0` while its neighbours returned words.** That is the
scorer/judge split, visible in one table.

> [!warning]
> **These scorers share neither a value vocabulary nor a polarity.** This run
> returned `yes`, `no`, `none` and `4.0`. For `user_frustration`, zero is the
> *good* outcome; for the other six, one is. Never sum them into a single
> "session score" — you would be adding a metric that improves as it falls to
> six that improve as they rise. Each is logged separately under
> `session/<name>` for exactly this reason.

## Key Takeaways

- Session scorers take `session=list[Trace]` and reject the single-turn
  parameters outright.
- Every trace in that list needs a `session_id`, and traces export
  asynchronously — flush first.
- These judges answer with **strings**. `bool("no")` is `True`, so coercing with
  `bool()` scores every failure as a pass.
- Polarity is not uniform. Log each scorer separately; never sum them.
- A judge that could not parse its own answer is not a zero.
- `discover_issues()` judges in words; session scorers judge in numbers. "0
  issues" is not proof — discovery is itself an LLM judge.

## Next Steps

**[`4_test_agent`](../4_test_agent/)** — the simulator and `discover_issues()`
in one call, from nothing but the agent itself.
