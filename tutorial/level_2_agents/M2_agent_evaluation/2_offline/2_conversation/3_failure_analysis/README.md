# L2-M2.2.2.3 — Where Conversations Break Down: Funnel and Slices

**Level:** AI Agents
**Duration:** 60 min

## Overview

A session scorer returns **one verdict for N turns**. That is enough to fail a
build and useless for fixing it — "the conversation failed" does not say whether
the agent fell over on turn 1 or held up until turn 4.

The funnel says which.

## Prerequisites

- Completed: [`../../1_turn/4_failure_analysis`](../../1_turn/4_failure_analysis/),
  [`1_conversation_gates`](../1_conversation_gates/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### The funnel is not a per-turn pass rate

Once a conversation has broken it **stays** broken, because the customer has
already had the bad experience. A later good turn does not undo it. So the
figure to read is "how many conversations are *still healthy* after turn N",
which only ever falls.

### The agent under test has a context budget

```python
HISTORY_WINDOW = 4
```

Truncating history is one of the most common production choices there is — it
caps token cost per turn and keeps latency flat as a conversation grows. It is
also never free, and **the funnel is how you find out what it cost**. Raise the
number and the cliff moves; remove it and the cliff disappears along with the
saving.

That makes the cliff in this lesson real rather than contrived: it is a
trade-off a team actually makes, being measured.

### Score every turn, not just the session

This analysis needs a per-turn pass flag and a turn number — which any
multi-turn suite already has. What is easy to omit is scoring **every turn**.
Score only the session and this reading is impossible afterwards.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/2_conversation/3_failure_analysis
uv sync
uv run python main.py
```

## Expected Output

What a session scorer would tell you:

```text
  1/4 conversations fully succeeded.
  One bit per conversation. True, and not actionable.
```

What the funnel tells you:

```text
  turn    still healthy   survival  dropped here
  ------ -------------- ---------- -------------
  1                   4       100%             0
  2                   4       100%             0
  3                   1        25%             3

  The cliff is TURN 3: 3 of 4 conversations died there.
```

And why:

```text
  slice                 convos  turn pass rate
  order_then_recall          2            0.67
  policy_only                2            0.83

  cause                          n   share
  lost_earlier_context           3   100%
```

**Compare the first block with the rest.** "1 of 4 succeeded" is a number you
can put in a report. "Conversations die at turn 3, and the cause is lost
context" is a number you can act on — it names a place in the code and a
trade-off to revisit.

## Key Takeaways

- A session verdict is **one bit for N turns**. It cannot localise a failure.
- The funnel localises it: which turn is the cliff.
- Survival only falls — a conversation that broke stays broken.
- Conversation-scope causes differ from turn-scope ones; `lost_earlier_context`
  is the characteristic one.
- **Score every turn**, or you cannot do this analysis later.
- The cliff here is a measured cost of a real production trade-off, not a bug.

## Next Steps

**[`4_multi_turn_benchmark`](../4_multi_turn_benchmark/)** — the same scope,
measured against a bar you do not own.
