# L2-M2.2.2.2 — Comparing Two Agent Versions Across Conversations

**Level:** AI Agents
**Duration:** 60 min

## Overview

Same question as [`../../1_turn/3_version_comparison`](../../1_turn/3_version_comparison/)
— ship v2 or keep v1 — asked at conversation scope, where the method from that
lesson sometimes works and sometimes cannot.

## Prerequisites

- Completed: [`../../1_turn/3_version_comparison`](../../1_turn/3_version_comparison/),
  [`1_conversation_gates`](../1_conversation_gates/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Which method you get depends on who plays the user

| User side | What happens | Method |
|:--|:--|:--|
| **Scripted** — fixed strings you wrote | Both versions face the identical conversation | **Paired**: win/loss/tie, sign test |
| **Simulated** — a model that reacts | v2 answers turn 1 differently, so the user asks a different turn 2 | **Distributions** over k runs |

That is not a detail. It decides which statistics you are entitled to use, and
it is the reason this lesson exists separately from the turn one.

### Why simulated conversations cannot be paired

A paired test needs the same input twice. By turn 3 the two versions are not
answering the same conversation at all — the user diverged in response to them.
Run 1 of v1 and run 1 of v2 are simply two different conversations, so there is
nothing to match and no win/loss to count.

The lesson prints the opening user turn of every simulated run so you **see**
the divergence rather than being told about it.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/2_conversation/2_version_comparison
uv sync
uv run python main.py
```

> [!note]
> Expect 10–20 minutes. Part 1 is cheap; Part 2 runs k simulated conversations
> per version with a second model playing the user.

## Expected Output

```text
  Part 1: SCRIPTED user -- both versions face identical turns
  conversation             v1     v2  result
  delayed_order          1.00   1.00  tie
  faulty_return          1.00   1.00  tie
  paired: 0 win / 0 loss / 2 tie   sign test p = 1.000
```

**Everything tied, and that is worth reading rather than skipping.**

> [!important]
> Session scorers answer yes/no — they are **coarse**. Two competent versions
> tie on a two-conversation suite almost every time. A tie means the suite lacks
> the **resolution** to separate them, not that the versions are equivalent. Add
> conversations, or score per turn as well —
> [`3_failure_analysis`](../3_failure_analysis/) is exactly that.

Then Part 2:

```text
  version      runs    mean   spread
  v1              2    1.00     0.00
  v2              2    1.00     0.00

  delta of means: +0.00
```

> [!warning]
> **With 2 runs per version you cannot tell a delta from noise** — the spread is
> the same size as any effect you care about. A distribution comparison needs
> enough runs that the spread is small relative to the difference. Budget for
> that before you promise a conversation-level A/B result.

## Key Takeaways

- **Scripted user → pairing is legitimate.** Cheap and sensitive, but it can
  only find what you wrote into the script.
- **Simulated user → pairing is meaningless.** Compare distributions, and pay k
  times as much for the same confidence.
- A tie from a coarse binary scorer is a **resolution** problem, not a verdict.
- Use scripted conversations for the gate you run on every commit, and simulated
  ones for the periodic sweep that finds what the gate cannot see. Not rivals.

## Next Steps

**[`3_failure_analysis`](../3_failure_analysis/)** — when a conversation does
fail, which turn broke it.
