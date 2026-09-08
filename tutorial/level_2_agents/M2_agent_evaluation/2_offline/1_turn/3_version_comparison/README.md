# L2-M2.2.1.3 — Comparing Two Versions of One Agent

**Level:** AI Agents
**Duration:** 60 min

## Overview

You rewrote the system prompt. Should you ship it? That is a different question
from `1_architecture_comparison`, which compares three *different* things — and
the difference is what this lesson is about.

## Prerequisites

- Completed: [`1_architecture_comparison`](../1_architecture_comparison/),
  [`2_offline_gates`](../2_offline_gates/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Independent versus paired

```text
independent   3 architectures, each with its own mean
PAIRED        1 agent, 2 versions, the SAME cases in both
```

Because both versions answer identical cases, you can compare **case by case**
rather than only mean against mean. That buys two things a mean cannot give you.

### 1. Win / loss / tie

A mean that rises by 0.05 is equally consistent with:

- six cases each a little better → **ship it**
- three better, three worse → you changed behaviour, you did not improve it
- one much better, two slightly worse → you traded, and you should know for what

Those call for opposite decisions. The mean cannot tell them apart; the per-case
column can.

### 2. A test that works on six cases

The **sign test** asks: if the two versions were truly equal, how often would
chance alone produce a split this lopsided? Ties are excluded — under the null
each non-tied case is a coin flip.

### Why the scoring is deterministic

A judge adds variance to **both** arms. On the 6–20 cases a real suite has, that
noise swamps the effect you are trying to measure. Keeping the scorer
deterministic is what makes a small difference detectable at all.

### Designing cases that can lose

`v2` adds "always quote the policy code". The suite deliberately contains cases
where that helps (`cites_policy: True`) **and** cases where the extra words could
cost it (`max_words`). A suite containing only the first kind cannot detect a
regression, and would make every prompt change look like an improvement.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/1_turn/3_version_comparison
uv sync
uv run python main.py
```

## Expected Output

```text
  case                                              v1    v2  result
  ---------------------------------------------- ----- -----  ------
  How long do I have to return something?         0.67  1.00  v2 WIN
  What does the warranty cover?                   0.67  1.00  v2 WIN
  Who pays return shipping on a faulty item?      0.67  1.00  v2 WIN
  What is the status of order A1001?              1.00  1.00  tie
  Has order A1003 arrived?                        1.00  1.00  tie
  Is order A1002 on its way?                      1.00  1.00  tie

  means      : v1 0.834  ->  v2 1.000   (delta +0.166)
  paired     : 3 win / 0 loss / 3 tie
  sign test  : p = 0.250 over 3 non-tied cases

  VERDICT: KEEP v1 (not proven)
```

**Read that verdict against that delta.** The mean rose by 0.166, v2 lost
nothing, and the answer is still "not proven" — because three wins out of three
non-tied cases is what a coin flip produces 25% of the time.

> [!warning]
> **With six cases the sign test almost never clears 0.05, and that is the
> correct answer rather than a defect in the test.** A six-case suite cannot
> prove a small improvement. Either collect more cases, or accept that you are
> shipping on judgement rather than on evidence — but know which one you are
> doing.

## Key Takeaways

- Pairing needs identical cases; it buys per-case verdicts and a usable test.
- **Win/loss/tie separates three situations a mean conflates.**
- Keep the scorer deterministic, or the noise you add exceeds the effect you
  are measuring.
- Include cases the new version might **lose**, or your suite cannot detect a
  regression.
- Not the same as **L2-M3.2 (Configuration Optimization)**: that *searches* a
  space of models and tool budgets. This *decides* between two candidates you
  already have.

## Next Steps

**[`4_failure_analysis`](../4_failure_analysis/)** — the same results read two
other ways: which segment is failing, and why.
