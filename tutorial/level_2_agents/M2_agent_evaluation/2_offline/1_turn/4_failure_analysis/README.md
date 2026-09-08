# L2-M2.2.1.4 — Reading an Evaluation: Slices and Failure Taxonomy

**Level:** AI Agents
**Duration:** 60 min

## Overview

Every lesson so far produces a **number**. This one produces a **decision about
what to fix next**, from results you already have. No new metric and no new
scorer — two ways of reading the same result set.

## Prerequisites

- Completed: [`2_offline_gates`](../2_offline_gates/),
  [`3_version_comparison`](../3_version_comparison/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Part 1 — the aggregate lies

A suite reporting 0.75 can be 1.00 on three slices and 0.00 on a fourth. The
mean says nothing about which, and **users experience their slice, not your
mean**. Ship on the aggregate and you ship something perfect for most people and
broken for one group — and you hear about it from them, not from your dashboard.

The whole mechanism is one field on each case:

```python
{"slice": "unknown_order", "q": "Status of order Z9999?", ...}
```

> [!important]
> **A slice you did not label cannot be analysed.** The field costs nothing at
> authoring time and is impossible to add retroactively without re-reading every
> case. Label by anything you might later suspect: length, language, tool
> needed, customer tier.

### Part 2 — "12 failed" is not actionable

"9 of the 12 failed for one reason" is. Ranking causes by frequency turns an
evaluation into a work queue.

The classifier is ordered **most-specific-first** on purpose. A case can show
several symptoms, and the ranking is only useful if each failure lands in
exactly one bucket — so you are counting root causes rather than symptoms.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/1_turn/4_failure_analysis
uv sync
uv run python main.py
```

## Expected Output

The number you would normally report:

```text
  overall pass rate: 0.67  (6/9)
```

Then the same results, sliced:

```text
  slice              n  pass rate   vs overall
  ---------------- --- ----------   ----------
  known_order        3       1.00        +0.33
  policy             3       1.00        +0.33
  unknown_order      3       0.00        -0.67
```

Then the same failures, by cause:

```text
  cause                          n   share  cumulative
  ---------------------------- --- ------- -----------
  ignored_tool_result            3   100%        100%
```

**Read those three blocks together.** 0.67 looks mediocre but survivable. The
slice table says two segments are perfect and one is *completely broken*. The
taxonomy says every failure is the same bug: the agent called the right tool,
the tool replied "no order found", and the agent answered as though it had not.

That is a hallucination on unknown IDs — a serious, specific, fixable defect,
and the aggregate hid it entirely.

> [!warning]
> **Read the `n` column before acting.** A slice of 3 scoring 0.00 is a *lead*,
> not a finding — three cases can miss by chance. A slice of 30 at 0.40 is a
> finding. Slice size is what separates the two.

> [!warning]
> **An aggregate can move the opposite way to every slice.** If two versions are
> measured on different slice *mixes*, one can win overall while losing on every
> single slice — Simpson's paradox.
> [`3_version_comparison`](../3_version_comparison/) avoids it by running both
> versions on identical cases. Comparing runs with different case mixes does not.

## Key Takeaways

- Slicing and taxonomy add **no new metric** — they re-read results you have.
- The aggregate is an average weighted by slice size, and hides everything else.
- Label slices **before** you need them; it cannot be done afterwards.
- Rank causes by frequency and fix the one that explains the most failures, not
  the first one you happened to read.
- Both techniques work with any scorer, deterministic or judge-based.

## Next Steps

**[`5_swe_bench`](../5_swe_bench/)** — the same offline question against a bar
your team does not own.
