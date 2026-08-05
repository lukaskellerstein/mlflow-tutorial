# L2-M3.3 — Optimizing Against Benchmarks Without Destroying Them

**Level:** AI Agents
**Duration:** 90 min

## Overview

Wanting the best possible SWE-Bench or GAIA score is a legitimate goal, and people pursue it. The trap is that **a benchmark you optimize against stops being a measurement and becomes training data** — the number then describes your tuning loop, not your agent. This lesson builds a benchmark with an explicit held-out split, tunes against the dev half only, and uses the dev-minus-held-out gap as an overfitting signal.

## Prerequisites

- Completed: L2-M2.2.3 (SWE-Bench) and L2-M2.2.4 (GAIA)
- Completed: L2-M3.1 (Prompt and Instruction Optimization)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)

## Concepts

### The discipline

```text
optimize on DEV  ->  report on HELD-OUT  ->  the gap is your overfitting signal
```

Tuning is allowed on dev. Reporting happens on held-out. The moment you tune on the split you report, the number stops being a capability claim.

### Real benchmarks differ on whether they let you

| Benchmark | Held-out half? | Consequence |
|:--|:--|:--|
| **GAIA** | Yes — `split="validation"` answers are public, `split="test"` answers are withheld and scored by leaderboard submission | You can tune honestly |
| **SWE-Bench Verified** | No — `split="test"` ships the gold patches *and* the `FAIL_TO_PASS` / `PASS_TO_PASS` lists | Tuning contaminates your only measure; carve your own dev subset |

This asymmetry is visible in the lessons themselves: L2-M2.2.4 loads `split="validation"`, L2-M2.2.3 loads `split="test"`. It is also a large part of why published leaderboard numbers routinely fail to reproduce in deployment.

### Why this lesson's split looks rigged

It is, deliberately. The dev half is dominated by short factual answers; the held-out half keeps some of those but adds questions needing an explanation. A prompt tuned to be terse therefore wins on dev and loses on held-out. That is exactly the failure mode real benchmark tuning produces — compressed into ten tasks so you can watch it happen instead of reading about it.

### Tag the split on every run

A run whose split is unrecorded is a run nobody can trust six months later. Every run here carries `split: dev` or `split: held_out` as a tag, so the distinction survives review.

## Step-by-Step

### Step 1: Split the benchmark, and mean it

```python
DEV = [...]  # tuning allowed
HELD_OUT = [...]  # reporting only, never tuned on
```

### Step 2: Optimize on dev only

```python
with mlflow.start_run(run_name=f"dev/{name}", nested=True):
    mlflow.set_tags({"split": "dev", "prompt_variant": name})
    mlflow.log_metrics(score_split(prompt, DEV, "dev"))
```

The held-out split is not touched — not scored, not looked at.

### Step 3: Score everything on held-out and log the gap

```python
gap = dev_scores[name] - metrics["held_out_accuracy"]
mlflow.log_metrics({**metrics, "dev_minus_heldout": gap})
```

Scoring *every* candidate on held-out (not just the dev winner) is what makes the gap legible.

### Step 4: Read the gap

If the dev winner and the held-out winner disagree, the dev winner was selected partly for a quirk of the dev split. Track the gap across optimization iterations: a rising gap means you are fitting the split, not improving the agent.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M3_agent_optimization/3_benchmark_optimization
uv sync
uv run python main.py
```

## Expected Output

```text
  Part 4: What the gap tells you

  variant         dev   held-out     gap   short  explain
  -------------------------------------------------------
  baseline       80%      100%    -20%    100%     100%
  terse         100%       40%    +60%    100%       0%
  balanced        0%       80%    -80%     50%     100%

  DEV winner      : terse
  HELD-OUT winner : baseline

  These disagree -- which is the entire lesson. The dev winner was
  selected partly for a quirk of the dev split, not for being better.

  Largest dev-minus-held-out gap: 'terse' at +60%
```

Read the `terse` row across: a perfect 100% on dev, 40% on held-out, and **0% on the explain questions**. It did not get better at answering; it got better at the shape of the dev split. That +60% gap is the overfitting signal, and it is only visible because a held-out half existed.

The `balanced` row shows the opposite failure — it explains everything, including the short questions, so it scores 0% on dev under a grader that penalises over-long answers. Both are miscalibration; only one of them looks like success on dev.

Exact numbers vary; free-tier models are non-deterministic even at `temperature=0.0`. What reproduces is the direction — `terse` carries the largest gap and collapses on `explain`.

## Key Takeaways

- Optimizing against a benchmark is legitimate **only** with a split you never report on.
- GAIA gives you one; SWE-Bench Verified does not — carve your own.
- Score every candidate on held-out, not just the dev winner, or the gap stays invisible.
- A rising dev-minus-held-out gap means stop tuning or enlarge dev.
- Tag the split on every run, or the distinction dies in review.

## Next Steps

This completes Level 2. Continue to **Level 3 — Advanced**, where the assessments from L2-M2.3.1 become production dashboards and alerting, and where L3-M4.2 builds a full cross-framework benchmark as a capstone.
