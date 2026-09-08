# L2-M2.2.2.4 — A Multi-Turn Agent Benchmark

**Level:** AI Agents
**Duration:** 60 min

## Overview

`1_turn/5_swe_bench` and `1_turn/6_gaia` are single-turn: one task in, one answer
out, compared to a frozen expected answer. That is a design decision, not
laziness — and this lesson shows exactly what it buys and what it costs.

Here we build a small **multi-turn** benchmark and fix the three things that
multi-turn breaks, using the pattern τ-bench established.

## Prerequisites

- Completed: [`1_conversation_gates`](../1_conversation_gates/),
  [`../../1_turn/6_gaia`](../../1_turn/6_gaia/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Why benchmarks are usually single-turn

A benchmark's whole value is that its number **means the same thing to
everyone**. Multi-turn breaks that in three places at once:

| Problem | Fix |
|:--|:--|
| The conversation depends on the user simulator, which is a model | **Pin the user model in the benchmark spec**, not in the caller's config |
| Judging conversation text needs a judge, which is a model too | **Score the final state** of a database the agent had to mutate |
| One multi-turn run is luck | **Report pass^k**, not just pass@1 |

### 1. The pinned user model

```python
# PART OF THE BENCHMARK SPEC, NOT A CONFIG KNOB.
BENCHMARK_USER_MODEL = "gemma-judge"
```

Every other lesson in this repo names an alias and lets the gateway decide what
it means. A benchmark cannot afford that: change the user simulator and every
conversation changes, so the score stops being comparable. **Publishing this
benchmark means publishing this line.**

### 2. State, not text

The agent gets tools that **change the world** — `cancel_order`,
`update_address`. Scoring reads the resulting dict:

```python
"check": lambda db: db["A1002"]["status"] == "cancelled",
```

No judge, no LLM, no parsing. Re-score a transcript from a year ago and you get
the same answer.

### 3. pass^k

```text
pass@1  every trial counted on its own
pass^k  a task counts only if ALL k trials passed
```

This is the one people skip and the one that matters most. An agent that solves
a task 3 times in 4 is not a 75% agent you can ship — **it is an agent that
fails one customer in four**, and pass^k is what makes that visible.

### A task that passes by not acting

```python
{
    "name": "refuse_impossible_cancel",
    "check": lambda db: db["A1001"]["status"] == "shipped",  # unchanged
}
```

Order A1001 has already shipped and cannot be cancelled. The agent passes by
**leaving the world alone** and explaining why. A benchmark that only rewards
mutation teaches agents to act when they should refuse.

## Step-by-Step

### Step 1: One database per trial

```python
db = copy.deepcopy(INITIAL_DB)
agent = create_agent(llm, tools=build_tools(db), ...)
```

Tools close over **that trial's** database, so trials cannot contaminate each
other. This also avoids a race — the simulator may run conversations
concurrently.

### Step 2: Simulate with the pinned user

```python
simulator = ConversationSimulator(test_cases=[task["scenario"]], user_model=USER_MODEL)
simulator.simulate(predict_fn)
```

### Step 3: Read the world, not the transcript

```python
passed = task["check"](db)
```

### Step 4: Report both numbers

pass@1 and pass^k side by side, so the gap is visible.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/2_conversation/4_multi_turn_benchmark
uv sync
uv run python main.py
```

> [!note]
> Expect 10–20 minutes. Two tasks times k trials, each a multi-turn simulation
> with a second model playing the user.

## Expected Output

```text
  task: cancel_pending_order
    goal     : Cancel order A1002 because you no longer want it
    expected : A1002.status == 'cancelled'
    trial 1/2: PASS  (4 turns)
    trial 2/2: PASS  (4 turns)

======================================================================
  Results
======================================================================
  task                             trials   pass^k
  ---------------------------- ----------  -------
  cancel_pending_order                 PP      YES
  refuse_impossible_cancel             PP      YES

  pass@1  : 1.00   (every trial counted on its own)
  pass^2  : 1.00   (a task counts only if ALL 2 trials passed)
```

When the two numbers diverge, that gap is the lesson. Same agent, same tasks —
**pass@1 is the number that flatters you, pass^k is the one your users feel.**

## Key Takeaways

- Multi-turn benchmarking is possible; it just costs three extra commitments.
- **Pin the user model in the spec.** It is part of the benchmark, not config.
- **Score state, not text.** A dict comparison has no judge in it, so it is
  reproducible forever.
- **Report pass^k.** A single multi-turn success can be luck.
- Include at least one task that passes by **refusing**, or you are training for
  the wrong behaviour.
- This still does **not** buy comparability with another team. That needs them
  to run the same user model, the same simulator version, and the same k. A
  benchmark is a frozen spec, not a script.

## Next Steps

**[`../../../3_online/2_conversation/`](../../../3_online/2_conversation/)** —
the same scope, but against traffic you did not choose.
