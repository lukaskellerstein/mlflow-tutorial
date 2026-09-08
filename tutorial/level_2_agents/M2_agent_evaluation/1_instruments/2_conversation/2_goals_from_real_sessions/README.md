# L2-M2.1.2.2 — Goals and Personas Distilled from Real Sessions

**Level:** AI Agents
**Duration:** 45 min

## Overview

The previous lesson ran the simulator forward: a goal you wrote became a
conversation. That does not scale, because every scenario costs a person the
time to imagine it — and people imagine the paths they already thought of.
`generate_test_cases()` reads conversations that already happened and infers the
goal and the persona behind each one. Nobody writes them.

## Prerequisites

- Completed: [`1_conversation_simulation`](../1_conversation_simulation/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### There is still only one direction

The simulator never runs backward. It always goes goal → conversation. What this
lesson adds is a second **source** of goal:

```text
   goal  ──── simulate ────▶  conversation        always this direction

   Where the goal comes from:
     you write it                      (previous lesson)
     distilled from old conversations  (this lesson)
```

So "backward" describes where the goal came from, not a second execution mode.
Read it as a goal factory that happens to read conversations.

### The input is a session, and nothing else

`generate_test_cases(sessions)` never sees your agent, your tools, or the goal
that produced those sessions. It reads the conversation and infers. Point it at
real production sessions and your users have written your test suite.

It returns dicts with `goal`, `persona` and `simulation_guidelines` — every key
`ConversationSimulator` accepts, so the output drops straight back into the
engine.

### `persona` is one user, not the participants

Same rule as the previous lesson. The persona it infers is the character of the
**single** user in that conversation. The simulator has two sides: that user and
your agent.

## Step-by-Step

### Step 1: Seed conversations to stand in for real traffic

In production you skip this and search the experiment your agent already logs
to. It is here so the lesson runs on a fresh machine with an empty server.

```python
seeder = ConversationSimulator(test_cases=SEED_SCENARIOS, max_turns=3, user_model=SIM_MODEL)
seed_traces = seeder.simulate(predict_fn)
```

### Step 2: Read them back as sessions

```python
sessions = mlflow.search_sessions(locations=[EXPERIMENT_ID], max_results=2)
```

One conversation is one session and several traces — one per turn.

### Step 3: Distil goals out of them

```python
distilled = generate_test_cases(sessions, model=SIM_MODEL)
```

### Step 4: Feed a distilled goal back into the simulator

```python
replay = ConversationSimulator(test_cases=distilled[:1], max_turns=2, user_model=SIM_MODEL)
replay_traces = replay.simulate(predict_fn)
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/2_conversation/2_goals_from_real_sessions
uv sync
uv run python main.py
```

> [!note]
> Expect several minutes. A second model plays the user for every seeded turn,
> then one more LLM call per session to distil.

## Expected Output

```text
Step 3: generate_test_cases() -- sessions back into goals
======================================================================
  one LLM call per session, inferring goal and persona...

  2 of 2 sessions produced a case

  goal     : Determine whether order A1002 will arrive this week and why it is...
  persona  : An impatient customer who asks short, blunt follow-up questions
```

> [!warning]
> **Count first, then read.** Distillation is one LLM call per session, and a
> session whose answer will not parse is **dropped** — the function returns a
> shorter list rather than raising. Print `len(distilled)` against
> `len(sessions)` before you print the contents. Otherwise "it produced nothing"
> reads exactly like "it produced nothing wrong".

## Key Takeaways

- The simulator only runs forward. This lesson supplies a different **source**
  of goal, not a different direction.
- `generate_test_cases()` needs only sessions. It never sees the agent.
- Its output keys are exactly the simulator's input keys, so the loop closes.
- Nothing here is graded. This lesson and the previous one both produce
  conversations; judging them is the next lesson.
- A silent drop is the failure mode to watch for, not an exception.

## Next Steps

**[`3_judging_conversations`](../3_judging_conversations/)** — turning a
conversation into a verdict, in words and in numbers.
