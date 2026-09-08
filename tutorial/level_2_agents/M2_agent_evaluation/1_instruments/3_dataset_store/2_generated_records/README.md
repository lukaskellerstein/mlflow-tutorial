# L2-M2.1.3.2 — The Dataset Store: Generated and Multi-Turn Records

**Level:** AI Agents
**Duration:** 45 min

## Overview

The previous lesson stored the richest record shape: a person knew the answer
**and** the route, so every record could be compared against ground truth.
Records that nobody wrote are thinner, and the store holds them all the same way.
That is the point — **who produced a record decides what you can assert about it,
and decides nothing about how it is stored.**

This lesson also shows the one record shape that exists nowhere else in the
tutorial: a **multi-turn** record.

## Prerequisites

- Completed: [`1_hand_written_records`](../1_hand_written_records/)
- MLflow server running at <http://127.0.0.1:5555>
- No gateway or model engine needed — this lesson makes **no LLM calls**

## Concepts

### Three expectation shapes, three things you can assert

| Record came from | Its `expectations` holds | So it is |
|:--|:--|:--|
| Hand-written tests | the answer **and** the tools | compared against ground truth |
| A distilled goal | the tools only | route compared, answer judged |
| `test_agent` | nothing at all | judged only, never compared |

A record with no expectations is not broken. It is an edge probe, and what you
assert about it is "did not hallucinate" — which is a judge's job.

### The multi-turn record

`inputs` is a message list, not a prompt string, precisely so a stored case can
begin partway through a conversation:

```python
{
    "inputs": {
        "messages": [
            {"role": "user", "content": "What is the status of order A1001?"},
            {"role": "assistant", "content": "Order A1001 has shipped, arriving Thursday."},
            {"role": "user", "content": "And if it turns up damaged, who pays to send it back?"},
        ]
    },
    "expectations": {"contains": "free", "expected_tools": ["policy_lookup"]},
}
```

The expectation is about **this** turn. The two before it are context.

### Two kinds of "test case", one store

This is the collision that causes the most confusion, and this lesson proves it
by running:

| Kind | `inputs` holds | Feeds |
|:--|:--|:--|
| **Evaluation record** | `messages`, plus `expectations` | `mlflow.genai.evaluate()` |
| **Simulator scenario** | `goal`, `persona` | `ConversationSimulator` |

MLflow calls both of them `test_cases`. They are not interchangeable.

## Step-by-Step

### Step 1: Start from the hand-written shape

So the merge has something to merge into.

### Step 2: Merge what nobody wrote

```python
dataset.merge_records(DISTILLED_GOALS)
dataset.merge_records(GENERATED)
dataset.merge_records(MULTI_TURN)
```

One store, three calls, four producers. `merge_records` never asks where a
record came from.

### Step 3: Read the shapes back

The lesson prints turn count and shape for every record, so the three
expectation shapes and the multi-turn ones are visible in one table.

### Step 4: Hand both kinds to the simulator

Constructing `ConversationSimulator` **validates** its input, so both checks cost
nothing — no LLM call, no `.simulate()`:

```python
ConversationSimulator(test_cases=evaluation_dataset)  # raises
ConversationSimulator(test_cases=scenario_dataset)  # accepted
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/3_dataset_store/2_generated_records
uv sync
uv run python main.py
```

Runs in seconds. No LLM calls.

## Expected Output

```text
  turns  shape -> what you can do         first message
  -----  -------------------------------- ------------------------
      1  hand-written  -> compared        What is the status of or
      1  distilled     -> route compared  Find out whether order A
      1  generated     -> judged only     What is your policy on p
      3  hand-written  -> compared        What is the status of or
      3  partial       -> judged in part  How long is the warranty

  4/9 records can be compared against a right answer.
  5/9 must be judged instead.
  2/9 start partway through a conversation.
```

And step 4, which is the lesson's real payoff:

```text
  ConversationSimulator(evaluation records  ) -> REJECTED
    EvaluationDataset passed to ConversationSimulator must contain con

  ConversationSimulator(simulator scenarios ) -> ACCEPTED
```

`ConversationSimulator` reads `to_df()["inputs"]` and demands a `goal` key. An
evaluation record has `messages` instead, so it raises. This is the clearest
proof available that a **simulator test case is a scenario, not an assertion**:
it holds a goal, not a right answer, so it can never pass or fail.

## Key Takeaways

- `merge_records` is producer-blind. One store holds every record shape.
- What you can assert follows from the **expectations**, not from who wrote them.
- `inputs` is a message list so a stored case can start at turn 3.
- MLflow calls two different things `test_cases`. An evaluation record feeds
  `evaluate()`; a scenario feeds `ConversationSimulator`. Swapping them raises.
- Constructing a simulator validates its input for free — a cheap way to check
  a dataset before spending LLM calls on it.

## Next Steps

**[`../../../2_offline/`](../../../2_offline/)** — running these datasets as a
ship gate: is this version good enough to release?
