# L2-M2.1.3.1 — The Dataset Store: Hand-Written Records

**Level:** AI Agents
**Duration:** 45 min

## Overview

Every lesson before this one leaves its test cases in a Python list that dies
with the script. This lesson turns them into an `EvaluationDataset`: named,
versioned, stored server-side, and loadable by name.

It owns the store API and the **richest record shape** — the one a person wrote
knowing both the answer and the route. That shape is the only one you can
compare against ground truth, which is why it is the one that yields pass or
fail. The thinner shapes, and the multi-turn record, are
[`2_generated_records`](../2_generated_records/).

It makes no LLM calls. It is about the store, not the agent, so it runs in
seconds.

## Prerequisites

- Completed: L2-M2.1.1.1, L2-M2.1.2.1, L2-M2.1.2.4, and L1-M4.2.3 (Datasets and Human
  Feedback)
- MLflow server running at <http://127.0.0.1:5555>

## Concepts

### Why an agent dataset is not a model dataset

L1-M4.2.3 built datasets for a model. Three things change for an agent:

| | model dataset | agent dataset |
|:--|:--|:--|
| `inputs` | a prompt string | a **message list**, so a case can start mid-conversation |
| `expectations` | what the answer should contain | that, **plus which tools should be called** |
| producers | one | three, and merging them is the point |

Half of agent failure is right answer, wrong route. An expectation that cannot
name a tool cannot catch it.

### The richest record shape

```python
# hand-written -- a person knew the answer AND the route
{
    "inputs": user_turn("What is the status of order A1001?"),
    "expectations": {"contains": "shipped", "expected_tools": ["order_status"]},
}
```

Both halves matter. `contains` asserts the answer; `expected_tools` asserts the
route. Half of agent failure is the right answer reached the wrong way — or
invented without calling anything.

Records with thinner expectations, or none at all, are not broken. They are
judged instead of compared, and the next lesson holds them.

### `merge_records` is an upsert

It keys on the record's inputs, so merging the same records twice updates them
instead of duplicating them. That is what makes it safe to run on every commit
in CI.

### Versioning is tags, not copies

```python
set_dataset_tags(dataset_id=..., tags={"version": "2", "sources": "..."})
```

Tags are **merged** — existing keys survive unless you set them to `None`. There
is no separate version entity for datasets; a tag plus `created_time` is how you
tell two states apart.

### Finding it again

```python
search_datasets(
    experiment_ids=[EXPERIMENT_ID],
    filter_string="tags.owner = 'support-quality'",
    max_results=5,
)
```

> [!warning]
> **A bare `search_datasets()` returns every dataset on the tracking server.**
> MLflow warns that this can be slow enough to crash the session. Always pass
> `experiment_ids`, a `filter_string`, or `max_results`.

## Step-by-Step

### Step 0: Delete any dataset left by an earlier run

The lesson is meant to be re-runnable and teach the same thing each time, so it
starts by removing its own previous dataset.

### Step 1: Create

```python
dataset = create_dataset(
    name="support_agent_regression",
    experiment_id=EXPERIMENT_ID,
    tags={"version": "1", "owner": "support-quality", "lesson": "L2-M2.1.3.1"},
)
```

### Step 2: Merge the hand-written records — 6 records

### Step 3: Merge the same six again — still 6

### Step 4: Read the record shape with `to_df()`

### Step 5: Re-tag it as version 2

### Step 6: Find it by tag, without knowing its id

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/3_dataset_store/1_hand_written_records
uv sync
uv run python main.py
```

Seconds — there are no model calls.

## Expected Output

```text
Step 2: merge the hand-written records
  after hand-written:           6 records

Step 3: merge the SAME records again
  after re-merging 6:           6 records

Step 4: the record shape
  columns: ['inputs', 'outputs', 'expectations', 'tags', 'source_type',
            'source_id', 'source', 'created_time', 'dataset_record_id']
  inputs       : What is the status of order A1001?
  expectations : {'contains': 'shipped', 'expected_tools': ['order_status']}

  6/6 records carry FULL expectations -- the answer and the route.

Step 5: version it
  tags now : {'version': '2', 'owner': 'support-quality', ... 'sources': '...'}
```

Note that `to_df()` returns more columns than you merged. `outputs`, `source`,
`created_time` and `dataset_record_id` are MLflow's, and `dataset_record_id` is
what `delete_records()` takes.

## Key Takeaways

- An agent dataset's `inputs` is a message list and its `expectations` name
  tools, because route correctness is half of agent correctness.
- A record with the answer AND the route is the only kind you can compare
  against ground truth, and comparable is what yields pass or fail.
- `merge_records` upserts on inputs, so re-running in CI does not grow the
  dataset.
- Dataset versioning is tags; `set_dataset_tags` merges rather than replaces.
- Never call `search_datasets()` without a filter.
- This dataset drops straight into `mlflow.genai.evaluate(data=dataset, ...)`.
  Feeding `ConversationSimulator` needs a *different* record shape — see the
  next lesson.

## Next Steps

**[`2_generated_records`](../2_generated_records/)** adds the records nobody
wrote — thinner expectations, the multi-turn shape, and the proof that MLflow's
two meanings of "test case" cannot be swapped.
