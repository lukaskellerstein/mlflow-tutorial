# L1-M4.4 — Datasets and Human-in-the-Loop Evaluation

**Level:** Essentials
**Duration:** 35 min

## Overview

Learn how to create structured evaluation datasets, log them with lineage tracking, run LLM inference, and attach human assessments to traces using MLflow's Assessment API. This lesson combines dataset management with human-in-the-loop evaluation workflows -- from automated pre-screening to targeted human review of borderline cases.

## Prerequisites

- Completed: L1-M4.3 (Scorers and Judges)
- MLflow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### The `mlflow.data` Module

MLflow provides dataset constructors for common formats:

| Constructor | Input Type |
|---|---|
| `mlflow.data.from_pandas()` | Pandas DataFrame |
| `mlflow.data.from_numpy()` | NumPy array |
| `mlflow.data.from_spark()` | Spark DataFrame |

Each constructor returns an `mlflow.data.Dataset` with a name, digest (content hash), schema, and source metadata. Call `mlflow.log_input(dataset, context="evaluation")` inside an active run to link the dataset with lineage tracking.

### MLflow Assessment API

MLflow provides first-class support for attaching assessments to traces:

- **`mlflow.log_feedback()`** -- record a judgment (human, LLM, or code) on a trace
- **`mlflow.log_expectation()`** -- record the expected/ground-truth answer on a trace
- **`mlflow.override_feedback()`** -- let a human correct an automated judgment while preserving the original for audit
- **`AssessmentSource`** -- tracks provenance: HUMAN, LLM_JUDGE, or CODE

### Combined Auto + Human Workflow

1. **Auto-judge** scores all examples quickly
2. **Triage** separates clear passes/failures from borderline cases
3. **Human reviewers** focus only on borderline cases (typically 10-30%)
4. **Override** preserves the original auto-score while recording the human correction

## Step-by-Step

### Step 1: Create and Log a Dataset

Build a pandas DataFrame with questions and expected answers. Wrap it as an MLflow Dataset and log it with context:

```python
dataset = mlflow.data.from_pandas(
    qa_data,
    source="tutorial_qa_pairs",
    name="qa_evaluation_dataset",
    targets="ground_truth_answer",
)
mlflow.log_input(dataset, context="evaluation")
```

### Step 2: Run Traced Inference

Send each question to the LLM inside a traced function. Each call produces a trace with a `trace_id` for attaching assessments later:

```python
@mlflow.trace(name=f"qa_{question[:30]}")
def traced_qa(question, expected):
    answer = ask_llm(client, question)
    return {"question": question, "expected": expected, "answer": answer}
```

### Step 3: Attach Human Assessments

Use `log_expectation()` for ground truth and `log_feedback()` for human judgments:

```python
mlflow.log_expectation(
    trace_id=trace_id, name="expected_answer", value="Paris", source=human_source
)
mlflow.log_feedback(
    trace_id=trace_id,
    name="human_correctness",
    value="correct",
    source=human_source,
    rationale="Exact match.",
)
```

### Step 4: Auto-Judge + Human Triage

Auto-judge scores all examples, then triage by score:
- Score >= 4: auto-approved (no human needed)
- Score <= 2: auto-rejected (no human needed)
- Score 3: borderline -- routed to human review with `override_feedback()`

### Step 5: Query Dataset Lineage

After the run, retrieve logged datasets and their contexts:

```python
run_data = mlflow.get_run(run_id)
for ds_input in run_data.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest)
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M4_evaluation/4_datasets_human_in_loop
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Create, Log, and Run Inference on a Dataset
============================================================
  Created Q&A dataset with 5 entries
  Logged dataset -- name: qa_evaluation_dataset, digest: <hash>
  Q: What is MLflow?
  A: MLflow is an open-source platform...

  Logged inference results as 'inference_results.json'

============================================================
Part 2: Attach Human Assessments to Traces
============================================================
  Q1: label=correct, confidence=0.95
  Q2: label=correct, confidence=0.90
  ...
  Assessment summary: {'correct': 3, 'partial': 2}

============================================================
Part 3: Combined Auto-Judge + Human Triage
============================================================
  Q1: auto_score=5, verdict=AUTO_APPROVED
  ...
  Triage summary (5 items):
    auto_approved             3
    borderline_human_review   2

============================================================
Part 4: Query Dataset Lineage and Labels
============================================================
  Dataset: qa_evaluation_dataset
    Digest:  <hash>
    Context: evaluation
```

In the MLflow UI, check:
- The **Datasets** section on the run for logged datasets with schemas
- The **Traces** tab to see Feedback and Expectation assessments on each trace
- The **Artifacts** tab for `inference_results.json` and `assessments.json`

## Key Takeaways

- `mlflow.data.from_pandas()` creates a versioned, trackable dataset with schema, digest, and source metadata
- `mlflow.log_input()` links a dataset to a run with context for lineage tracking
- `mlflow.log_feedback()` and `mlflow.log_expectation()` attach structured assessments directly to traces
- `mlflow.override_feedback()` preserves the original auto-judgment while recording the human correction
- Auto-judge triage reduces human review to borderline cases only, cutting costs significantly
- Dataset lineage lets you trace any result back to the exact data that produced it

## Next Steps

In L1-M5 (Prompt Engineering), you will explore prompt management, A/B testing, and optimization using the Prompt Registry.
