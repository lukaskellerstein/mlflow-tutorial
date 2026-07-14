# L1-M4.4 — Datasets

**Level:** Essentials
**Duration:** 25 min

## Overview

Learn how to create structured evaluation datasets in MLflow, inspect their schemas, log them with lineage tracking, run LLM inference on them, and attach human labels. These are the building blocks for systematic GenAI quality measurement and reproducible evaluations.

## Prerequisites

- Completed: L1-M4.3 (Scorers and Judges)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Track Datasets?

When you evaluate an LLM, the quality of your assessment depends on the dataset you use. Without dataset tracking, you lose the connection between "which questions were asked" and "which results were produced." MLflow's dataset logging solves this by:

- **Recording what data was used** — schema, size, and a content digest (hash)
- **Distinguishing contexts** — was this dataset used for evaluation, testing, or fine-tuning?
- **Providing lineage** — given a set of results, trace back to the exact questions and expected answers that produced them

### The `mlflow.data` Module

MLflow provides dataset constructors for common formats:

| Constructor | Input Type |
|---|---|
| `mlflow.data.from_pandas()` | Pandas DataFrame |
| `mlflow.data.from_numpy()` | NumPy array |
| `mlflow.data.from_spark()` | Spark DataFrame |
| `mlflow.data.from_huggingface()` | Hugging Face Dataset |

Each constructor returns an `mlflow.data.Dataset` object that captures:

- **Name** — a human-readable label
- **Digest** — a hash of the data content (for change detection)
- **Schema** — column names and types
- **Source** — where the data came from
- **Targets** — which column contains the expected answers

### Logging and Lineage

Call `mlflow.log_input(dataset, context="evaluation")` inside an active run. The `context` parameter is a free-form string that records how the dataset was used. After the run completes, the dataset info is visible in the MLflow UI and queryable via `mlflow.get_run()`.

### Labeling and Assessments

After a model generates answers, humans review and label them. MLflow's `mlflow.log_table()` stores structured label data as a JSON artifact attached to a run. This creates a reusable ground truth that can be loaded back with `mlflow.load_table()`.

## Step-by-Step

### Step 1: Create a Q&A Evaluation Dataset

We build a pandas DataFrame with questions, expected answers, context, and difficulty labels. This represents a typical evaluation dataset for a Q&A system.

```python
qa_data = pd.DataFrame({
    "question": ["What is MLflow?", ...],
    "ground_truth_answer": ["An open-source platform...", ...],
    "context": ["overview", ...],
    "difficulty": ["easy", "medium", "hard", ...],
})
```

### Step 2: Log Dataset with Schema Inspection

Wrap the DataFrame as an MLflow Dataset, inspect its schema, and log it with a context tag. You can log multiple datasets per run (e.g., a full set and a subset).

```python
dataset = mlflow.data.from_pandas(
    qa_data, source="tutorial_qa_pairs",
    name="qa_evaluation_dataset", targets="ground_truth_answer",
)
mlflow.log_input(dataset, context="evaluation")
print(dataset.schema)  # column names and types
print(dataset.digest)  # content hash for change detection
```

### Step 3: Run LLM Inference on the Dataset

Send each question to the LLM and log the results as a table artifact:

```python
for _, row in qa_data.iterrows():
    answer = ask_llm(client, row["question"])
mlflow.log_table(results_df, artifact_file="inference_results.json")
```

### Step 4: Add Human Labels

Create a labels table with model answers and human judgments, then log it:

```python
mlflow.log_table(labels_data, artifact_file="labels.json")
```

### Step 5: Query Dataset Lineage

After the run, retrieve the logged datasets and their contexts programmatically:

```python
run_data = mlflow.get_run(run_id)
for ds_input in run_data.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest)
```

### Step 6: Load Labels Back for Analysis

Retrieve the labels table and compute quality metrics:

```python
loaded_labels = mlflow.load_table("labels.json", run_ids=[run_id])
accuracy = (loaded_labels["human_label"] == "correct").sum() / len(loaded_labels)
```

## Running the Lesson

```bash
cd tutorial/level_1/M4_evaluations/4_datasets
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Creating an Evaluation Dataset
============================================================
  Created Q&A dataset with 6 entries
  Columns: ['question', 'ground_truth_answer', 'context', 'difficulty']
  Difficulty distribution:
    easy: 2
    medium: 2
    hard: 2

  Logged dataset — name: qa_evaluation_dataset, digest: <hash>
  Schema: ...

  Logged subset — name: qa_hard_subset, digest: <hash>

============================================================
Part 2: Running LLM Inference on the Dataset
============================================================
  Q: What is MLflow?
  A: MLflow is an open-source platform...

  Logged inference results as 'inference_results.json'

============================================================
Part 3: Adding Human Labels / Assessments
============================================================
  Logged 6 labels as 'labels.json'
    correct: 4
    partial: 2

============================================================
Part 4: Querying Dataset Lineage
============================================================
  Dataset: qa_evaluation_dataset
    Digest:  <hash>
    Source:  tutorial_qa_pairs
    Context: evaluation

  Dataset: qa_hard_subset
    Digest:  <hash>
    Source:  tutorial_qa_pairs
    Context: evaluation_subset

============================================================
Part 5: Loading Labels Back for Analysis
============================================================
  Label summary:
    correct     : 4/6 (67%)
    partial     : 2/6 (33%)
    incorrect   : 0/6 (0%)

  Overall accuracy: 66.7%
```

In the MLflow UI, click on the run and check:
- The **Datasets** section shows both logged datasets with schemas
- The **Artifacts** tab contains `inference_results.json` and `labels.json`

## Key Takeaways

- `mlflow.data.from_pandas()` creates a versioned, trackable dataset with schema, digest, and source metadata
- `mlflow.log_input()` links a dataset to a run with context (e.g., "evaluation")
- The dataset digest acts as a fingerprint — if the data changes, the digest changes
- You can log multiple datasets per run with different contexts (full set, subsets)
- `mlflow.log_table()` stores structured results and labels as JSON artifacts
- `mlflow.load_table()` retrieves them for downstream analysis
- Dataset lineage lets you trace any result back to the exact data that produced it

## Next Steps

In L1-M5 (Prompt Engineering), we'll explore prompt management, A/B testing, and optimization. In Level 2 (M3.4), we'll build full human-in-the-loop labeling pipelines with iterative refinement.
