# L1-M7.1 -- Dataset Logging and Lineage

**Level:** Essentials
**Duration:** ~20 minutes

## Overview

MLflow can track not just models and metrics, but also the datasets used in
evaluation. This lesson shows how to log an LLM evaluation dataset alongside
an inference run, creating a clear lineage from data to results. Understanding
data lineage is critical for reproducibility and auditing.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Track Datasets?

When you evaluate an LLM, the quality of your assessment depends on the
dataset you use. Without dataset tracking, you lose the connection between
"which questions were asked" and "which results were produced." MLflow's
dataset logging solves this by:

- **Recording what data was used** -- schema, size, and a content digest (hash)
- **Distinguishing contexts** -- was this dataset used for evaluation, testing,
  or fine-tuning?
- **Providing lineage** -- given a set of results, trace back to the exact
  questions and expected answers that produced them

### The `mlflow.data` Module

MLflow provides dataset constructors for common formats:

| Constructor | Input Type |
|---|---|
| `mlflow.data.from_pandas()` | Pandas DataFrame |
| `mlflow.data.from_numpy()` | NumPy array |
| `mlflow.data.from_spark()` | Spark DataFrame |
| `mlflow.data.from_huggingface()` | Hugging Face Dataset |

Each constructor returns an `mlflow.data.Dataset` object that captures:

- **Name** -- a human-readable label
- **Digest** -- a hash of the data content (for change detection)
- **Schema** -- column names and types
- **Source** -- where the data came from
- **Targets** -- which column contains the expected answers

### Logging and Lineage

Call `mlflow.log_input(dataset, context="evaluation")` inside an active run.
The `context` parameter is a free-form string that records how the dataset
was used. After the run completes, the dataset info is visible in the MLflow
UI and queryable via `mlflow.get_run()`.

## Step-by-Step

### Step 1: Create a Q&A evaluation dataset

We build a pandas DataFrame with questions, expected answers, and categories.
This represents a typical LLM evaluation dataset.

```python
qa_data = pd.DataFrame({
    "question": ["What is the capital of France?", ...],
    "expected_answer": ["Paris", ...],
    "category": ["geography", ...],
})
```

### Step 2: Wrap as an MLflow Dataset

`mlflow.data.from_pandas()` takes the DataFrame, an optional source string,
the target column name, and a human-readable dataset name.

```python
eval_dataset = mlflow.data.from_pandas(
    qa_data, source="manual_qa_pairs",
    targets="expected_answer", name="llm_eval_qa",
)
```

### Step 3: Run LLM inference and log with lineage

Inside an MLflow run, we link the dataset, run the LLM on each question,
and log accuracy metrics.

```python
with mlflow.start_run(run_name="llm_eval_with_dataset"):
    mlflow.log_input(eval_dataset, context="evaluation")
    for row in qa_data.iterrows():
        answer = ask_llm(client, row["question"])
    mlflow.log_metric("accuracy", accuracy)
```

### Step 4: Query dataset lineage

After the run completes, retrieve the logged datasets programmatically.

```python
run_data = mlflow.get_run(run_id)
for ds_input in run_data.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest)
```

## Running the Lesson

```bash
cd tutorial/level_1/M7_data_datasets/1_dataset_logging
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Creating a Q&A evaluation dataset
============================================================
  Dataset size: 5 Q&A pairs
  Columns: ['question', 'expected_answer', 'category']

============================================================
Step 2: Creating MLflow datasets
============================================================
  Name:   llm_eval_qa
  Digest: <hash>
  Schema: ...

============================================================
Step 3: Running LLM inference and logging results
============================================================
  Q: What is the capital of France?
  A: Paris is the capital of France...
  Expected: Paris  Match: yes
  ...
  Accuracy: 4/5 = 80.0%

============================================================
Step 4: Querying dataset lineage from the completed run
============================================================
  Dataset: llm_eval_qa
    Digest:  <hash>
    Source:  manual_qa_pairs
    Context: evaluation
```

In the MLflow UI, open the run and look for the **Datasets** section.

## Key Takeaways

- `mlflow.data.from_pandas()` wraps a DataFrame into a trackable Dataset
  with schema, digest, and source metadata.
- `mlflow.log_input(dataset, context="evaluation")` links a dataset to the
  current run with a context label.
- Dataset lineage lets you trace any evaluation result back to the exact
  questions and expected answers that produced it.
- You can log multiple datasets per run with different contexts.
- The dataset digest acts as a fingerprint -- if the data changes, the
  digest changes.

## Next Steps

Continue to **Level 2, M1** (Advanced Tracking) to explore nested runs,
async logging, and rich artifact types. In Level 2, we will use datasets
extensively in LLM evaluation workflows.
