# L1-6.3 — Datasets and Labeling

**Level:** Essentials
**Duration:** 20 min

## Overview

Learn how to create structured evaluation datasets in MLflow, attach human labels and assessments, and load labeled data back for evaluation. These are the building blocks for systematic GenAI quality measurement.

## Prerequisites

- Completed: L1-6.1 (Prompt Registry), L1-6.2 (Scorers and Judges)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Why Evaluation Datasets Matter

Evaluating LLMs and AI agents requires structured test data — input/expected-output pairs that cover different difficulty levels and edge cases. MLflow provides tools to create, version, and track these datasets so evaluations are reproducible.

### Dataset Logging

`mlflow.data.from_pandas()` wraps a pandas DataFrame as an MLflow Dataset object. When logged with `mlflow.log_input()`, MLflow records:
- The dataset schema and digest (hash) for reproducibility
- A link between the dataset and the run that used it
- The context (e.g., "training", "evaluation") describing how it was used

### Labeling and Assessments

After a model generates answers, humans review and label them. MLflow's `mlflow.log_table()` stores structured label data as a JSON artifact attached to a run. This creates a reusable ground truth that can be loaded back with `mlflow.load_table()`.

## Step-by-Step

### Step 1: Create a Q&A Evaluation Dataset

We build a pandas DataFrame with questions, expected answers, optional context, and difficulty labels. This represents a typical evaluation dataset for a Q&A system.

```python
qa_data = pd.DataFrame({
    "question": ["What is MLflow?", ...],
    "ground_truth_answer": ["MLflow is an open-source platform...", ...],
    "context": ["MLflow docs: overview page", ...],
    "difficulty": ["easy", "medium", "hard", ...],
})
```

### Step 2: Log the Dataset to MLflow

Wrap the DataFrame as an MLflow Dataset and log it with a context tag:

```python
dataset = mlflow.data.from_pandas(
    qa_data,
    source="tutorial_qa_pairs",
    name="qa_evaluation_dataset",
    targets="ground_truth_answer",
)
mlflow.log_input(dataset, context="evaluation")
```

### Step 3: Add Human Labels

Create a labels table with model answers and human judgments, then log it as a table artifact:

```python
mlflow.log_table(labels_data, artifact_file="labels.json")
```

### Step 4: Load Labels Back for Analysis

Retrieve the dataset info from the run and load the labels table:

```python
loaded_labels = mlflow.load_table(
    artifact_file="labels.json",
    run_ids=[run_id],
)
```

## Running the Lesson

```bash
cd tutorial/level_1/M6_genai_features/3_datasets_labeling
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

  Logged dataset to run <run_id>
  Dataset name  : qa_evaluation_dataset
  Dataset digest: <hash>

============================================================
Part 2: Adding Human Labels / Assessments
============================================================
  Logged 6 labels as artifact 'labels.json'
  Label distribution:
    correct: 4
    partial: 2

============================================================
Part 3: Loading Labeled Data Back
============================================================
  Dataset name   : qa_evaluation_dataset
  Dataset digest : <hash>
  Context        : evaluation

  Summary of loaded labels:
    correct     : 4/6 (67%)
    partial     : 2/6 (33%)
    incorrect   : 0/6 (0%)

  Overall accuracy: 66.7%

============================================================
Done!
============================================================
  Open the MLflow UI at http://127.0.0.1:5000
  Navigate to experiment: L1/M6_genai_features/3_datasets_labeling
  Check the Datasets tab and the labels.json artifact.
```

In the MLflow UI, click on the run and check:
- The **Datasets** section shows the logged evaluation dataset with its schema
- The **Artifacts** tab contains `labels.json` with the human assessments

## Key Takeaways

- `mlflow.data.from_pandas()` creates a versioned, trackable dataset from a DataFrame
- `mlflow.log_input()` links a dataset to a run with context (e.g., "evaluation")
- `mlflow.log_table()` stores structured label data as a JSON artifact
- `mlflow.load_table()` retrieves label data for analysis and downstream evaluation
- Combining datasets with labels creates reproducible ground truth for GenAI evaluation

## Next Steps

In L1-7.1 (Dataset Logging), we'll explore MLflow's broader data tracking capabilities beyond GenAI. In Level 2 (M3.4), we'll build full human-in-the-loop labeling pipelines with iterative refinement.
