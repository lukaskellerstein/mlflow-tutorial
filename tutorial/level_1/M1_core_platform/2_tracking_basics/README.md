# L1-1.2 — Experiment Tracking Basics

**Level:** Essentials
**Duration:** 30 min

## Overview

Learn the fundamentals of MLflow experiment tracking: how to organize work into experiments and runs, log parameters and metrics, attach artifacts like plots, and tag runs with metadata. These are the building blocks you will use in every subsequent lesson.

## Prerequisites

- Completed: L1-1.1 Architecture Overview
- MLFlow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (uses scikit-learn only)

## Concepts

### Experiments

An **experiment** is a named container that groups related runs together. Think of it as a project folder. We create one with `mlflow.set_experiment()` and use a hierarchical name so the MLflow UI stays organized.

### Runs

A **run** is a single execution of your code. Each run records its own parameters, metrics, artifacts, and tags. Use `mlflow.start_run()` as a context manager so the run is automatically closed when the block exits.

### Parameters

**Parameters** are the input configuration for a run — hyperparameters, file paths, thresholds. They are logged once and do not change during the run. Use `mlflow.log_param()` for a single value or `mlflow.log_params()` for a dictionary.

### Metrics

**Metrics** are numeric measurements that describe the outcome — accuracy, loss, latency. Use `mlflow.log_metric()` for a single value or `mlflow.log_metrics()` for a dictionary. Metrics can also be logged at successive **steps** to record training curves.

### Artifacts

**Artifacts** are output files — plots, serialized models, data samples. Use `mlflow.log_artifact()` to upload a file or `mlflow.log_artifacts()` to upload an entire directory.

### Tags

**Tags** are key-value string metadata — model type, dataset name, author. They help you filter and search runs later. Use `mlflow.set_tag()` or `mlflow.set_tags()`.

## Step-by-Step

### Step 1: Connect to MLflow and set the experiment

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M1_core_platform/2_tracking_basics")
```

This tells the MLflow client where the server lives and which experiment to log into.

### Step 2: Start a run and log parameters

```python
with mlflow.start_run(run_name="iris_random_forest") as run:
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "test_size": 0.2,
    })
```

`log_params()` accepts a dictionary so you can log everything in one call.

### Step 3: Train the model and log metrics

```python
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1": f1_score(y_test, y_pred, average="weighted"),
}
mlflow.log_metrics(metrics)
```

### Step 4: Log step-based metrics

```python
for step in range(10):
    loss *= 0.75 + 0.05 * rng.random()
    mlflow.log_metric("training_loss", loss, step=step)
```

Step-based metrics create a time-series you can visualize as a chart in the MLflow UI.

### Step 5: Set tags

```python
mlflow.set_tags({
    "model_type": "RandomForestClassifier",
    "dataset": "iris",
    "task_type": "classification",
})
```

Tags are searchable — you can later filter runs by `tags.model_type = 'RandomForestClassifier'`.

### Step 6: Log an artifact

```python
fig.savefig(plot_path, bbox_inches="tight")
mlflow.log_artifact(plot_path)
```

The confusion matrix PNG is uploaded and viewable directly in the MLflow UI's artifact browser.

## Running the Lesson

```bash
cd tutorial/level_1/M1_core_platform/2_tracking_basics
uv sync
uv run python main.py
```

## Expected Output

Terminal output will look like:

```
============================================================
Step 1: Loading the Iris dataset
============================================================
  Training samples: 120
  Test samples:     30

============================================================
Step 2: Starting MLflow run and logging parameters
============================================================
  Run ID:   <generated-id>
  Run Name: iris_random_forest

  Logged params: {'n_estimators': 100, 'max_depth': 5, ...}

============================================================
Step 3: Training the model
============================================================
  Model trained successfully.

============================================================
Step 4: Logging evaluation metrics
============================================================
    accuracy: 1.0000
   precision: 1.0000
      recall: 1.0000
          f1: 1.0000

============================================================
Step 5: Logging step-based metrics (simulated training loss)
============================================================
  Step  0  loss=0.7xxx
  Step  1  loss=0.5xxx
  ...

============================================================
Step 6: Setting tags
============================================================
  model_type: RandomForestClassifier
  dataset: iris
  task_type: classification

============================================================
Step 7: Logging artifact (confusion matrix plot)
============================================================
  Saved and logged: confusion_matrix.png

============================================================
Done! View results in the MLflow UI:
  http://127.0.0.1:5000/#/experiments
============================================================
```

In the MLflow UI you will see:

- The experiment **L1/M1_core_platform/2_tracking_basics** with one run
- **Parameters** tab: n_estimators, max_depth, random_state, test_size
- **Metrics** tab: accuracy, precision, recall, f1, and a training_loss chart
- **Artifacts** tab: confusion_matrix.png (viewable inline)
- **Tags**: model_type, dataset, task_type

## Key Takeaways

- Use `mlflow.set_experiment()` to organize runs into logical groups.
- `mlflow.start_run()` as a context manager ensures runs are properly closed.
- `log_params()` / `log_metrics()` accept dictionaries for bulk logging.
- `log_metric()` with a `step` argument records training curves.
- `log_artifact()` uploads any file (plots, data, configs) to the run.
- Tags are free-form metadata that make runs searchable and filterable.

## Next Steps

In **L1-1.3 — Search & Query API** you will learn how to programmatically search and filter the runs you just logged, using `mlflow.search_runs()` and the MLflow query syntax.
