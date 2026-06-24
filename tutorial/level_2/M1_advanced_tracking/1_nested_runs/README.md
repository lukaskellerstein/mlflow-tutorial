# L2-1.1 — Nested Runs and Run Hierarchies

**Level:** Practitioner
**Duration:** 1 hour

## Overview

Learn how to organize related MLflow runs into parent-child hierarchies using nested runs. This lesson builds a hyperparameter grid search where a parent run groups six child runs (three models times two hyperparameter values), then demonstrates how to query and compare the children programmatically. Nested runs are the foundation for structured experimentation at scale.

## Prerequisites

- Completed: L1-1.2 Tracking Basics, L1-1.3 Search & Query API
- MLFlow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (uses scikit-learn only)

## Concepts

### Why Nested Runs?

In Level 1 you created individual runs. That works for quick experiments, but real-world workflows generate dozens or hundreds of runs for a single question ("which model and hyperparameter combination is best?"). Without structure, the MLflow UI becomes a flat, unsorted list.

**Nested runs** solve this by letting you create parent-child relationships:

- A **parent run** represents the high-level task (e.g., "hyperparameter sweep" or "cross-validation").
- **Child runs** represent individual attempts within that task (e.g., one per configuration or fold).

The parent groups everything together in the UI, and you can expand or collapse the hierarchy.

### How Nested Runs Work

The key is the `nested=True` parameter in `mlflow.start_run()`:

```python
with mlflow.start_run(run_name="sweep") as parent:
    for config in configs:
        with mlflow.start_run(run_name=f"config_{config}", nested=True):
            # This run is a child of "sweep"
            mlflow.log_params(config)
            ...
```

When `nested=True`, MLflow automatically sets the tag `mlflow.parentRunId` on the child run, linking it to the currently active parent.

### Common Use Cases

| Pattern | Parent run | Child runs |
|---------|-----------|------------|
| Hyperparameter sweep | The sweep itself | One per hyperparameter configuration |
| Cross-validation | The CV procedure | One per fold |
| Ensemble training | The ensemble | One per base model |
| Multi-stage pipeline | The pipeline | One per stage |
| A/B testing | The experiment | One per variant |

### Querying Child Runs

You can retrieve all children of a parent run using `mlflow.search_runs()` with a filter on the `mlflow.parentRunId` tag:

```python
child_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    order_by=["metrics.accuracy DESC"],
)
```

This returns a pandas DataFrame that you can sort, filter, and analyze programmatically.

## Step-by-Step

### Step 1: Define the search grid

We test three models with two `max_depth` values each, giving six configurations total:

```python
MODEL_CONFIGS = [
    {"name": "RandomForest", "class": RandomForestClassifier, ...},
    {"name": "GradientBoosting", "class": GradientBoostingClassifier, ...},
    {"name": "LogisticRegression", "class": LogisticRegression, ...},
]
MAX_DEPTH_VALUES = [3, 7]
```

LogisticRegression does not use `max_depth`, so the lesson handles that gracefully by skipping the parameter for that model.

### Step 2: Create the parent run

```python
with mlflow.start_run(run_name="hyperparameter_sweep") as parent_run:
    mlflow.set_tags({
        "sweep_type": "grid_search",
        "dataset": "wine",
        "num_configs": "6",
    })
```

The parent run captures metadata about the sweep as a whole.

### Step 3: Create nested child runs

Inside the parent's context, each child run is created with `nested=True`:

```python
for model_cfg in MODEL_CONFIGS:
    for max_depth in MAX_DEPTH_VALUES:
        with mlflow.start_run(run_name=f"{model_name}_depth_{max_depth}", nested=True):
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")
```

Each child logs its own parameters, metrics, model artifact, and tags.

### Step 4: Summarize on the parent

After all children complete, the parent run logs the best result:

```python
best = max(results, key=lambda r: r["accuracy"])
mlflow.log_params({"best_model": best["model"], "best_max_depth": best["max_depth"]})
mlflow.log_metrics({"best_accuracy": best["accuracy"], "best_f1": best["f1"]})
```

### Step 5: Query children with search_runs()

After the sweep, use `search_runs()` to retrieve all children, sorted by accuracy:

```python
child_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
    order_by=["metrics.accuracy DESC"],
)
```

The result is a pandas DataFrame you can print as a summary table.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_tracking/1_nested_runs
uv sync
uv run python main.py
```

## Expected Output

Terminal output will look like:

```
============================================================
Step 1: Loading the Wine dataset
============================================================
  Training samples: 142
  Test samples:     36
  Features:         13

============================================================
Step 2: Running hyperparameter grid search (nested runs)
============================================================
  RandomForest_depth_3                     accuracy=0.9722  f1=0.9720
  RandomForest_depth_7                     accuracy=1.0000  f1=1.0000
  GradientBoosting_depth_3                 accuracy=0.9722  f1=0.9722
  GradientBoosting_depth_7                 accuracy=0.9444  f1=0.9438
  LogisticRegression_depth_3               accuracy=0.9722  f1=0.9727
  LogisticRegression_depth_7               accuracy=0.9722  f1=0.9727

============================================================
Step 3: Logging parent-run summary
============================================================
  Best config:   RandomForest_depth_7
  Best accuracy: 1.0000
  Best F1:       1.0000
  Parent run ID: <generated-id>

============================================================
Step 4: Querying child runs with search_runs()
============================================================

 run_id  model_family  max_depth  accuracy    f1
 ...     RandomForest  7          1.0000      1.0000
 ...     RandomForest  3          0.9722      0.9720
 ...     (remaining rows sorted by accuracy)

============================================================
Done! View the nested run hierarchy in the MLflow UI:
  http://127.0.0.1:5000/#/experiments
  Expand the 'hyperparameter_sweep' parent run to see children.
============================================================
```

In the MLflow UI you will see:

- The experiment **L2/M1_advanced_tracking/1_nested_runs** with a parent run named "hyperparameter_sweep"
- Expanding the parent reveals six child runs, each with its own parameters, metrics, and model artifact
- The parent run has summary metrics (`best_accuracy`, `best_f1`) and tags pointing to the best child
- You can compare child runs side-by-side using the MLflow comparison view

## Key Takeaways

- Use `nested=True` in `mlflow.start_run()` to create parent-child run hierarchies.
- MLflow automatically sets `mlflow.parentRunId` on child runs, linking them to the parent.
- The parent run is the right place for sweep-level metadata and summary metrics.
- Use `search_runs()` with a `tags.mlflow.parentRunId` filter to programmatically retrieve children.
- Tags on child runs (`model_family`, `sweep_param`) make filtering easy.
- Nested runs keep the MLflow UI organized when you have many related runs.

## Next Steps

In **L2-1.2 -- Async and Batch Logging** you will learn how to log large volumes of data efficiently using MLflow's async logging API, which is critical when your training loop or sweep generates thousands of metrics.
