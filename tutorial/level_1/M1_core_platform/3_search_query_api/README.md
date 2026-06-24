# L1-M1.3 — Search and Query API

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

MLflow stores every run, metric, parameter, and tag in a queryable backend. The Search and Query API lets you filter, sort, and retrieve runs programmatically -- essential for comparing experiments, finding the best model, and building automation pipelines on top of MLflow.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- Infrastructure started with `podman compose up -d` (from `infra/`)

## Concepts

### Why search matters

Once you have dozens or hundreds of runs, scrolling through the UI is impractical. MLflow's search API gives you SQL-like filtering over runs and experiments. You can:

- Find all runs where accuracy exceeded a threshold.
- Filter by parameter values (e.g., model type, learning rate).
- Sort results to surface the best-performing run.
- Export to pandas for downstream analysis.

### Search filter syntax

MLflow uses a simple filter DSL:

| Pattern | Example |
|---------|---------|
| Metric comparison | `metrics.accuracy > 0.9` |
| Parameter equality | `params.model_type = 'random_forest'` |
| Tag matching | `tags.lesson = 'L1-M1.3'` |
| Combined (AND) | `metrics.accuracy > 0.8 AND params.model_type = 'random_forest'` |
| String LIKE | `params.model_type LIKE 'random%'` |

Supported operators: `=`, `!=`, `>`, `>=`, `<`, `<=`, `LIKE`, `ILIKE`.

### Two interfaces

1. **`mlflow.search_runs()`** -- returns a pandas DataFrame. Great for quick analysis.
2. **`MlflowClient.search_runs()`** -- returns a list of `Run` objects. Better for programmatic access (e.g., getting run IDs, artifacts).

## Step-by-Step

### Step 1: Create sample runs

We train six sklearn models on the Iris dataset and log parameters, metrics, and tags for each. This gives us data to query.

```python
models = [
    ("random_forest", RandomForestClassifier(n_estimators=100)),
    ("random_forest", RandomForestClassifier(n_estimators=50)),
    ("gradient_boosting", GradientBoostingClassifier(n_estimators=100)),
    ("logistic_regression", LogisticRegression(max_iter=200)),
    ("svm", SVC(kernel="rbf")),
    ("decision_tree", DecisionTreeClassifier(max_depth=3)),
]

for model_type, model in models:
    with mlflow.start_run(run_name=f"{model_type}_{model.__class__.__name__}"):
        model.fit(X_train, y_train)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
```

### Step 2: Search with no filter

Retrieve every run in the experiment:

```python
all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
```

### Step 3: Filter by metric

Find runs where accuracy exceeds 0.9:

```python
high_acc = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
)
```

### Step 4: Filter by parameter

Find all random forest runs:

```python
rf_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.model_type = 'random_forest'",
)
```

### Step 5: Order results

Sort runs by accuracy, best first:

```python
ordered = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
)
```

### Step 6: Combine filters

Multiple conditions with AND:

```python
combined = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.8 AND params.model_type = 'random_forest'",
)
```

### Step 7: Search experiments

List all experiments on the server:

```python
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"[{exp.experiment_id}] {exp.name}")
```

### Step 8: MlflowClient

Use `MlflowClient` for richer programmatic access:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
experiment = client.get_experiment(experiment_id)
best_runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
```

### Step 9: DataFrame export

Since `mlflow.search_runs()` returns a DataFrame, you can use pandas groupby, aggregation, and plotting:

```python
df = mlflow.search_runs(experiment_ids=[experiment_id])
summary = df.groupby("params.model_type")["metrics.accuracy"].agg(["count", "mean", "max"])
```

## Running the Lesson

```bash
cd tutorial/level_1/M1_core_platform/3_search_query_api
uv sync
uv run python main.py
```

## Expected Output

You should see nine labeled steps printed to the terminal:

1. Six model runs logged (random_forest, gradient_boosting, logistic_regression, svm, decision_tree).
2. All runs listed with run IDs and accuracy values.
3. Filtered runs where accuracy > 0.9.
4. Random forest runs only.
5. Runs sorted by accuracy (best first).
6. Combined filter results.
7. All experiments listed from the server.
8. Best run details from MlflowClient.
9. A summary table grouped by model type with count, mean, and max accuracy.

In the MLflow UI at http://127.0.0.1:5000, navigate to the experiment `L1/M1_core_platform/3_search_query_api` to see all runs with their parameters, metrics, and tags.

## Key Takeaways

- `mlflow.search_runs()` returns a pandas DataFrame -- ideal for analysis and comparison.
- Filter syntax supports metric comparisons, parameter matching, tags, and logical AND.
- `order_by` lets you sort results by any metric or parameter.
- `MlflowClient` provides object-oriented access to runs, experiments, and artifacts.
- `mlflow.search_experiments()` lists all experiments on the tracking server.

## Next Steps

Next up is **L1-M1.4 System Metrics**, where you will learn how MLflow can automatically track CPU, memory, and GPU usage during training runs. In Level 2, we will explore the `MlflowClient` in more depth for advanced programmatic workflows.
