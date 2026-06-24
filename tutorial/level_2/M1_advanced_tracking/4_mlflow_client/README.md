# L2-1.4 — MlflowClient: Programmatic Access

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

The fluent API (`mlflow.log_param()`, `mlflow.start_run()`, etc.) is convenient for interactive work, but it relies on global state (the "active run") and only supports the most common operations. `MlflowClient` gives you full programmatic control over every MLflow entity — experiments, runs, metrics, tags — without any global state. This lesson teaches you when and how to use it.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics), L1-M1.3 (Search & Query API)
- MLflow server running at http://127.0.0.1:5000
- Familiarity with scikit-learn basics

## Concepts

### Why MlflowClient?

The fluent API manages a single "active run" behind the scenes. This works well when you are inside a training loop, but breaks down when you need to:

- **Manage runs after the fact** — rename, tag, delete, or restore runs from a script
- **Work across experiments** — query and compare runs from multiple experiments
- **Build automation** — CI/CD pipelines, dashboards, admin tools
- **Avoid global state** — in multi-threaded or multi-process scenarios, the fluent API's global active run can cause conflicts

`MlflowClient` solves all of these by requiring you to pass `run_id` explicitly for every operation. There is no hidden state.

### Key Differences

| Aspect | Fluent API | MlflowClient |
|--------|-----------|--------------|
| State | Implicit "active run" | Explicit `run_id` |
| Run lifecycle | `start_run()` / `end_run()` | `create_run()` / `update_run()` |
| Logging | `mlflow.log_param(key, val)` | `client.log_param(run_id, key, val)` |
| Querying | `mlflow.search_runs()` | `client.search_runs()` |
| Delete/Restore | Not available | `client.delete_run()` / `client.restore_run()` |
| Rename | Not directly available | `client.update_run(run_id, name=...)` |

### CRUD Operations

`MlflowClient` supports the full set of CRUD operations:

- **Create**: `create_experiment()`, `create_run()`
- **Read**: `get_experiment()`, `get_experiment_by_name()`, `get_run()`, `search_experiments()`, `search_runs()`
- **Update**: `log_param()`, `log_metric()`, `set_tag()`, `update_run()`
- **Delete**: `delete_run()`, `restore_run()`

## Step-by-Step

### Step 1: Create Experiment and Runs

Instead of `mlflow.set_experiment()` and `mlflow.start_run()`, we use `client.create_experiment()` and `client.create_run()`:

```python
client = MlflowClient()

# create_experiment raises if the name exists, so check first
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME,
        tags={"project": "mlflow-tutorial", "level": "2"},
    )

# Create a run — this does NOT set a global "active run"
run = client.create_run(experiment_id, run_name="my_run")
run_id = run.info.run_id

# Log params and metrics by run_id
client.log_param(run_id, "learning_rate", 0.01)
client.log_metric(run_id, "accuracy", 0.95)

# Mark the run as finished (client runs start as RUNNING)
client.update_run(run_id, status="FINISHED")
```

We train three different scikit-learn models (Logistic Regression, Random Forest, Gradient Boosting) and log each one this way.

### Step 2: Query Operations

Search and retrieve experiments and runs:

```python
# Find experiments by name
experiments = client.search_experiments(
    filter_string=f"name = '{EXPERIMENT_NAME}'"
)

# Get experiment by ID or name
exp = client.get_experiment(experiment_id)
exp = client.get_experiment_by_name(EXPERIMENT_NAME)

# Search runs with ordering and filtering
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
)

# Filter to high-accuracy runs
good_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
)

# Get full details for a single run
run = client.get_run(run_id)
print(run.data.params)   # dict of all params
print(run.data.metrics)  # dict of latest metrics
print(run.info.status)   # RUNNING, FINISHED, FAILED, etc.
```

### Step 3: Update and Manage Runs

Operations only available through MlflowClient:

```python
# Add tags to existing runs
client.set_tag(run_id, "dataset", "iris")

# Rename a run
client.update_run(run_id, name="new_name")

# Delete a run (soft delete — moves to "deleted" lifecycle stage)
client.delete_run(run_id)

# Deleted runs are excluded from active searches
active = client.search_runs([experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
all_runs = client.search_runs([experiment_id], run_view_type=ViewType.ALL)

# Restore a deleted run
client.restore_run(run_id)
```

### Step 4: Build a Comparison Report

Query all runs and build a formatted table:

```python
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
)

for r in runs:
    name = r.info.run_name
    acc = r.data.metrics.get("accuracy", 0.0)
    f1 = r.data.metrics.get("f1_score", 0.0)
    print(f"  {name:<25s} {acc:.4f}  {f1:.4f}")
```

This pattern is the foundation for building dashboards, automated reports, and CI/CD quality gates.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_tracking/4_mlflow_client
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Create experiment and runs via MlflowClient
============================================================
  Created experiment: L2/M1_advanced_tracking/4_mlflow_client (id=...)

  Training and logging 3 models...
  logistic_regression: accuracy=0.9778, f1=0.9777, time=0.0123s
  random_forest: accuracy=1.0000, f1=1.0000, time=0.0456s
  gradient_boosting: accuracy=0.9778, f1=0.9777, time=0.0789s

============================================================
Part 2: Query operations
============================================================
  search_experiments found 1 matching experiment(s)
  ...
  search_runs found 3 run(s) (ordered by accuracy DESC)
  ...

============================================================
Part 3: Update and manage operations
============================================================
  Added tags 'tutorial_lesson' and 'dataset' to all runs.
  Renamed run: 'logistic_regression' -> 'lr_renamed'
  ...
  lifecycle_stage after delete: deleted
  lifecycle_stage after restore: active

============================================================
Part 4: Model comparison report
============================================================
  Model                        Accuracy     F1     Time (s)
  ...

  Best model: random_forest (accuracy=1.0000)
```

In the MLflow UI at http://127.0.0.1:5000, you will see the experiment with all three runs, their parameters, metrics, and tags.

## Key Takeaways

- **MlflowClient** provides explicit, stateless access to all MLflow entities via `run_id`.
- Use `create_run()` instead of `start_run()` when you need to manage the run lifecycle yourself.
- `search_runs()` and `search_experiments()` support rich filtering and ordering — identical to the fluent API but with more flexibility.
- **Delete/restore** and **rename** operations are only available through `MlflowClient`, not the fluent API.
- Use the fluent API for interactive work and notebooks; use `MlflowClient` for automation, batch operations, and admin scripts.

## Next Steps

Continue to **L2-M2.1 (Signatures Deep Dive)** to learn how to define and validate model input/output schemas with `ModelSignature`.
