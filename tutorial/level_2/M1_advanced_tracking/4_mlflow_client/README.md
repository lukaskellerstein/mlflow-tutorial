# L2-1.4 -- MlflowClient: Programmatic Access

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

The fluent API (`mlflow.log_param()`, `mlflow.start_run()`, etc.) is convenient for interactive work, but it relies on global state (the "active run") and only supports the most common operations. `MlflowClient` gives you full programmatic control over every MLflow entity -- experiments, runs, metrics, tags -- without any global state. This lesson uses it to manage and compare LLM experiments across different configurations.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics), L1-M1.3 (Search & Query API)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why MlflowClient?

The fluent API manages a single "active run" behind the scenes. This works well when you are inside a training loop, but breaks down when you need to:

- **Manage runs after the fact** -- rename, tag, delete, or restore runs from a script
- **Work across experiments** -- query and compare runs from multiple experiments
- **Build automation** -- CI/CD pipelines, dashboards, admin tools
- **Avoid global state** -- in multi-threaded or multi-process scenarios, the fluent API's global active run can cause conflicts

In an LLM context, `MlflowClient` is especially useful for building comparison reports across different model configurations (temperature, system prompt, model variant) and for managing large numbers of evaluation runs programmatically.

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

### Step 1: Create Experiment and Runs (LLM Configs)

Instead of `mlflow.set_experiment()` and `mlflow.start_run()`, we use `client.create_experiment()` and `client.create_run()`. Each run corresponds to a different LLM configuration (temperature and system prompt):

```python
ml_client = MlflowClient()

# create_experiment raises if the name exists, so check first
experiment = ml_client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment and experiment.lifecycle_stage == "active":
    experiment_id = experiment.experiment_id
else:
    experiment_id = ml_client.create_experiment(
        EXPERIMENT_NAME,
        tags={"project": "mlflow-tutorial", "level": "2"},
    )

# Create a run -- no global "active run" is set
run = ml_client.create_run(experiment_id, run_name="conservative")
run_id = run.info.run_id

# Log LLM params and metrics by run_id
ml_client.log_param(run_id, "model", "google/gemma-4-e4b")
ml_client.log_param(run_id, "temperature", 0.2)
ml_client.log_metric(run_id, "latency_seconds", 1.23)
ml_client.log_metric(run_id, "total_tokens", 350)

# Mark the run as finished (client runs start as RUNNING)
ml_client.update_run(run_id, status="FINISHED")
```

We call the LLM three times with different configurations (conservative, balanced, creative) and log each one this way.

### Step 2: Query Operations

Search and retrieve experiments and runs:

```python
# Find experiments by name
experiments = ml_client.search_experiments(
    filter_string=f"name = '{EXPERIMENT_NAME}'"
)

# Search runs ordered by latency
runs = ml_client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.latency_seconds ASC"],
)

# Filter to runs with short responses
short_runs = ml_client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.response_length < 500",
)

# Get full details for a single run
run = ml_client.get_run(run_id)
print(run.data.params)   # dict of all params
print(run.data.metrics)  # dict of latest metrics
```

### Step 3: Update and Manage Runs

Operations only available through `MlflowClient`:

```python
# Add tags to existing runs
ml_client.set_tag(run_id, "llm_provider", "lm-studio")

# Rename a run
ml_client.update_run(run_id, name="conservative_renamed")

# Delete a run (soft delete -- moves to "deleted" lifecycle stage)
ml_client.delete_run(run_id)

# Deleted runs are excluded from active searches
active = ml_client.search_runs([experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
all_runs = ml_client.search_runs([experiment_id], run_view_type=ViewType.ALL)

# Restore a deleted run
ml_client.restore_run(run_id)
```

### Step 4: Build a Comparison Report

Query all runs and build a formatted comparison table using pandas:

```python
runs = ml_client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.model = 'google/gemma-4-e4b'",
    order_by=["metrics.latency_seconds ASC"],
)

rows = []
for r in runs:
    rows.append({
        "Config": r.info.run_name,
        "Temperature": r.data.params.get("temperature", "N/A"),
        "Latency (s)": r.data.metrics.get("latency_seconds", 0.0),
        "Total Tokens": int(r.data.metrics.get("total_tokens", 0)),
        "Response Len": int(r.data.metrics.get("response_length", 0)),
    })

df = pd.DataFrame(rows)
```

This pattern is the foundation for building dashboards, automated reports, and CI/CD quality gates over LLM experiments.

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

  Calling LLM with 3 configurations...
  conservative: latency=1.23s, tokens=350, response_length=420
  balanced: latency=1.45s, tokens=410, response_length=530
  creative: latency=1.89s, tokens=480, response_length=620

============================================================
Part 2: Query operations
============================================================
  search_experiments found 1 matching experiment(s)
    id=..., name=L2/M1_advanced_tracking/4_mlflow_client

  search_runs found 3 run(s) (ordered by latency ASC):
    conservative         latency=1.23s  (id=abc12345...)
    balanced             latency=1.45s  (id=def67890...)
    creative             latency=1.89s  (id=ghi13579...)

  Runs with response_length < 500: 1
    conservative: 420 chars

  get_run(abc12345...) details:
    name:    conservative
    status:  FINISHED
    params:  4 logged
    metrics: {latency_seconds: 1.23, total_tokens: 350, ...}

============================================================
Part 3: Update and manage operations
============================================================
  Added tags 'tutorial_lesson' and 'llm_provider' to all runs.
  Renamed run: 'conservative' -> 'conservative_renamed'
  Renamed back to: 'conservative'

  Deleting run ghi13579...
    lifecycle_stage after delete: deleted
    Active runs: 2, All runs (incl. deleted): 3
    lifecycle_stage after restore: active

============================================================
Part 4: LLM configuration comparison report
============================================================
  Config           Temp    Latency   Tokens   Resp Len
  ---------------- ------ ---------- -------- ----------
  conservative        0.2       1.23      350        420
  balanced            0.7       1.45      410        530
  creative            1.0       1.89      480        620

  Fastest config: conservative (latency=1.23s)

============================================================
Fluent API vs MlflowClient
============================================================

  Fluent API (mlflow.log_param, mlflow.start_run, etc.):
    - Simple, concise -- great for interactive work and single-run scripts
    - Manages "active run" state automatically
    - Best for: notebooks, single experiments, quick prototyping

  MlflowClient:
    - Full CRUD control -- create, read, update, delete any entity
    - No global state -- you pass run_id explicitly
    - Can manage runs across experiments, rename/delete/restore runs
    - Best for: automation scripts, dashboards, CI/CD pipelines,
      batch operations, admin tools, multi-experiment workflows

============================================================
Done! View results in the MLflow UI:
  http://127.0.0.1:5000/#/experiments
============================================================
```

Note: actual latency, token counts, and response lengths will vary depending on your LMStudio setup and the model's behavior.

In the MLflow UI at http://127.0.0.1:5000, you will see the experiment with all three runs, their LLM parameters (temperature, system prompt), and metrics (latency, tokens, response length).

## Key Takeaways

- **MlflowClient** provides explicit, stateless access to all MLflow entities via `run_id` -- no hidden "active run."
- Use `create_run()` instead of `start_run()` when you need to manage the run lifecycle yourself.
- `search_runs()` and `search_experiments()` support rich filtering and ordering -- use them to build comparison reports across LLM configurations.
- **Delete/restore** and **rename** operations are only available through `MlflowClient`, not the fluent API.
- Combining `MlflowClient` queries with pandas DataFrames is a powerful pattern for building automated LLM experiment dashboards and CI/CD quality gates.

## Next Steps

Continue to **L2-M2.1 (Signatures Deep Dive)** to learn how to define and validate model input/output schemas with `ModelSignature`.
