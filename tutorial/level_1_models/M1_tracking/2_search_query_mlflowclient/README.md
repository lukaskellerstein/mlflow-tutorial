# L1-M1.2 -- Search, Query, and MlflowClient

**Level:** Essentials
**Duration:** 40 min

## Overview

This lesson teaches you to programmatically search, filter, and manage MLflow runs. You will start with the fluent API (`mlflow.search_runs()`, `mlflow.search_experiments()`), then move to the full `MlflowClient` API for CRUD operations -- creating experiments and runs explicitly, querying with filters, renaming, deleting, restoring, and building comparison reports.

## Prerequisites

- Completed: L1-M1.1 (Tracking Fundamentals)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Search Programmatically?

The MLflow UI is great for browsing, but real workflows need programmatic access:
- **Automated comparison** -- find the best-performing configuration across hundreds of runs
- **Reporting** -- export results to pandas for analysis and visualization
- **CI/CD integration** -- query runs in scripts to enforce quality gates

### Search Filter Syntax

MLflow uses a SQL-like filter syntax:

| Filter | Example |
|--------|---------|
| Parameter equality | `params.temperature = '0.7'` |
| Metric comparison | `metrics.total_tokens > 50` |
| Tag matching | `tags.lesson = 'L1-M1.2'` |
| AND combinations | `params.topic = 'rag' AND metrics.total_tokens > 100` |
| LIKE (wildcards) | `params.prompt_topic LIKE 'trans%'` |

### Two Ways to Search

| Method | Returns | Best for |
|--------|---------|----------|
| `mlflow.search_runs()` | pandas DataFrame | Data analysis, aggregation |
| `MlflowClient.search_runs()` | List of Run objects | Programmatic access, automation |

### Why MlflowClient?

The fluent API manages a single "active run" behind the scenes. MlflowClient gives you full programmatic control without global state:

| Aspect | Fluent API | MlflowClient |
|--------|-----------|--------------|
| State | Implicit "active run" | Explicit `run_id` |
| Run lifecycle | `start_run()` / `end_run()` | `create_run()` / `update_run()` |
| Logging | `mlflow.log_param(key, val)` | `client.log_param(run_id, key, val)` |
| Delete/Restore | Not available | `client.delete_run()` / `client.restore_run()` |
| Rename | Not available | `client.update_run(run_id, name=...)` |

## Step-by-Step

### Steps 1-5: Fluent Search API

Create 6 runs with different topics and temperatures, then demonstrate search and filter operations:

```python
# All runs
mlflow.search_runs(experiment_ids=[experiment_id])

# Filter by parameter
mlflow.search_runs(..., filter_string="params.temperature = '0.3'")

# Order by metric
mlflow.search_runs(..., order_by=["metrics.total_tokens DESC"])

# Combined filter
mlflow.search_runs(..., filter_string="params.prompt_topic = 'transformers' AND metrics.total_tokens > 100")
```

### Step 6: List Experiments

```python
mlflow.search_experiments()
```

### Step 7: DataFrame Export

```python
df = mlflow.search_runs(experiment_ids=[experiment_id])
summary = df.groupby("params.prompt_topic")["metrics.total_tokens"].agg(["count", "mean", "max"])
```

### Steps 8-11: MlflowClient CRUD

```python
ml_client = MlflowClient()

# Create experiment and runs explicitly
experiment_id = ml_client.create_experiment("my_experiment", tags={...})
run = ml_client.create_run(experiment_id, run_name="conservative")
ml_client.log_param(run.info.run_id, "temperature", 0.2)
ml_client.update_run(run.info.run_id, status="FINISHED")

# Query
runs = ml_client.search_runs(experiment_ids=[id], order_by=[...])

# Delete and restore
ml_client.delete_run(run_id)
ml_client.restore_run(run_id)

# ViewType for filtering deleted runs
active = ml_client.search_runs([id], run_view_type=ViewType.ACTIVE_ONLY)
all_runs = ml_client.search_runs([id], run_view_type=ViewType.ALL)
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M1_tracking/2_search_query_mlflowclient
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Creating sample runs with different configurations
============================================================
  transformers     temp=0.3  tokens=  45  time=1.2s
  transformers     temp=0.7  tokens=  52  time=1.4s
  ...

============================================================
Step 4: search_runs -- order by total tokens DESC
============================================================
  Runs ranked by total tokens (most first):
  <table of runs sorted by token count>

============================================================
Step 7: DataFrame export -- summary statistics
============================================================
  Token usage summary by topic:
                  runs  avg_tokens  max_tokens
  transformers      3        48.3          52
  ...

============================================================
Step 8: MlflowClient -- create experiment and runs
============================================================
  Created experiment: ...
  conservative: latency=1.23s, tokens=350
  balanced: latency=1.45s, tokens=410
  creative: latency=1.89s, tokens=480

============================================================
Step 10: MlflowClient -- update, delete, restore
============================================================
  Added tags to all runs.
  Renamed run: 'conservative' -> 'conservative_renamed'
  Renamed back to: 'conservative'
  Deleted run abc12345... (stage=deleted)
    Active: 2, All (incl. deleted): 3
    After restore: stage=active

============================================================
Step 11: Comparison report via MlflowClient
============================================================
  Config           Temp    Latency   Tokens   Resp Len
  ---------------- ------ ---------- -------- ----------
  conservative        0.2       1.23      350        420
  balanced            0.7       1.45      410        530
  creative            1.0       1.89      480        620

============================================================
Fluent API vs MlflowClient
============================================================
  Fluent API: simple, manages 'active run' automatically.
  MlflowClient: full CRUD, explicit run_id, no global state.
```

## Key Takeaways

- `mlflow.search_runs()` returns a pandas DataFrame -- great for analysis, aggregation, and finding the best run.
- The filter syntax supports parameter, metric, and tag comparisons with AND logic and ordering.
- `mlflow.search_experiments()` lists all experiments on the tracking server.
- `MlflowClient` provides explicit, stateless access via `run_id` -- no hidden "active run."
- Only MlflowClient can delete/restore runs, rename runs, and create experiments with tags.
- `ViewType.ACTIVE_ONLY` vs `ViewType.ALL` controls whether deleted runs appear in search results.
- Combining MlflowClient queries with pandas DataFrames is a powerful pattern for comparison reports and CI/CD quality gates.

## Next Steps

Continue to **L1-M1.3 Advanced Tracking Patterns** where you will learn nested runs for configuration sweeps, async logging for high-throughput evaluation, and artifact organization patterns.
