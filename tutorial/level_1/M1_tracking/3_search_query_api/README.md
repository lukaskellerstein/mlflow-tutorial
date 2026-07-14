# L1-M1.3 — Search and Query API

**Level:** Essentials
**Duration:** 20 min

## Overview

This lesson teaches you to programmatically search, filter, and compare MLflow runs. You will create several LLM runs with different configurations, then use `search_runs()`, `MlflowClient`, and pandas to query and analyze the results.

## Prerequisites

- Completed: L1-M1.1 (First Run), L1-M1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Search Programmatically?

The MLflow UI is great for browsing, but real workflows need programmatic access:
- **Automated comparison** — find the best-performing configuration across hundreds of runs
- **Reporting** — export results to pandas for analysis and visualization
- **CI/CD integration** — query runs in scripts to enforce quality gates

### Search Filter Syntax

MLflow uses a SQL-like filter syntax:

| Filter | Example |
|--------|---------|
| Parameter equality | `params.temperature = '0.7'` |
| Metric comparison | `metrics.total_tokens > 50` |
| Tag matching | `tags.lesson = 'L1-M1.3'` |
| AND combinations | `params.topic = 'rag' AND metrics.total_tokens > 100` |
| LIKE (wildcards) | `params.prompt_topic LIKE 'trans%'` |

### Two Ways to Search

| Method | Returns | Best for |
|--------|---------|----------|
| `mlflow.search_runs()` | pandas DataFrame | Data analysis, aggregation |
| `MlflowClient.search_runs()` | List of Run objects | Programmatic access, automation |

## Step-by-Step

### Step 1: Create Sample Runs

We create 6 runs with different topics (transformers, RAG, agents), temperatures (0.3, 0.7, 1.0), and system prompts to give us interesting data to query.

### Steps 2-6: Search and Filter

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

### Steps 7-9: Experiments, Client, and DataFrame Export

```python
# List all experiments
mlflow.search_experiments()

# MlflowClient for programmatic access
client = MlflowClient(tracking_uri=TRACKING_URI)
client.search_runs(experiment_ids=[id], order_by=["metrics.response_time_seconds ASC"], max_results=1)

# DataFrame aggregation
df.groupby("params.prompt_topic")["metrics.total_tokens"].agg(["count", "mean", "max"])
```

## Running the Lesson

```bash
cd tutorial/level_1/M1_tracking/3_search_query_api
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
Step 5: search_runs — order by total tokens DESC
============================================================
  Runs ranked by total tokens (most first):
  <table of runs sorted by token count>

============================================================
Step 9: DataFrame export — summary statistics
============================================================
  Token usage summary by topic:
                  runs  avg_tokens  max_tokens
  transformers      3        48.3          52
  rag               2        41.0          45
  agents            1        55.0          55
```

## Key Takeaways

- `mlflow.search_runs()` returns a pandas DataFrame — great for analysis and aggregation.
- The filter syntax supports parameter, metric, and tag comparisons with AND logic.
- `MlflowClient` gives programmatic access to runs, experiments, and artifacts.
- Use `order_by` to rank runs by any metric (e.g., find the fastest or most token-efficient run).
- DataFrame operations (groupby, agg) let you summarize results across topics or configurations.

## Next Steps

Continue to **L1-M1.4 System Metrics Logging** where you will learn to automatically collect CPU, memory, and GPU metrics during LLM inference runs.
