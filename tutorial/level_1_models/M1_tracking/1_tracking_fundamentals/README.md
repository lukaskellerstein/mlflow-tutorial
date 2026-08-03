# L1-M1.1 -- Tracking Fundamentals

**Level:** Essentials
**Duration:** 45 min

## Overview

This lesson introduces MLflow's five core pillars, walks you through your first tracked LLM run, demonstrates every logging method MLflow offers (parameters, metrics, tags, text, dicts, tables, files, figures, images, step-based metrics), and shows how to automatically collect system-level hardware metrics during a run.

## Prerequisites

- MLflow server running at <http://127.0.0.1:5555> (start with `podman compose up -d` from `infra/`)
- LMStudio running with `google/gemma-4-e4b` loaded (`lms load google/gemma-4-e4b --gpu max -y`)
- LMStudio server started (`lms server start`)
- Python 3.10+
- `uv` package manager installed

## Concepts

### The 5 Pillars of MLflow

MLflow is an open-source platform for managing the complete machine learning and AI lifecycle:

1. **Tracking** -- Record and query parameters, metrics, code versions, and artifacts for every experiment run.
2. **Models** -- Package ML and AI models from any framework in a standard format.
3. **Model Registry** -- A centralized model store with versioning and lifecycle management.
4. **Evaluation** -- Evaluate model quality with built-in and custom metrics.
5. **Deployment** -- Serve models as REST APIs or run batch predictions.

### Key Concepts

| Concept | Description |
|------------|-------------|
| **Experiment** | A named collection of runs -- typically one per project or task. |
| **Run** | A single execution that stores parameters, metrics, artifacts, and tags. |
| **Parameters** | Input configuration values (model name, temperature, max tokens). |
| **Metrics** | Numeric results to track (response time, token count, latency). |
| **Artifacts** | Output files -- models, responses, data snapshots. |
| **Tags** | Free-form key/value metadata for organizing and filtering runs. |

### Architecture at a Glance

```text
 Your Code (main.py)
       |
       v
 MLflow Client (Python SDK)
       |
       v
 Tracking Server (http://127.0.0.1:5555)
       |
       v
 Backend Store (PostgreSQL)  +  Artifact Store (local / S3)
```

### The MLflow Tracking Toolbox

| Method | Data Type | Where It Shows in UI |
|--------|-----------|---------------------|
| `log_param()` / `log_params()` | Config strings/numbers | Parameters tab |
| `log_metric()` / `log_metrics()` | Numeric measurements | Metrics tab (charts) |
| `log_metric(..., step=N)` | Time-series numerics | Metrics tab (line charts) |
| `set_tag()` / `set_tags()` | Metadata labels | Tags section |
| `log_text()` | Plain text content | Artifacts tab |
| `log_dict()` | JSON/YAML dicts | Artifacts tab |
| `log_table()` | Tabular data (DataFrames) | Artifacts tab (table view) |
| `log_artifact()` | Any single file | Artifacts tab |
| `log_artifacts()` | A directory of files | Artifacts tab (folder) |
| `log_figure()` | Matplotlib/Plotly figures | Artifacts tab (rendered) |
| `log_image()` | PIL/numpy images | Artifacts tab (rendered) |

### System Metrics

When system metrics logging is enabled, MLflow periodically samples hardware metrics:

| Metric | Description |
|--------|-------------|
| `system/cpu_utilization_percentage` | CPU usage across all cores |
| `system/system_memory_usage_megabytes` | Total RAM in use |
| `system/disk_usage_percentage` | Disk utilization |
| `system/network_receive_megabytes` | Network bytes received |

## Step-by-Step

### Step 1: Parameters and Metrics (first LLM call)

Log LLM configuration as parameters and response statistics as metrics -- both individually and in batch:

```python
mlflow.log_param("model", MODEL)                    # single param
mlflow.log_params({"temperature": 0.7, ...})        # batch params
mlflow.log_metric("response_time", 2.1)             # single metric
mlflow.log_metrics({"total_tokens": 85, ...})       # batch metrics
```

### Step 2: Tags

Attach metadata labels for organizing and filtering runs:

```python
mlflow.set_tag("model_family", "gemma")
mlflow.set_tags({"level": "1", "module": "tracking"})
```

### Step 3: Text and Dict Artifacts

Save the raw LLM response as text, and the full call details as structured JSON:

```python
mlflow.log_text(result["content"], "response.txt")
mlflow.log_dict({"prompt": ..., "response": ...}, "call_details.json")
```

### Step 4: Table (temperature comparison)

Log tabular data (e.g., comparing temperatures) as a structured table:

```python
mlflow.log_table(data=pd.DataFrame(rows), artifact_file="comparison.json")
```

### Step 5: File Artifacts

Save files and directories:

```python
mlflow.log_artifact("summary.md")
mlflow.log_artifacts("responses/", artifact_path="responses")
```

### Step 6: Figure and Image

Save visualizations -- matplotlib charts and PIL-generated images:

```python
mlflow.log_figure(fig, "token_chart.png")
mlflow.log_image(img, artifact_file="card.png")
```

### Step 7: Step-Based Metrics

Track metrics across sequential steps -- creates line charts in the UI:

```python
for step, prompt in enumerate(prompts):
    result = call_llm(client, prompt)
    mlflow.log_metric("cumulative_tokens", total, step=step)
```

### Step 8: System Metrics Logging

Enable automatic hardware metric collection and inspect the results:

```python
mlflow.enable_system_metrics_logging()
mlflow.set_system_metrics_sampling_interval(5)
mlflow.set_system_metrics_samples_before_logging(1)
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M1_tracking/1_tracking_fundamentals
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
MLflow -- The 5 Pillars
============================================================
  1. Tracking
    Record parameters, metrics, and artifacts for every experiment run.
  ...

============================================================
Step 1: Parameters and Metrics (first LLM call)
============================================================
  log_param()  -- logged 'model' and 'prompt' individually
  log_params() -- logged 'temperature' and 'max_tokens' as a batch
  log_metric() -- logged response_time=1.234s
  log_metrics() -- logged token counts (total=85)

============================================================
Step 2: Tags
============================================================
  set_tag()  -- tagged 'model_family'='gemma'
  set_tags() -- tagged level, module, lesson as a batch

...

============================================================
Step 7: Step-based Metrics
============================================================
  Step 0: 'What is a transformer model?...'  tokens=45  cumulative=45
  Step 1: 'What is attention in machine ...'  tokens=52  cumulative=97
  ...

============================================================
Step 8: System Metrics Logging
============================================================
  System metrics logging: ENABLED (sampling every 5s)
  Run ID: <generated-run-id>
  Making LLM calls to generate measurable load...
  Waiting 12 seconds for system metrics collection...
  Found 7 system metric(s):
    system/cpu_utilization_percentage                    = 45.20
    system/system_memory_usage_megabytes                 = 24576.00
    ...
```

## Key Takeaways

- MLflow is built on five pillars: Tracking, Models, Model Registry, Evaluation, and Deployment.
- The **experiment** groups related runs; the **run** stores params, metrics, artifacts, and tags.
- MLflow provides **11+ logging methods** -- each suited to a different data type (params, metrics, tags, text, dicts, tables, files, figures, images).
- Step-based `log_metric(..., step=N)` creates time-series line charts in the UI.
- `mlflow.enable_system_metrics_logging()` captures CPU, memory, disk, and network metrics automatically during a run.

## Next Steps

Continue to **L1-M1.2 Search, Query, and MlflowClient** where you will learn to programmatically search, filter, and manage runs using both the fluent API and the low-level MlflowClient.
