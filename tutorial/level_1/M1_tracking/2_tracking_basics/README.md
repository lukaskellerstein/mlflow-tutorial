# L1-M1.2 — Tracking LLM Experiments

**Level:** Essentials
**Duration:** 30 min

## Overview

This lesson is the **catalog of MLflow tracking methods**. You will use every `log_*` function MLflow offers to record an LLM experiment — parameters, metrics, tags, text, dicts, tables, files, figures, images, and step-based metrics — all in a single run.

## Prerequisites

- Completed: L1-M1.1 (Your First MLflow Run)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### The MLflow Tracking Toolbox

MLflow provides many ways to attach data to a run. Each method is suited to a different type of data:

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

## Step-by-Step

### Step 1: Parameters and Metrics

Log LLM configuration as parameters and response statistics as metrics — both individually and in batch:

```python
mlflow.log_param("model", MODEL)                    # single param
mlflow.log_params({"temperature": 0.7, ...})        # batch params
mlflow.log_metric("response_time", 2.1)             # single metric
mlflow.log_metrics({"total_tokens": 85, ...})       # batch metrics
```

### Step 2: Tags

Attach metadata labels for organizing and filtering runs:

```python
mlflow.set_tag("model_family", "gemma")              # single tag
mlflow.set_tags({"level": "1", "module": "tracking"}) # batch tags
```

### Step 3: Text and Dict Artifacts

Save the raw LLM response as text, and the full call details as structured JSON:

```python
mlflow.log_text(result["content"], "response.txt")
mlflow.log_dict({"prompt": ..., "response": ...}, "call_details.json")
```

### Step 4: Table

Log tabular data (e.g., comparing temperatures) as a structured table:

```python
mlflow.log_table(data=pd.DataFrame(rows), artifact_file="comparison.json")
```

### Step 5: File Artifacts

Save files and directories — useful for reports, configs, or collections of outputs:

```python
mlflow.log_artifact("summary.md")                      # single file
mlflow.log_artifacts("responses/", artifact_path="responses")  # directory
```

### Step 6: Figure and Image

Save visualizations — matplotlib charts and PIL-generated images:

```python
mlflow.log_figure(fig, "token_chart.png")      # matplotlib figure
mlflow.log_image(img, artifact_file="card.png") # PIL image
```

### Step 7: Step-Based Metrics

Track metrics across sequential steps — creates line charts in the UI:

```python
for step, prompt in enumerate(prompts):
    result = call_llm(client, prompt)
    mlflow.log_metric("cumulative_tokens", total, step=step)
```

## Running the Lesson

```bash
cd tutorial/level_1/M1_tracking/2_tracking_basics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Parameters and Metrics
============================================================
  log_param() — logged 'model' and 'prompt' individually
  log_params() — logged 'temperature' and 'max_tokens' as a batch
  log_metric() — logged response_time=2.1s
  log_metrics() — logged token counts (total=85)

============================================================
Step 2: Tags
============================================================
  set_tag() — tagged 'model_family'='gemma'
  set_tags() — tagged level, module, lesson as a batch

============================================================
Step 3: Text and Dict Artifacts
============================================================
  log_text() — saved LLM response as 'response.txt'
  log_dict() — saved structured call details as 'call_details.json'

============================================================
Step 4: Table (temperature comparison)
============================================================
  temp=0.3  tokens=  85  time=2.1s
  temp=0.7  tokens=  92  time=2.3s
  temp=1.0  tokens= 110  time=2.8s
  log_table() — saved temperature comparison table

============================================================
Step 5: File Artifacts
============================================================
  log_artifact() — saved 'summary.md' (single file)
  log_artifacts() — saved 'responses/' directory (3 files)

============================================================
Step 6: Figure and Image
============================================================
  log_figure() — saved matplotlib bar chart as 'token_chart.png'
  log_image() — saved PIL-generated summary card as 'summary_card.png'

============================================================
Step 7: Step-based Metrics
============================================================
  Step 0: 'What is a transformer model?...'  tokens=45  cumulative=45
  Step 1: 'What is attention in machine ...'  tokens=52  cumulative=97
  ...
```

In the MLflow UI, the single run "all_logging_methods" contains everything: parameters, metrics with step charts, tags, and a rich set of artifacts (text, JSON, table, markdown, images, charts).

## Key Takeaways

- MLflow provides **11+ logging methods** — each suited to a different data type.
- `log_param` / `log_metric` / `set_tag` are the core three — config, numbers, and labels.
- `log_text`, `log_dict`, `log_table` handle structured data without managing temp files.
- `log_artifact` / `log_artifacts` upload any file or directory.
- `log_figure` / `log_image` save visualizations that render inline in the UI.
- Step-based `log_metric(..., step=N)` creates time-series line charts.

## Next Steps

Continue to **L1-M1.3 Search and Query API** where you will learn to programmatically search, filter, and compare your tracked runs using MLflow's query API.
