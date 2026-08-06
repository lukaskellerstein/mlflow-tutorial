# L1-M1.3 -- Advanced Tracking Patterns

**Level:** Essentials
**Duration:** 45 min

## Overview

This lesson covers three advanced tracking patterns: nested runs for organizing configuration sweeps into parent-child hierarchies, async logging for non-blocking metric recording during batch LLM evaluation, and artifact organization patterns for maintaining a clean folder structure inside runs.

## Prerequisites

- Completed: L1-M1.1 (Tracking Fundamentals), L1-M1.2 (Search, Query, and MlflowClient)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway up (`cd infra && podman compose up -d`), with LMStudio
  serving `google/gemma-4-26b-a4b` behind the `gemma-chat` alias

## Concepts

### Nested Runs

In earlier lessons you created individual, flat runs. Real-world LLM workflows generate dozens of runs for a single question ("which temperature and prompt style combination works best?"). Without structure, the MLflow UI becomes an unsorted list.

**Nested runs** solve this with parent-child relationships:
- A **parent run** represents the high-level task (e.g., "LLM config sweep")
- **Child runs** represent individual attempts within that task

The key is the `nested=True` parameter:

```python
with mlflow.start_run(run_name="LLM Config Sweep") as parent:
    for temp in temperatures:
        with mlflow.start_run(run_name=f"temp_{temp}", nested=True):
            # This run is a child of "LLM Config Sweep"
            mlflow.log_params({"temperature": temp})
```

MLflow automatically sets `mlflow.parentRunId` on child runs. Query children with:

```python
mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'")
```

### Async Logging

By default, every `mlflow.log_metric()` call blocks until the tracking server acknowledges the write. When evaluating an LLM across many prompts, these blocking calls add up. Async logging offloads the I/O to a background thread:

```python
mlflow.config.enable_async_logging(True)
```

The tracking server still receives every metric -- it just happens in the background. The speedup is most noticeable with remote tracking servers.

### Artifact Organization

Use `artifact_path` to create a clean folder structure inside each run:

```text
run/artifacts/
    config/
        llm/              -- model configuration
        evaluation/       -- eval settings
    responses/            -- individual LLM responses
    reports/              -- evaluation reports
```

A consistent naming convention makes artifacts easy to browse in the UI and download programmatically.

### Artifact Storage Backends

MLflow stores artifacts in a configurable backend:

| Backend | URI scheme | When to use |
|---------|-----------|-------------|
| Local filesystem | `./mlartifacts` | Development, single machine |
| S3 / MinIO | `s3://bucket/path` | AWS production |
| GCS | `gs://bucket/path` | Google Cloud production |
| Azure Blob | `wasbs://container@account/path` | Azure production |

Our tutorial uses a local artifact store. In production, point `--default-artifact-root` at an object store when starting the MLflow server.

## Step-by-Step

### Steps 1-3: Nested Runs (Config Sweep)

Run a 3x3 sweep (3 temperatures x 3 prompt styles = 9 configs) inside a parent run. Each child logs its own params, metrics, and tags. The parent summarizes the best configurations.

```python
with mlflow.start_run(run_name="LLM Config Sweep"):
    for temperature in [0.3, 0.7, 1.0]:
        for variant_name, system_prompt in PROMPT_VARIANTS.items():
            with mlflow.start_run(run_name=f"temp_{temperature}_style_{variant_name}", nested=True):
                mlflow.log_params({"temperature": temperature, "prompt_variant": variant_name})
                result = call_llm(client, question, temperature=temperature, system_prompt=system_prompt)
                mlflow.log_metrics({"response_length": len(result), "latency_seconds": latency})
```

### Steps 4-5: Async Logging

Enable async logging, process 8 prompts with step-based metrics, then compare sync vs async timing:

```python
mlflow.config.enable_async_logging(True)
for i, prompt in enumerate(prompts):
    result = call_llm(client, prompt)
    mlflow.log_metric("latency_ms", result["latency"], step=i)  # returns immediately
```

### Steps 6-7: Artifact Organization

Use `artifact_path` to create nested subfolders and `log_artifacts()` for bulk directory upload:

```python
mlflow.log_artifact(config_path, artifact_path="config/llm")
mlflow.log_artifact(eval_path, artifact_path="config/evaluation")
mlflow.log_artifacts(reports_dir, artifact_path="reports")
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M1_tracking/3_advanced_tracking_patterns
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
Part 1: Nested Runs -- LLM Configuration Sweep
============================================================
  Model:           gemma-chat
  Temperatures:    [0.3, 0.7, 1.0]
  Prompt variants: ['concise', 'detailed', 'creative']
  Total configs:   9

============================================================
Step 1: Running sweep (nested runs)
============================================================
  temp_0.3_style_concise             length=  142  tokens=   58  latency=1.23s
  temp_0.3_style_detailed            length=  891  tokens=  234  latency=3.45s
  ...

============================================================
Step 2: Parent-run summary
============================================================
  Most detailed: temp_1.0_style_detailed  (length=1023)
  Most concise:  temp_0.3_style_concise  (length=142)
  Fastest:       temp_0.3_style_concise  (latency=1.23s)

============================================================
Step 4: Async step-based logging
============================================================
  Async logging ENABLED
  Processing 8 prompts through LLM...
    [ 0] What is machine learning?                      latency=1.23s  tokens= 85
    ...

============================================================
Step 5: Sync vs Async timing comparison
============================================================
  Synchronous logging...
    Time: 0.1234s
  Asynchronous logging...
    Time: 0.0012s
  Results:
    Sync:  0.1234s
    Async: 0.0012s
    Speedup: 102.8x

============================================================
Step 6: Organized artifact subfolders
============================================================
  Logged -> config/llm/llm_config.json
  Logged -> config/evaluation/eval_config.json
  Logged -> responses/prompt_0.txt
  ...

============================================================
Step 7: Bulk directory upload
============================================================
  Logged directory -> reports/
    - reports/train_report.json
    - reports/validation_report.json
    - reports/test_report.json
```

## Key Takeaways

- **Nested runs** (`nested=True`) create parent-child hierarchies -- ideal for LLM configuration sweeps and prompt engineering experiments.
- Query child runs with `filter_string="tags.mlflow.parentRunId = '<parent_id>'"`.
- **Async logging** (`mlflow.config.enable_async_logging(True)`) offloads metric I/O to a background thread, keeping evaluation code unblocked.
- **Batch APIs** (`log_params()`, `log_metrics()`) reduce round-trips when logging many values at once.
- Use **`artifact_path`** to organize artifacts into clean subfolder hierarchies (config/, responses/, reports/).
- Use **`log_artifacts()`** to upload an entire directory of files at once.
- In production, configure an object store (S3, GCS, Azure Blob) as the artifact backend for durability and shared access.

## Next Steps

You have completed **Module 1: Tracking**. Continue to **L1-M2 Tracing** where you will learn how MLflow captures execution traces from LLM calls -- both automatically via autologging and manually via decorators and spans.
