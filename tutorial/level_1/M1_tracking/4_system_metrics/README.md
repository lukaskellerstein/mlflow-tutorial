# L1-M1.4 — System Metrics Logging

**Level:** Essentials
**Duration:** 15 min

## Overview

This lesson shows how MLflow can automatically collect system-level metrics (CPU, memory, disk, network) during a run. You will enable system metrics logging, run LLM calls that generate measurable load, and inspect the captured metrics in both code and the MLflow UI.

## Prerequisites

- Completed: L1-M1.1 (First Run), L1-M1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### What Gets Logged?

When system metrics logging is enabled, MLflow periodically samples:

| Metric | Description |
|--------|-------------|
| `system/cpu_utilization_percentage` | CPU usage across all cores |
| `system/system_memory_usage_megabytes` | Total RAM in use |
| `system/system_memory_usage_percentage` | RAM usage as a percentage |
| `system/disk_usage_percentage` | Disk utilization |
| `system/disk_available_megabytes` | Free disk space |
| `system/network_receive_megabytes` | Network bytes received |
| `system/network_transmit_megabytes` | Network bytes sent |

GPU metrics are also captured if an NVIDIA or AMD GPU is available.

### When Is This Useful?

- **Identifying bottlenecks** — is your LLM inference CPU-bound or memory-bound?
- **Capacity planning** — how much memory do different models need?
- **Cost optimization** — correlate resource usage with token throughput

## Step-by-Step

### Step 1: Enable System Metrics

```python
mlflow.enable_system_metrics_logging()
mlflow.set_system_metrics_sampling_interval(5)      # sample every 5 seconds
mlflow.set_system_metrics_samples_before_logging(1)  # log after each sample
```

### Step 2: Run a Workload

We run multiple LLM calls inside a tracked run. The script then waits 12 seconds to ensure at least two sampling intervals are captured.

### Step 3: Inspect the Metrics

System metrics are prefixed with `system/` and can be queried via `MlflowClient`:

```python
client = MlflowClient(TRACKING_URI)
run_data = client.get_run(run_id)
system_metrics = {k: v for k, v in run_data.data.metrics.items() if k.startswith("system/")}
```

## Running the Lesson

```bash
cd tutorial/level_1/M1_tracking/4_system_metrics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 2: Run LLM calls with system metrics collection
============================================================
  Run ID: <generated-run-id>
  Making LLM calls (this generates CPU/memory load)...
  Total tokens: 285
  Waiting 12 seconds for system metrics collection...
  Run completed.

============================================================
Step 3: Inspect collected system metrics
============================================================
  Found 7 system metric(s):

    system/cpu_utilization_percentage                    = 45.20
    system/disk_available_megabytes                      = 234567.00
    system/disk_usage_percentage                         = 52.30
    system/network_receive_megabytes                     = 12.50
    system/network_transmit_megabytes                    = 8.30
    system/system_memory_usage_megabytes                 = 24576.00
    system/system_memory_usage_percentage                = 51.20
```

In the MLflow UI, click on the run and go to the **System Metrics** tab to see time-series charts.

## Key Takeaways

- `mlflow.enable_system_metrics_logging()` enables automatic hardware metric collection.
- System metrics are sampled periodically (configurable interval) and logged as step-based metrics.
- Metrics are prefixed with `system/` and visible in both the API and the UI's System Metrics tab.
- Useful for correlating LLM performance with resource usage.

## Next Steps

You have completed **Module 1: Tracking**. Continue to **L1-M2 Models and Registry** where you will learn to package and version LLM models with MLflow. In Level 2, M1 covers advanced tracking — nested runs, async logging, and the MlflowClient deep dive.
