# L1-1.4 — System Metrics Logging

**Level:** Essentials
**Duration:** ~15 minutes

## Overview

MLflow can automatically collect hardware-level metrics (CPU, memory, disk, network, GPU) during any run. This is useful for understanding the resource footprint of your training jobs and identifying bottlenecks. In this lesson you will enable system metrics logging, run a compute-intensive task, and inspect the captured metrics.

## Prerequisites

- Completed: L1-1.1 (Architecture Overview), L1-1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- `psutil` package (declared in `pyproject.toml`)

## Concepts

### What are system metrics?

System metrics are hardware-level measurements that MLflow collects in the background while a run is active. They give you visibility into how much CPU, memory, disk, and network your code consumes — without any manual instrumentation.

### What gets logged?

| Metric | Key | Description |
|--------|-----|-------------|
| CPU utilization | `system/cpu_utilization_percentage` | Percentage of CPU in use |
| Memory usage (MB) | `system/system_memory_usage_megabytes` | RAM used in megabytes |
| Memory usage (%) | `system/system_memory_usage_percentage` | RAM used as percentage of total |
| Disk usage (%) | `system/disk_usage_percentage` | Disk space used as percentage |
| Disk usage (MB) | `system/disk_usage_megabytes` | Disk space used in megabytes |
| Disk available (MB) | `system/disk_available_megabytes` | Free disk space in megabytes |
| Network received (MB) | `system/network_receive_megabytes` | Data received since run start |
| Network transmitted (MB) | `system/network_transmit_megabytes` | Data sent since run start |
| GPU utilization (%) | `system/gpu_utilization_percentage` | GPU usage (NVIDIA/AMD only) |
| GPU memory (%) | `system/gpu_memory_usage_percentage` | GPU memory usage (if available) |

### How does collection work?

MLflow spawns a background thread that samples metrics at a configurable interval (default: every 10 seconds). The samples are aggregated and logged as time-series data attached to the run.

Key configuration functions:
- `mlflow.enable_system_metrics_logging()` — turn on collection globally
- `mlflow.set_system_metrics_sampling_interval(seconds)` — how often to sample
- `mlflow.set_system_metrics_samples_before_logging(n)` — aggregate n samples before logging

You can also enable per-run: `mlflow.start_run(log_system_metrics=True)`.

### When is this useful?

- **Training bottleneck analysis**: is the job CPU-bound or memory-bound?
- **Resource planning**: how much RAM does your model need at peak?
- **Cost optimization**: are you over-provisioning GPU machines?
- **Debugging OOM errors**: track memory usage leading up to a crash.

## Step-by-Step

### Step 1: Enable system metrics logging

```python
mlflow.enable_system_metrics_logging()
mlflow.set_system_metrics_sampling_interval(5)
mlflow.set_system_metrics_samples_before_logging(1)
```

This enables collection globally and sets sampling to every 5 seconds.

### Step 2: Run a compute-intensive task

We train a large `RandomForestClassifier` on 50,000 samples with 40 features. This generates noticeable CPU and memory load that the system metrics monitor will capture.

```python
with mlflow.start_run(run_name="system_metrics_demo") as run:
    results = run_intensive_task()
    mlflow.log_metric("accuracy", results["accuracy"])
    time.sleep(12)  # Wait for metrics collection
```

The `time.sleep(12)` gives the background thread enough time to collect at least two data points at the 5-second interval.

### Step 3: Inspect collected metrics

```python
client = MlflowClient(TRACKING_URI)
run_data = client.get_run(run_id)
system_metrics = {
    k: v for k, v in run_data.data.metrics.items()
    if k.startswith("system/")
}
```

System metrics are stored alongside your custom metrics but prefixed with `system/`.

## Running the Lesson

```bash
cd tutorial/level_1/M1_core_platform/4_system_metrics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
System Metrics Logging
============================================================

MLflow can automatically collect hardware metrics during a run:
  - CPU utilization (%)
  - System memory usage (MB and %)
  - Disk usage (%, MB used, MB available)
  - Network I/O (MB received and transmitted)
  - GPU utilization (if NVIDIA/AMD GPU is available)

============================================================
Step 1: Enable system metrics logging
============================================================
  System metrics logging: ENABLED
  Sampling interval:      5 seconds

============================================================
Step 2: Run a compute-intensive task
============================================================
  Run ID: <run-id>
  Training a large RandomForest (this generates CPU/memory load)...
  Accuracy: 0.9985
  Waiting 12 seconds for system metrics collection...
  Run completed.

============================================================
Step 3: Inspect collected system metrics
============================================================
  Found 8 system metric(s):

    system/cpu_utilization_percentage                   = 45.30
    system/disk_available_megabytes                     = 234567.80
    system/disk_usage_megabytes                         = 456789.00
    system/disk_usage_percentage                        = 66.10
    system/network_receive_megabytes                    = 0.12
    system/network_transmit_megabytes                   = 0.05
    system/system_memory_usage_megabytes                = 12345.60
    system/system_memory_usage_percentage               = 75.20

============================================================
Done!
============================================================
  Open the MLflow UI at http://127.0.0.1:5000
  Navigate to experiment: L1/M1_core_platform/4_system_metrics
  Click on the run and go to the 'System Metrics' tab to see
  time-series charts for CPU, memory, disk, and network usage.
```

Exact values will vary based on your machine's hardware and current load.

In the MLflow UI, click on the run and look for the **System Metrics** tab — it shows time-series charts for each metric.

## Key Takeaways

- `mlflow.enable_system_metrics_logging()` turns on automatic hardware monitoring for all runs.
- Metrics are collected by a background thread at a configurable interval (default 10 seconds).
- CPU, memory, disk, and network metrics are logged automatically; GPU metrics appear if a compatible GPU is detected.
- System metrics appear under the `system/` prefix alongside your custom metrics.
- Use these metrics to diagnose resource bottlenecks and plan infrastructure.

## Next Steps

Continue to **L1-2.1 — Model Flavors** to learn how MLflow packages models from different frameworks into a standard format. In Level 2, we will explore advanced tracking features like nested runs and async logging.
