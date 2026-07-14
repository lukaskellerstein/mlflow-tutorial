# L2-1.2 — Async and Batch Logging

**Level:** Practitioner
**Duration:** 30 min

## Overview

When training loops log hundreds or thousands of metrics, synchronous logging can become a bottleneck. This lesson shows how to use MLflow's async logging to keep training unblocked, how to log step-based metrics that produce training curves in the UI, and how to use batch APIs (`log_metrics`, `log_params`) to reduce round-trips to the tracking server.

## Prerequisites

- Completed: L1-M1 (Tracking), L2-1.1 (Nested Runs)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Async Logging

By default, every `mlflow.log_metric()` call blocks until the tracking server acknowledges the write. For fast training loops this creates unnecessary pauses. Enabling async logging offloads the I/O to a background thread so your training code continues immediately.

```python
mlflow.config.enable_async_logging(True)
```

The tracking server still receives every metric -- it just happens in the background. MLflow handles ordering and delivery guarantees internally.

### Step-Based Metrics

Pass the `step` parameter to `log_metric()` to record a metric at a specific training step. MLflow uses steps to render training curves in the UI.

```python
for step in range(num_steps):
    mlflow.log_metric("loss", loss_value, step=step)
    mlflow.log_metric("accuracy", acc_value, step=step)
```

This produces line charts in the MLflow UI showing how loss decreases and accuracy increases over training.

### Batch Logging

When you have many metrics or parameters to log at once, use the batch APIs to send them in a single request instead of one-at-a-time:

```python
mlflow.log_params({"lr": "0.01", "epochs": "100", "batch_size": "32"})
mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93, "loss": 0.12})
```

This reduces network overhead -- one round-trip instead of N.

## Step-by-Step

### Step 1: Async Step-Based Logging

Enable async logging, then simulate a 25-step training loop that logs loss and accuracy at each step. The `log_metric()` calls return immediately while the data is sent in the background.

```python
mlflow.config.enable_async_logging(True)

with mlflow.start_run(run_name="async_training_loop"):
    for step in range(25):
        loss = 2.0 * math.exp(-0.12 * step)
        accuracy = 1.0 - 0.9 * math.exp(-0.1 * step)
        mlflow.log_metric("train_loss", loss, step=step)
        mlflow.log_metric("train_accuracy", accuracy, step=step)
```

### Step 2: Batch Logging

Log 9 parameters and 15 metrics in just two calls. This is the preferred pattern when you have a dictionary of results to record.

```python
mlflow.log_params({"model_type": "gradient_boosting", "n_estimators": "200", ...})
mlflow.log_metrics({"train_accuracy": 0.95, "val_accuracy": 0.93, ...})
```

### Step 3: Sync vs Async Timing

Run the same logging loop twice -- once synchronous, once asynchronous -- and compare wall-clock time. The async version should be measurably faster, especially against remote tracking servers with network latency.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_tracking/2_async_batch_logging
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Async Logging with Step-Based Metrics
============================================================
  Async logging ENABLED
  Simulating a 25-step training loop...

    Step  0: loss=2.0000  accuracy=0.1000
    Step  5: loss=1.1941  accuracy=0.4470
    Step 10: loss=0.6619  accuracy=0.6988
    Step 15: loss=0.3296  accuracy=0.8534
    Step 20: loss=0.2816  accuracy=0.9349
    ...

============================================================
Part 2: Batch Logging with log_metrics() and log_params()
============================================================
  Logged 9 params in a single log_params() call
  Logged 15 metrics in a single log_metrics() call
    ...

============================================================
Part 3: Sync vs Async Timing Comparison
============================================================
  Synchronous logging (30 steps)...
    Time: 0.XXXX s
  Asynchronous logging (30 steps)...
    Time: 0.XXXX s

  Results:
    Sync:  0.XXXX s
    Async: 0.XXXX s
    Speedup: X.Xx
```

In the MLflow UI, navigate to the experiment and open the "async_training_loop" run. You will see line charts for `train_loss` (decreasing) and `train_accuracy` (increasing) plotted over steps.

## Key Takeaways

- **`mlflow.config.enable_async_logging(True)`** offloads logging I/O to a background thread so training code is not blocked.
- **Step-based metrics** (`step=` parameter) produce training curves in the MLflow UI -- essential for monitoring convergence.
- **`log_params()`** and **`log_metrics()`** accept dictionaries, sending all values in a single server round-trip.
- Async logging provides the biggest speedup against remote tracking servers; locally the difference may be modest.
- Always disable async logging when you need to guarantee that all metrics are flushed before reading them back.

## Next Steps

In L2-1.3 (Artifact Management Deep Dive), you will explore advanced artifact logging -- images, tables, figures -- and learn how to organize artifacts for complex experiments.
