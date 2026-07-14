# L2-1.2 — Async and Batch Logging

**Level:** Practitioner
**Duration:** 30 min

## Overview

When running batch LLM evaluation across many prompts, synchronous logging can become a bottleneck -- each `log_metric()` call blocks until the tracking server acknowledges the write. This lesson shows how to use MLflow's async logging to keep evaluation unblocked, how to log step-based metrics that produce per-prompt charts in the UI, and how to use batch APIs (`log_metrics`, `log_params`) to reduce round-trips when recording aggregate results.

## Prerequisites

- Completed: L1-M1 (Tracking), L2-1.1 (Nested Runs)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Async Logging

By default, every `mlflow.log_metric()` call blocks until the tracking server acknowledges the write. When you are evaluating an LLM across dozens of prompts and logging metrics for each one, these blocking calls add up. Enabling async logging offloads the I/O to a background thread so your evaluation code continues immediately.

```python
mlflow.config.enable_async_logging(True)
```

The tracking server still receives every metric -- it just happens in the background. MLflow handles ordering and delivery guarantees internally.

### Step-Based Metrics

Pass the `step` parameter to `log_metric()` to record a metric at a specific evaluation step. In batch LLM evaluation, each step corresponds to one prompt. MLflow uses steps to render line charts in the UI, so you can see how response length, latency, and token count vary across your prompt set.

```python
for i, prompt in enumerate(prompts):
    text, latency_ms, token_count = call_llm(client, prompt)
    mlflow.log_metric("latency_ms", latency_ms, step=i)
    mlflow.log_metric("token_count", token_count, step=i)
```

### Batch Logging

When you have many metrics or parameters to log at once -- for example, all the LLM configuration settings or aggregate evaluation results -- use the batch APIs to send them in a single request instead of one-at-a-time:

```python
mlflow.log_params({"model": "gemma-4-e4b", "temperature": "0.7", "max_tokens": "150"})
mlflow.log_metrics({"avg_latency_ms": 342.5, "avg_token_count": 87.3, "total_tokens": 1048})
```

This reduces network overhead -- one round-trip instead of N.

## Step-by-Step

### Step 1: Async Step-Based Logging

Enable async logging, then process a batch of 12 diverse prompts through the LLM. For each prompt, log three step-based metrics: `response_length`, `latency_ms`, and `token_count`. The `log_metric()` calls return immediately while the data is sent in the background.

```python
mlflow.config.enable_async_logging(True)

with mlflow.start_run(run_name="async_batch_eval"):
    for i, prompt in enumerate(PROMPTS):
        text, latency_ms, token_count = call_llm(client, prompt)
        mlflow.log_metric("response_length", len(text), step=i)
        mlflow.log_metric("latency_ms", latency_ms, step=i)
        mlflow.log_metric("token_count", token_count, step=i)
```

### Step 2: Batch Logging

Log all LLM configuration parameters at once with `log_params()`, then run a subset of prompts and log aggregate metrics (average latency, total tokens, response length statistics) in a single `log_metrics()` call.

```python
mlflow.log_params({"model": "google/gemma-4-e4b", "temperature": "0.7", ...})
mlflow.log_metrics({"avg_latency_ms": 342.5, "total_tokens": 523, ...})
```

### Step 3: Sync vs Async Timing

Pre-generate a set of LLM responses so that LLM latency does not skew the comparison. Then log the same metrics twice -- once synchronously, once asynchronously -- and compare wall-clock time. The async version should be measurably faster, especially against remote tracking servers with network latency.

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
  Processing 12 prompts through LLM...

    [ 0] What is machine learning?                      latency=  XXX.Xms  tokens=XXX  len=XXXX
    [ 1] Explain neural networks in simple terms.       latency=  XXX.Xms  tokens=XXX  len=XXXX
    ...

============================================================
Part 2: Batch Logging with log_metrics() and log_params()
============================================================
  Gathering aggregate stats from LLM responses...

  Logged 6 params in a single log_params() call
  Logged 8 metrics in a single log_metrics() call

  Aggregate Results:
    Avg latency:         XXX.XX ms
    Avg token count:     XX
    Avg response length: XXXX chars
    ...

============================================================
Part 3: Sync vs Async Timing Comparison
============================================================
  Pre-generating LLM responses for fair comparison...
  Collected 6 responses. Now comparing logging speed.

  Synchronous logging...
    Time: 0.XXXXs
  Asynchronous logging...
    Time: 0.XXXXs

  Results:
    Sync:  0.XXXXs
    Async: 0.XXXXs
    Speedup: X.Xx
```

In the MLflow UI, navigate to the experiment and open the "async_batch_eval" run. You will see line charts for `latency_ms`, `token_count`, and `response_length` plotted over steps -- each step corresponds to one evaluated prompt.

## Key Takeaways

- **`mlflow.config.enable_async_logging(True)`** offloads logging I/O to a background thread so LLM evaluation is not blocked by tracking server writes.
- **Step-based metrics** (`step=` parameter) produce per-prompt charts in the MLflow UI -- useful for spotting outlier prompts with unusual latency or token counts.
- **`log_params()`** and **`log_metrics()`** accept dictionaries, sending all values in a single server round-trip -- ideal for logging LLM configuration and aggregate evaluation results.
- Async logging provides the biggest speedup against remote tracking servers; locally the difference may be modest.
- Always disable async logging when you need to guarantee that all metrics are flushed before reading them back.

## Next Steps

In L2-1.3 (Artifact Management Deep Dive), you will explore advanced artifact logging -- images, tables, figures -- and learn how to organize artifacts for complex experiments.
