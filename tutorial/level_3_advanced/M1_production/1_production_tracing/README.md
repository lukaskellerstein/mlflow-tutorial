# L3-3.1 — Production Tracing Strategies

**Level:** Expert
**Duration:** ~45 minutes

## Overview

In production, tracing every single LLM request creates unacceptable overhead. This lesson teaches four sampling strategies that balance observability with performance, plus structured metadata patterns for filtering traces, error tracing with full context capture, and latency/token monitoring from trace data.

## Prerequisites

- Completed: L1-M5 (Tracing basics), L2-M4 (Advanced Tracing)
- MLFlow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Why Sampling Matters

A production system serving thousands of LLM requests per minute cannot trace 100% of traffic. Tracing adds serialization overhead, network I/O to the MLflow backend, and storage costs. Sampling strategies let you capture the traces that matter most while keeping overhead minimal.

### Four Sampling Strategies

| Strategy | Sampling Rate | Use Case | Overhead |
|----------|--------------|----------|----------|
| `sample_all` | 100% | Dev / Test | High |
| `sample_percentage` | 1-10% | Staging | Low |
| `sample_errors_only` | Variable | Production | Minimal |
| `sample_slow_requests` | Variable | Production | Minimal |

### Structured Metadata

Production traces need context beyond the raw LLM input/output. Attaching metadata like `request_id`, `user_id`, `environment`, and `app_version` to every trace enables filtering, debugging, and correlating traces with external systems.

### Error Tracing

When something goes wrong, you need the full picture: what was the input, what error occurred, what was the stack trace, and how does this error correlate with other failures. Error-only sampling ensures you capture every failure without tracing successful requests.

## Step-by-Step

### Step 1: Trace Sampling Strategies

The `TraceSampler` class implements four strategies as static methods. Each returns a boolean indicating whether to trace the current request.

```python
class TraceSampler:
    @staticmethod
    def sample_all() -> bool:
        return True

    @staticmethod
    def sample_percentage(rate: float) -> bool:
        return random.random() < rate

    @staticmethod
    def sample_errors_only(has_error: bool) -> bool:
        return has_error

    @staticmethod
    def sample_slow_requests(duration_ms: float, threshold_ms: float) -> bool:
        return duration_ms >= threshold_ms
```

The first two strategies (all, percentage) are **pre-call** decisions: you decide whether to trace before making the request. The last two (errors, slow) are **post-call** decisions: you call without tracing, then retroactively trace if the result meets the criteria.

### Step 2: Structured Trace Metadata

Every traced call attaches production metadata via `span.set_attributes()` and `mlflow.set_trace_tag()`:

```python
with mlflow.start_span(name="llm_call") as span:
    span.set_attributes({
        "request_id": request_id,
        "user_id": metadata["user_id"],
        "environment": metadata["environment"],
        "app_version": metadata["app_version"],
    })
    # ... make LLM call ...
    mlflow.set_trace_tag(span.trace_id, "environment", "production")
```

Tags are searchable in the MLflow UI, so you can filter traces by user, environment, or version.

### Step 3: Error Tracing

Errors are captured with full context: the original input, error type, error message, and stack trace. This information is stored both in the trace span and in a `TraceRecord` dataclass for aggregation.

```python
except Exception as exc:
    return TraceRecord(
        error=str(exc),
        error_type=type(exc).__name__,
        stack_trace=traceback.format_exc(),
    )
```

### Step 4: Performance Monitoring

Latency percentiles (p50, p95, p99) and token usage trends are computed from the collected trace records and logged to MLflow as metrics:

```python
summary = build_performance_summary(records)
mlflow.log_metrics({
    "p50_latency_ms": summary["p50_ms"],
    "p95_latency_ms": summary["p95_ms"],
    "p99_latency_ms": summary["p99_ms"],
    "total_token_estimate": summary["total_token_estimate"],
})
```

## Running the Lesson

```bash
cd tutorial/level_3/M3_production/1_production_tracing
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-3.1 — Production Tracing Strategies
============================================================
============================================================
Part 1: Trace Sampling Strategies
============================================================

  Strategy: sample_all (dev)
    Traced: yes | Duration: 1234.5ms

  Strategy: sample_percentage 50% (staging)
    Traced: yes | Duration: 987.2ms
    (or: Traced: no | Duration: 876.1ms — depends on random sampling)

  Strategy: sample_errors_only (prod)
    Would trace: False | Duration: 654.3ms

  Strategy: sample_slow_requests >500ms (prod)
    Would trace: False | Duration: 432.1ms

============================================================
Part 2: Structured Trace Metadata
============================================================
  Request 1: user=user_alice, env=production
    Duration: 1100.2ms | Tokens: ~45
    Tags set: [request_id, user_id, environment, app_version, session_id]
  ...

============================================================
Part 3: Error Tracing
============================================================
  3a. Normal call (should succeed):
    Status: OK | Duration: 890.1ms

  3b. Simulated error (invalid model):
    Status: ERROR | Type: ResponseError
    Error: model "nonexistent_model_xyz" not found...

============================================================
Part 4: Performance Monitoring
============================================================
  Requests analyzed: 7
  Latency percentiles:
    p50: 890.1 ms
    p95: 1234.5 ms
    p99: 1300.0 ms
  ...

============================================================
Production Tracing Strategy Comparison
============================================================
  Strategy                       Use Case        Overhead   Coverage
  sample_all (100%)              Dev / Test      High       Complete
  sample_percentage (1-10%)      Staging         Low        Statistical
  sample_errors_only             Production      Minimal    Errors only
  sample_slow_requests           Production      Minimal    Tail latency
```

## Key Takeaways

- **Never trace 100% in production** — use sampling to balance observability with performance overhead.
- **Pre-call vs post-call sampling**: percentage sampling decides before the call; error/slow sampling decides after.
- **Structured metadata** (request_id, user_id, environment) enables powerful filtering in the MLflow UI.
- **Error traces should capture full context**: input, error type, message, and stack trace.
- **Track latency percentiles** (p50/p95/p99) rather than averages to understand tail latency behavior.

## Next Steps

Continue to L3-3.2 (Grafana Dashboards) to build real-time monitoring dashboards that visualize the metrics collected in this lesson.
