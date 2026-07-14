# L3-3.2 — Grafana Dashboards for MLflow

**Level:** Expert
**Duration:** 2 hours

## Overview

MLflow is excellent for experiment tracking and model evaluation, but production LLM services need real-time operational monitoring: request rates, latency percentiles, error budgets, and token consumption trends. This lesson bridges MLflow's experiment-centric view with Prometheus/Grafana's time-series monitoring by building an instrumented LLM service that exports metrics to both systems simultaneously.

You will learn how to define custom Prometheus metrics, instrument LLM calls, generate a reusable Grafana dashboard, and verify that the full pipeline works end-to-end.

## Prerequisites

- Completed: L1-M1 (Tracking), L1-M4 (Evaluation), L3-M3.1 (Production Tracing)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled
- Prometheus running at http://localhost:9090 (via `podman compose up -d` from `infra/`)
- Grafana running at http://localhost:3000 (admin/admin)

## Concepts

### Why Two Systems?

MLflow and Prometheus/Grafana solve different problems:

| Aspect | MLflow | Prometheus + Grafana |
|--------|--------|---------------------|
| Focus | Experiments, runs, model versions | Real-time operational metrics |
| Time resolution | Per-run or per-step | Seconds-level scraping |
| Querying | `mlflow.search_runs()` | PromQL (rate, histogram_quantile) |
| Alerting | None built-in | Grafana alerting with thresholds |
| Use case | "Which model version is best?" | "Is the service healthy right now?" |

In production you want both: MLflow for offline analysis and model lifecycle, Prometheus/Grafana for live dashboards and alerting.

### Prometheus Metric Types

- **Counter** — monotonically increasing value (e.g., total requests, total tokens). Use `rate()` in PromQL to get per-second rates.
- **Histogram** — samples observations into configurable buckets (e.g., latency). Enables percentile queries via `histogram_quantile()`.
- **Gauge** — value that can go up and down (e.g., active in-flight requests).

### Key LLM Metrics to Monitor

1. **Request rate** — how many LLM calls per second/minute
2. **Latency distribution** — p50, p95, p99 response times
3. **Error rate** — failed requests as a fraction of total
4. **Token usage** — prompt and completion tokens over time
5. **Active requests** — concurrency / saturation indicator

## Step-by-Step

### Step 1: Define Prometheus Metrics

We create five metrics covering the RED method (Rate, Errors, Duration) plus token usage and saturation:

```python
from prometheus_client import Counter, Gauge, Histogram

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "Latency of LLM requests in seconds",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60),
)

LLM_REQUEST_TOTAL = Counter(
    "llm_request_total",
    "Total number of LLM requests",
    ["model", "status"],
)

LLM_TOKENS_USED = Counter(
    "llm_tokens_used_total",
    "Total tokens consumed",
    ["model", "direction"],  # prompt | completion
)

LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM request errors",
    ["model", "error_type"],
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of requests currently in flight",
)
```

The histogram buckets are tuned for LLM latencies (0.5s to 60s). Counters use labels so you can filter by model name or error type in PromQL.

### Step 2: Start the Metrics Server

`prometheus_client.start_http_server(8099)` exposes a `/metrics` endpoint that Prometheus can scrape. In production, you would add this target to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "llm-service"
    static_configs:
      - targets: ["host.containers.internal:8099"]
```

### Step 3: Instrument LLM Calls

The `InstrumentedLLMService` class wraps every `ChatOllama` call:

- Increments `llm_active_requests` on entry, decrements on exit
- Observes `llm_request_duration_seconds` with the elapsed time
- Increments `llm_request_total` with a `status` label (success/error)
- Estimates token counts and increments `llm_tokens_used_total`
- Logs the same metrics to MLflow for offline analysis

### Step 4: Generate Traffic and Verify

The script sends 10 varied queries, then scrapes its own `/metrics` endpoint to verify that all metric series are being exported correctly.

### Step 5: Grafana Dashboard

A JSON dashboard configuration is generated and logged as an MLflow artifact. It contains four panels:

1. **LLM Request Rate** — `rate(llm_request_total{status="success"}[1m])`
2. **LLM Latency (p95)** — `histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))`
3. **LLM Error Rate** — `rate(llm_request_total{status="error"}[1m])`
4. **Token Usage Rate** — `rate(llm_tokens_used_total[1m])`

To import: open Grafana (http://localhost:3000), go to Dashboards > Import, and upload the JSON file from the MLflow artifact.

## Running the Lesson

```bash
cd tutorial/level_3/M3_production/2_grafana_dashboards
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-3.2 — Grafana Dashboards for MLflow
============================================================

--- Part 1: Starting Prometheus metrics server ---
  Metrics server listening on http://localhost:8099/metrics

--- Part 2: Creating instrumented LLM service ---
  Model: gemma4:e2b

--- Part 3: Generating sample LLM traffic ---
  [ 1/10] What is the capital of France?                      OK  1.23s
  [ 2/10] Explain quantum computing in two sentences.          OK  2.45s
  ...

--- Part 4: Generating Grafana dashboard config ---
  Dashboard JSON saved and logged as MLflow artifact
  Panels: ['LLM Request Rate', 'LLM Latency ...', ...]

--- Part 5: Verifying Prometheus metrics ---
  Scraped N metric series from :8099/metrics
    llm_request_total{model="gemma4:e2b",status="success"}: 10.0
    llm_tokens_used_total{...}: ...

============================================================
  Summary
============================================================
  Requests sent:       10
  ...
```

You can also visit http://localhost:8099/metrics in your browser to see raw Prometheus output, and import the generated dashboard JSON into Grafana.

## Key Takeaways

- Use Prometheus counters, histograms, and gauges to capture the RED metrics (Rate, Errors, Duration) for LLM services.
- The `prometheus_client` library makes it trivial to expose a `/metrics` endpoint that Prometheus can scrape.
- Instrument both Prometheus (real-time ops) and MLflow (offline analysis) in the same service for complete observability.
- Grafana dashboards can be version-controlled as JSON and logged as MLflow artifacts alongside the model they monitor.
- Histogram buckets should be tuned to expected LLM latencies (seconds, not milliseconds).

## Next Steps

In **L3-3.3 (Feedback Loops)**, you will close the production loop by collecting user feedback on agent responses, detecting quality drift, and feeding production data back into evaluation datasets.
