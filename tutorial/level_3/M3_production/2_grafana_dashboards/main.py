"""
L3-3.2 — Grafana Dashboards for MLflow

Export LLM service metrics to Prometheus and generate a Grafana dashboard
configuration for production monitoring.  The script:

  1. Defines custom Prometheus metrics (histogram, counters, gauge)
  2. Starts a Prometheus-compatible metrics server on port 8099
  3. Wraps ChatOllama calls with an instrumented service class that records
     latency, token usage, errors, and active-request count
  4. Generates sample traffic (10 varied queries) to populate metrics
  5. Produces a Grafana dashboard JSON and logs it as an MLflow artifact
  6. Queries the local /metrics endpoint to verify export, then shuts down
"""

import json
import time
import threading
from typing import Any

import mlflow
import pandas as pd
import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    REGISTRY,
    generate_latest,
)


# ---------------------------------------------------------------------------
# 1. Prometheus metrics definitions
# ---------------------------------------------------------------------------
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
    "Total tokens consumed by LLM requests",
    ["model", "direction"],  # direction: prompt | completion
)

LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM request errors",
    ["model", "error_type"],
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of LLM requests currently in flight",
)


# ---------------------------------------------------------------------------
# 2. Instrumented LLM service
# ---------------------------------------------------------------------------
class InstrumentedLLMService:
    """Wraps a ChatOllama model with Prometheus and MLflow instrumentation."""

    def __init__(self, model: str = "gemma4:e2b", temperature: float = 0.7):
        self.model_name = model
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.call_count = 0

    def invoke(self, prompt: str) -> dict[str, Any]:
        """Send *prompt* to the LLM; record metrics in Prometheus and MLflow."""
        self.call_count += 1
        LLM_ACTIVE_REQUESTS.inc()
        start = time.time()

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            output_text = result.content

            # -- Prometheus metrics --
            LLM_REQUEST_DURATION.observe(elapsed)
            LLM_REQUEST_TOTAL.labels(model=self.model_name, status="success").inc()

            # Estimate tokens (rough: 1 token ~ 4 chars)
            prompt_tokens = max(len(prompt) // 4, 1)
            completion_tokens = max(len(output_text) // 4, 1)
            LLM_TOKENS_USED.labels(model=self.model_name, direction="prompt").inc(prompt_tokens)
            LLM_TOKENS_USED.labels(model=self.model_name, direction="completion").inc(completion_tokens)

            # -- MLflow logging --
            mlflow.log_metrics(
                {
                    "latency_s": round(elapsed, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                step=self.call_count,
            )

            return {
                "output": output_text,
                "latency_s": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "status": "success",
            }

        except Exception as exc:
            elapsed = time.time() - start
            error_type = type(exc).__name__

            LLM_REQUEST_DURATION.observe(elapsed)
            LLM_REQUEST_TOTAL.labels(model=self.model_name, status="error").inc()
            LLM_ERRORS_TOTAL.labels(model=self.model_name, error_type=error_type).inc()

            mlflow.log_metrics(
                {"latency_s": round(elapsed, 3), "error": 1},
                step=self.call_count,
            )

            return {
                "output": "",
                "latency_s": round(elapsed, 3),
                "status": "error",
                "error": str(exc),
            }

        finally:
            LLM_ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# 3. Grafana dashboard generator
# ---------------------------------------------------------------------------
def generate_grafana_dashboard() -> dict:
    """Return a Grafana dashboard JSON structure with four panels."""

    def _panel(title: str, expr: str, panel_id: int, y: int,
               panel_type: str = "timeseries") -> dict:
        return {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "gridPos": {"h": 8, "w": 12, "x": (panel_id % 2) * 12, "y": y},
            "targets": [
                {
                    "expr": expr,
                    "refId": "A",
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                }
            ],
            "datasource": {"type": "prometheus", "uid": "prometheus"},
        }

    panels = [
        _panel(
            "LLM Request Rate",
            'rate(llm_request_total{status="success"}[1m])',
            1, 0,
        ),
        _panel(
            "LLM Latency (p50 / p95 / p99)",
            "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            2, 0,
        ),
        _panel(
            "LLM Error Rate",
            'rate(llm_request_total{status="error"}[1m])',
            3, 8,
        ),
        _panel(
            "Token Usage Rate",
            "rate(llm_tokens_used_total[1m])",
            4, 8,
        ),
    ]

    return {
        "dashboard": {
            "id": None,
            "uid": "mlflow-llm-monitoring",
            "title": "MLflow LLM Monitoring",
            "tags": ["mlflow", "llm", "production"],
            "timezone": "browser",
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
        },
        "overwrite": True,
    }


# ---------------------------------------------------------------------------
# 4. Sample queries for traffic generation
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in two sentences.",
    "Write a haiku about machine learning.",
    "List three benefits of exercise.",
    "What is 42 * 17?",
    "Summarize the theory of relativity briefly.",
    "Give me a fun fact about octopuses.",
    "What does the acronym REST stand for?",
    "Name three programming languages created in the 1990s.",
    "Why is the sky blue? Answer in one sentence.",
]


# ---------------------------------------------------------------------------
# 5. Metrics verification
# ---------------------------------------------------------------------------
def verify_metrics(port: int) -> dict[str, float]:
    """Scrape the local Prometheus endpoint and return key metric values."""
    url = f"http://localhost:{port}/metrics"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()

    text = resp.text
    summary: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("llm_request_total"):
            parts = line.split()
            summary[parts[0]] = float(parts[1])
        elif line.startswith("llm_tokens_used_total"):
            parts = line.split()
            summary[parts[0]] = float(parts[1])
        elif line.startswith("llm_errors_total"):
            parts = line.split()
            summary[parts[0]] = float(parts[1])
        elif line.startswith("llm_request_duration_seconds_count"):
            parts = line.split()
            summary["llm_request_duration_seconds_count"] = float(parts[1])
    return summary


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    metrics_port = 8099

    print("=" * 60)
    print("L3-3.2 — Grafana Dashboards for MLflow")
    print("=" * 60)

    # -- Part 1: Start Prometheus metrics server ----------------------------
    print("\n--- Part 1: Starting Prometheus metrics server ---")
    start_http_server(metrics_port)
    print(f"  Metrics server listening on http://localhost:{metrics_port}/metrics")

    # -- Part 2: Create instrumented LLM service ----------------------------
    print("\n--- Part 2: Creating instrumented LLM service ---")
    service = InstrumentedLLMService(model="gemma4:e2b", temperature=0.7)
    print(f"  Model: {service.model_name}")

    # -- Part 3: Generate sample traffic ------------------------------------
    print("\n--- Part 3: Generating sample LLM traffic ---")
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="grafana_dashboard_demo") as run:
        mlflow.log_params({
            "model": service.model_name,
            "num_queries": len(SAMPLE_QUERIES),
            "metrics_port": metrics_port,
        })

        for idx, query in enumerate(SAMPLE_QUERIES, 1):
            print(f"  [{idx:2d}/{len(SAMPLE_QUERIES)}] {query[:50]:50s} ", end="", flush=True)
            res = service.invoke(query)
            results.append({"query": query, **res})
            status_icon = "OK" if res["status"] == "success" else "ERR"
            print(f" {status_icon}  {res['latency_s']:.2f}s")

        # Aggregate MLflow summary
        successes = sum(1 for r in results if r["status"] == "success")
        errors = len(results) - successes
        avg_latency = (
            sum(r["latency_s"] for r in results if r["status"] == "success")
            / max(successes, 1)
        )
        total_prompt_tok = sum(r.get("prompt_tokens", 0) for r in results)
        total_compl_tok = sum(r.get("completion_tokens", 0) for r in results)

        mlflow.log_metrics({
            "total_requests": len(results),
            "total_successes": successes,
            "total_errors": errors,
            "avg_latency_s": round(avg_latency, 3),
            "total_prompt_tokens": total_prompt_tok,
            "total_completion_tokens": total_compl_tok,
        })

        # Save results table
        df = pd.DataFrame([
            {
                "query": r["query"],
                "status": r["status"],
                "latency_s": r["latency_s"],
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
            }
            for r in results
        ])
        csv_path = "/tmp/llm_traffic_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="traffic")

        # -- Part 4: Generate Grafana dashboard config ----------------------
        print("\n--- Part 4: Generating Grafana dashboard config ---")
        dashboard = generate_grafana_dashboard()
        dash_path = "/tmp/grafana_llm_dashboard.json"
        with open(dash_path, "w") as fh:
            json.dump(dashboard, fh, indent=2)
        mlflow.log_artifact(dash_path, artifact_path="dashboards")
        print(f"  Dashboard JSON saved and logged as MLflow artifact")
        print(f"  Panels: {[p['title'] for p in dashboard['dashboard']['panels']]}")
        print(f"  Import via Grafana UI -> Dashboards -> Import -> Upload JSON")

        # -- Part 5: Verify Prometheus metrics ------------------------------
        print("\n--- Part 5: Verifying Prometheus metrics ---")
        try:
            metrics = verify_metrics(metrics_port)
            print(f"  Scraped {len(metrics)} metric series from :{ metrics_port}/metrics")
            for name, value in sorted(metrics.items()):
                print(f"    {name}: {value}")
        except Exception as exc:
            print(f"  Warning: could not scrape metrics endpoint: {exc}")

        # Final summary
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(f"  Requests sent:       {len(results)}")
        print(f"  Successes:           {successes}")
        print(f"  Errors:              {errors}")
        print(f"  Avg latency:         {avg_latency:.2f}s")
        print(f"  Total prompt tokens: {total_prompt_tok}")
        print(f"  Total compl. tokens: {total_compl_tok}")
        print(f"  Prometheus:          http://localhost:{metrics_port}/metrics")
        print(f"  MLflow run ID:       {run.info.run_id}")
        print(f"  MLflow UI:           http://127.0.0.1:5000")
        print(f"  Grafana dashboard:   {dash_path}")
        print(f"{'=' * 60}")

    print("\nDone! The Prometheus metrics server will stop when this process exits.")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M3_production/2_grafana_dashboards")
    main()
