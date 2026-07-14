"""
L1-M1.4 — System Metrics Logging

Demonstrates how MLflow can automatically collect system-level
metrics (CPU, memory, disk, network) during a run. We enable system metrics
logging, run an LLM call, and inspect the captured metrics.
"""

import time

import mlflow
from mlflow import MlflowClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_tracking/4_system_metrics"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"


def run_llm_workload(client: OpenAI) -> dict:
    """Run multiple LLM calls to generate measurable system load."""
    prompts = [
        "Write a detailed explanation of how gradient descent works in neural networks.",
        "Explain the difference between supervised, unsupervised, and reinforcement learning.",
        "Describe the architecture of a large language model and how it generates text.",
    ]

    total_tokens = 0
    for prompt in prompts:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        total_tokens += response.usage.total_tokens

    return {"total_tokens": total_tokens, "num_calls": len(prompts)}


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    # ------------------------------------------------------------------
    # 1. What are system metrics?
    # ------------------------------------------------------------------
    print("=" * 60)
    print("System Metrics Logging")
    print("=" * 60)
    print()
    print("MLflow can automatically collect hardware metrics during a run:")
    print("  - CPU utilization (%)")
    print("  - System memory usage (MB and %)")
    print("  - Disk usage (%, MB used, MB available)")
    print("  - Network I/O (MB received and transmitted)")
    print("  - GPU utilization (if NVIDIA/AMD GPU is available)")
    print()

    # ------------------------------------------------------------------
    # 2. Enable system metrics and configure sampling
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Enable system metrics logging")
    print("=" * 60)

    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(5)
    mlflow.set_system_metrics_samples_before_logging(1)

    print("  System metrics logging: ENABLED")
    print("  Sampling interval:      5 seconds")
    print()

    # ------------------------------------------------------------------
    # 3. Run LLM calls inside a tracked run
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Run LLM calls with system metrics collection")
    print("=" * 60)

    with mlflow.start_run(run_name="system_metrics_demo") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        print("  Making LLM calls (this generates CPU/memory load)...")

        results = run_llm_workload(client)
        mlflow.log_params({"model": MODEL, "num_calls": results["num_calls"]})
        mlflow.log_metric("total_tokens", results["total_tokens"])

        print(f"  Total tokens: {results['total_tokens']}")
        print("  Waiting 12 seconds for system metrics collection...")
        time.sleep(12)

    print("  Run completed.")
    print()

    # ------------------------------------------------------------------
    # 4. Query and display the system metrics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Inspect collected system metrics")
    print("=" * 60)

    mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)
    run_data = mlflow_client.get_run(run_id)

    system_metrics = {
        k: v
        for k, v in run_data.data.metrics.items()
        if k.startswith("system/")
    }

    if system_metrics:
        print(f"  Found {len(system_metrics)} system metric(s):\n")
        for name, value in sorted(system_metrics.items()):
            print(f"    {name:50s} = {value:.2f}")
    else:
        print("  No system metrics found (collection may need more time).")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Open the MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Click on the run and go to the 'System Metrics' tab to see")
    print("  time-series charts for CPU, memory, disk, and network usage.")


if __name__ == "__main__":
    main()
