"""
L1-1.4 — System Metrics Logging

This lesson demonstrates how MLflow can automatically collect system-level
metrics (CPU, memory, disk, network) during a run. We enable system metrics
logging, run a computationally intensive task, and then inspect the captured
metrics.
"""

import time

import mlflow
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_core_platform/4_system_metrics"


def run_intensive_task() -> dict[str, float]:
    """Train a large RandomForest to generate CPU and memory load."""
    X, y = make_classification(
        n_samples=50_000, n_features=40, n_informative=20, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)
    return {"accuracy": accuracy, "n_samples": len(X), "n_estimators": 100}


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

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

    # Enable globally — all subsequent runs will collect system metrics.
    mlflow.enable_system_metrics_logging()

    # Speed up collection: sample every 5 seconds instead of default 10.
    mlflow.set_system_metrics_sampling_interval(5)
    mlflow.set_system_metrics_samples_before_logging(1)

    print("  System metrics logging: ENABLED")
    print("  Sampling interval:      5 seconds")
    print()

    # ------------------------------------------------------------------
    # 3. Run a compute-heavy task inside a tracked run
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Run a compute-intensive task")
    print("=" * 60)

    with mlflow.start_run(run_name="system_metrics_demo") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        print("  Training a large RandomForest (this generates CPU/memory load)...")

        results = run_intensive_task()
        mlflow.log_params(
            {"model": "RandomForestClassifier", "n_estimators": results["n_estimators"]}
        )
        mlflow.log_metric("accuracy", results["accuracy"])

        print(f"  Accuracy: {results['accuracy']:.4f}")
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

    client = MlflowClient(TRACKING_URI)
    run_data = client.get_run(run_id)

    # System metrics are prefixed with "system/"
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
    print(f"  Open the MLflow UI at {TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Click on the run and go to the 'System Metrics' tab to see")
    print("  time-series charts for CPU, memory, disk, and network usage.")


if __name__ == "__main__":
    main()
