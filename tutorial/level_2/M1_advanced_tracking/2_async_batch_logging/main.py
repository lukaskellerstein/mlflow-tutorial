"""
L2-1.2 — Async and Batch Logging

Demonstrates MLflow's async and batch logging capabilities:
- Async logging: non-blocking metric logging during training loops
- Step-based metrics: logging loss/accuracy curves over training steps
- Batch logging: log_metrics() and log_params() for bulk operations
- Sync vs async timing comparison
"""

import math
import time

import mlflow


def part1_async_step_logging() -> None:
    """Enable async logging and simulate a training loop with step-based metrics."""
    print("=" * 60)
    print("Part 1: Async Logging with Step-Based Metrics")
    print("=" * 60)

    mlflow.config.enable_async_logging(True)
    print("  Async logging ENABLED")
    print("  Simulating a 25-step training loop...\n")

    with mlflow.start_run(run_name="async_training_loop") as run:
        mlflow.log_param("model_type", "simulated_nn")
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("num_steps", 25)

        for step in range(25):
            # Simulate decreasing loss and increasing accuracy
            loss = 2.0 * math.exp(-0.12 * step) + 0.05 * (step % 3)
            accuracy = 1.0 - 0.9 * math.exp(-0.1 * step)

            # Step-based logging -- these calls return immediately (async)
            mlflow.log_metric("train_loss", round(loss, 4), step=step)
            mlflow.log_metric("train_accuracy", round(accuracy, 4), step=step)

            if step % 5 == 0:
                print(f"    Step {step:2d}: loss={loss:.4f}  accuracy={accuracy:.4f}")

        # Log final values
        mlflow.log_metric("final_loss", round(loss, 4))
        mlflow.log_metric("final_accuracy", round(accuracy, 4))

    mlflow.config.enable_async_logging(False)

    print(f"\n  Run ID: {run.info.run_id}")
    print("  Async logging produced step-based curves in MLflow UI.")
    print("  View the charts at: http://127.0.0.1:5000\n")


def part2_batch_logging() -> None:
    """Demonstrate batch logging with log_metrics() and log_params()."""
    print("=" * 60)
    print("Part 2: Batch Logging with log_metrics() and log_params()")
    print("=" * 60)

    with mlflow.start_run(run_name="batch_logging_demo") as run:
        # Log many params at once
        params = {
            "model_type": "gradient_boosting",
            "n_estimators": "200",
            "max_depth": "5",
            "learning_rate": "0.05",
            "subsample": "0.8",
            "min_samples_split": "10",
            "min_samples_leaf": "4",
            "max_features": "sqrt",
            "random_state": "42",
        }
        mlflow.log_params(params)
        print(f"  Logged {len(params)} params in a single log_params() call")

        # Log many metrics at once
        metrics = {
            "train_accuracy": 0.9542,
            "val_accuracy": 0.9310,
            "test_accuracy": 0.9285,
            "train_f1": 0.9538,
            "val_f1": 0.9295,
            "test_f1": 0.9271,
            "train_precision": 0.9601,
            "val_precision": 0.9350,
            "test_precision": 0.9320,
            "train_recall": 0.9480,
            "val_recall": 0.9242,
            "test_recall": 0.9225,
            "train_loss": 0.1234,
            "val_loss": 0.1567,
            "test_loss": 0.1612,
        }
        mlflow.log_metrics(metrics)
        print(f"  Logged {len(metrics)} metrics in a single log_metrics() call")

    print(f"\n  Run ID: {run.info.run_id}")
    print("  Batch logging avoids multiple round-trips to the server.\n")


def part3_sync_vs_async_timing() -> None:
    """Compare sync vs async logging performance."""
    print("=" * 60)
    print("Part 3: Sync vs Async Timing Comparison")
    print("=" * 60)

    num_steps = 30

    # --- Synchronous logging ---
    mlflow.config.enable_async_logging(False)
    print(f"\n  Synchronous logging ({num_steps} steps)...")

    with mlflow.start_run(run_name="sync_timing_test"):
        t_start = time.perf_counter()
        for step in range(num_steps):
            mlflow.log_metric("sync_metric_a", step * 0.1, step=step)
            mlflow.log_metric("sync_metric_b", step * 0.2, step=step)
        sync_elapsed = time.perf_counter() - t_start

    print(f"    Time: {sync_elapsed:.4f}s")

    # --- Asynchronous logging ---
    mlflow.config.enable_async_logging(True)
    print(f"\n  Asynchronous logging ({num_steps} steps)...")

    with mlflow.start_run(run_name="async_timing_test"):
        t_start = time.perf_counter()
        for step in range(num_steps):
            mlflow.log_metric("async_metric_a", step * 0.1, step=step)
            mlflow.log_metric("async_metric_b", step * 0.2, step=step)
        async_elapsed = time.perf_counter() - t_start

    mlflow.config.enable_async_logging(False)

    print(f"    Time: {async_elapsed:.4f}s")

    # --- Results ---
    print(f"\n  Results:")
    print(f"    Sync:  {sync_elapsed:.4f}s")
    print(f"    Async: {async_elapsed:.4f}s")
    if sync_elapsed > 0:
        speedup = sync_elapsed / max(async_elapsed, 0.0001)
        print(f"    Speedup: {speedup:.1f}x")
    print()
    print("  Async logging returns immediately, offloading I/O to a")
    print("  background thread. The speedup is most noticeable when the")
    print("  tracking server has higher latency (remote servers, network).\n")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M1_advanced_tracking/2_async_batch_logging")

    part1_async_step_logging()
    part2_batch_logging()
    part3_sync_vs_async_timing()

    print("=" * 60)
    print("Done! View all runs in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)
