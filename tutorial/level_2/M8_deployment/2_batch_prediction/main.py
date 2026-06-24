"""
L2-8.2 -- Batch Prediction Pipelines

Demonstrates building a production-style batch prediction pipeline with
MLflow: train and log a model, run batch predictions using mlflow.pyfunc,
track results as artifacts and metrics, handle errors gracefully, and
generate CLI commands for offline batch scoring.
"""

import json
import os
import tempfile
import time

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M8_deployment/2_batch_prediction"
MODEL_NAME = "wine-batch-demo"


# ---------------------------------------------------------------------------
# Part 1: Train and log a model
# ---------------------------------------------------------------------------
def part1_train_and_log(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
) -> str:
    """Train a classifier on the wine dataset, log it with a signature."""
    print("=" * 60)
    print("Part 1: Train and Log a Model")
    print("=" * 60)

    with mlflow.start_run(run_name="train_wine_model") as run:
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)

        mlflow.log_params({
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
        })
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_test, clf.predict(X_test))
        mlflow.sklearn.log_model(
            clf, name="model", signature=signature,
            input_example=X_test.head(2),
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Model URI: {model_uri}")
        print()
        return model_uri


# ---------------------------------------------------------------------------
# Part 2: Batch prediction with mlflow.pyfunc
# ---------------------------------------------------------------------------
def part2_batch_prediction(
    model_uri: str, feature_names: list[str],
) -> tuple[pd.DataFrame, float]:
    """Load the model and run batch predictions on a synthetic batch."""
    print("=" * 60)
    print("Part 2: Batch Prediction with mlflow.pyfunc")
    print("=" * 60)

    model = mlflow.pyfunc.load_model(model_uri)
    print(f"  Loaded model from: {model_uri}")

    # Build a 60-row batch from the wine dataset (full test set + extras)
    wine = load_wine()
    batch_df = pd.DataFrame(wine.data[:60], columns=feature_names)
    print(f"  Batch size: {len(batch_df)} rows")

    start = time.perf_counter()
    predictions = model.predict(batch_df)
    elapsed = time.perf_counter() - start

    batch_df["prediction"] = predictions
    print(f"  Prediction time : {elapsed:.4f}s")
    print(f"  Predictions/sec : {len(batch_df) / elapsed:.1f}")
    print(f"  Unique classes  : {sorted(int(c) for c in set(predictions))}")
    print()
    return batch_df, elapsed


# ---------------------------------------------------------------------------
# Part 3: Result tracking
# ---------------------------------------------------------------------------
def part3_result_tracking(results_df: pd.DataFrame, elapsed: float) -> None:
    """Log batch results as artifacts and record batch metrics."""
    print("=" * 60)
    print("Part 3: Result Tracking")
    print("=" * 60)

    with mlflow.start_run(run_name="batch_prediction_results"):
        # --- Metrics ---
        batch_size = len(results_df)
        pps = batch_size / elapsed
        mlflow.log_metrics({
            "batch_size": batch_size,
            "prediction_time_sec": round(elapsed, 4),
            "predictions_per_second": round(pps, 2),
        })
        print(f"  Logged metrics: batch_size={batch_size}, "
              f"time={elapsed:.4f}s, pps={pps:.1f}")

        # --- Prediction CSV artifact ---
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "batch_predictions.csv")
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="predictions")
            print("  Logged artifact: predictions/batch_predictions.csv")

            # --- Summary JSON artifact ---
            class_dist = results_df["prediction"].value_counts().to_dict()
            summary = {
                "batch_size": batch_size,
                "prediction_time_sec": round(elapsed, 4),
                "predictions_per_second": round(pps, 2),
                "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
                "num_classes": len(class_dist),
            }
            summary_path = os.path.join(tmp, "prediction_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(summary_path, artifact_path="predictions")
            print("  Logged artifact: predictions/prediction_summary.json")
            print(f"  Class distribution: {class_dist}")
    print()


# ---------------------------------------------------------------------------
# Part 4: Pipeline pattern
# ---------------------------------------------------------------------------
def part4_pipeline(model_uri: str, feature_names: list[str]) -> None:
    """End-to-end pipeline: load -> validate -> predict -> log results."""
    print("=" * 60)
    print("Part 4: Complete Pipeline Pattern")
    print("=" * 60)

    with mlflow.start_run(run_name="batch_pipeline"):
        # Step 1 -- Load data
        print("  [1/4] Loading data ...")
        wine = load_wine()
        batch_df = pd.DataFrame(wine.data, columns=feature_names)
        mlflow.log_metric("input_rows", len(batch_df))

        # Step 2 -- Validate
        print("  [2/4] Validating inputs ...")
        missing = batch_df.isnull().sum().sum()
        valid_mask = ~batch_df.isnull().any(axis=1)
        valid_df = batch_df[valid_mask]
        skipped = len(batch_df) - len(valid_df)
        mlflow.log_metrics({"valid_rows": len(valid_df), "skipped_rows": skipped})
        print(f"        {len(valid_df)} valid, {skipped} skipped "
              f"({missing} missing values)")

        # Step 3 -- Predict (with error handling)
        print("  [3/4] Running predictions ...")
        model = mlflow.pyfunc.load_model(model_uri)
        start = time.perf_counter()
        try:
            preds = model.predict(valid_df)
            elapsed = time.perf_counter() - start
            valid_df = valid_df.copy()
            valid_df["prediction"] = preds
            failed = 0
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"        ERROR: {exc}")
            failed = len(valid_df)
            valid_df = valid_df.copy()
            valid_df["prediction"] = None

        mlflow.log_metrics({
            "predicted_rows": len(valid_df) - failed,
            "failed_rows": failed,
            "pipeline_prediction_time_sec": round(elapsed, 4),
        })
        print(f"        {len(valid_df) - failed} predicted, "
              f"{failed} failed in {elapsed:.4f}s")

        # Step 4 -- Log results
        print("  [4/4] Logging results ...")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "pipeline_output.csv")
            valid_df.to_csv(out_path, index=False)
            mlflow.log_artifact(out_path, artifact_path="pipeline")
            print("        Artifact: pipeline/pipeline_output.csv")

        mlflow.set_tag("pipeline_status", "success" if failed == 0 else "partial")
        print(f"        Status: {'success' if failed == 0 else 'partial'}")
    print()


# ---------------------------------------------------------------------------
# Part 5: CLI batch prediction
# ---------------------------------------------------------------------------
def part5_cli_batch(model_uri: str, feature_names: list[str]) -> None:
    """Print CLI commands and create a sample input file."""
    print("=" * 60)
    print("Part 5: CLI Batch Prediction")
    print("=" * 60)

    with mlflow.start_run(run_name="cli_batch_setup"):
        # Create sample input CSV
        wine = load_wine()
        sample = pd.DataFrame(wine.data[:5], columns=feature_names)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sample_input.csv")
            sample.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="cli")
            print("  Logged sample input: cli/sample_input.csv")

    print(f"\n  CLI commands for batch prediction:")
    print(f'    mlflow models predict -m "{model_uri}" \\')
    print(f'      -i sample_input.csv -o predictions.csv')
    print()
    print(f"  With content type:")
    print(f'    mlflow models predict -m "{model_uri}" \\')
    print(f'      -i sample_input.csv --content-type csv')
    print()
    print("  Scheduling examples:")
    print("    # Cron (daily at 2 AM)")
    print('    0 2 * * * cd /app && mlflow models predict -m "models:/model/1" \\')
    print("      -i /data/daily_batch.csv -o /output/predictions.csv")
    print()
    print("    # Airflow / Temporal / Prefect")
    print("    #   Wrap the pipeline function (Part 4) in a task/activity")
    print("    #   and schedule it as part of a DAG or workflow.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print()
    print("=" * 60)
    print("L2-8.2 -- Batch Prediction Pipelines")
    print("=" * 60)
    print()

    # Load dataset
    wine = load_wine()
    feature_names = wine.feature_names
    X = pd.DataFrame(wine.data, columns=feature_names)
    y = pd.Series(wine.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"  Wine dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print()

    model_uri = part1_train_and_log(X_train, X_test, y_train, y_test)
    results_df, elapsed = part2_batch_prediction(model_uri, list(feature_names))
    part3_result_tracking(results_df, elapsed)
    part4_pipeline(model_uri, list(feature_names))
    part5_cli_batch(model_uri, list(feature_names))

    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  View runs at: {TRACKING_URI}")
    print(f"  Experiment  : {EXPERIMENT_NAME}")
    print()


if __name__ == "__main__":
    main()
