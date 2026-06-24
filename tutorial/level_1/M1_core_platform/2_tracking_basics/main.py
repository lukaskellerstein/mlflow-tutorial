"""
L1-1.2 — Experiment Tracking Basics

Demonstrates MLflow's core tracking capabilities:
- Creating experiments and runs
- Logging parameters, metrics, artifacts, and tags
- Bulk logging with log_params() and log_metrics()
- Step-based metric logging for training curves
"""

import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def train_model(
    n_estimators: int = 100, max_depth: int = 5, random_state: int = 42
) -> None:
    """Train a RandomForest on iris data and log everything to MLflow."""

    # -- Step 1: Prepare data ------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading the Iris dataset")
    print("=" * 60)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}\n")

    # -- Step 2: Start an MLflow run and log parameters ----------------------
    print("=" * 60)
    print("Step 2: Starting MLflow run and logging parameters")
    print("=" * 60)
    with mlflow.start_run(run_name="iris_random_forest") as run:
        print(f"  Run ID:   {run.info.run_id}")
        print(f"  Run Name: iris_random_forest\n")

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "test_size": 0.2,
        }
        mlflow.log_params(params)
        print(f"  Logged params: {params}\n")

        # -- Step 3: Train the model -----------------------------------------
        print("=" * 60)
        print("Step 3: Training the model")
        print("=" * 60)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("  Model trained successfully.\n")

        # -- Step 4: Log evaluation metrics (bulk) ---------------------------
        print("=" * 60)
        print("Step 4: Logging evaluation metrics")
        print("=" * 60)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            print(f"  {name:>10s}: {value:.4f}")
        print()

        # -- Step 5: Log step-based metrics (simulated training loss) --------
        print("=" * 60)
        print("Step 5: Logging step-based metrics (simulated training loss)")
        print("=" * 60)
        rng = np.random.default_rng(random_state)
        loss = 1.0
        for step in range(10):
            loss *= 0.75 + 0.05 * rng.random()
            mlflow.log_metric("training_loss", loss, step=step)
            print(f"  Step {step:>2d}  loss={loss:.4f}")
        print()

        # -- Step 6: Log tags ------------------------------------------------
        print("=" * 60)
        print("Step 6: Setting tags")
        print("=" * 60)
        tags = {
            "model_type": "RandomForestClassifier",
            "dataset": "iris",
            "task_type": "classification",
        }
        mlflow.set_tags(tags)
        for k, v in tags.items():
            print(f"  {k}: {v}")
        print()

        # -- Step 7: Log an artifact (confusion matrix plot) -----------------
        print("=" * 60)
        print("Step 7: Logging artifact (confusion matrix plot)")
        print("=" * 60)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig.savefig(plot_path, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            print("  Saved and logged: confusion_matrix.png")
        plt.close(fig)
        print()

        # -- Done ------------------------------------------------------------
        print("=" * 60)
        print("Done! View results in the MLflow UI:")
        print("  http://127.0.0.1:5000/#/experiments")
        print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M1_core_platform/2_tracking_basics")
    train_model(n_estimators=100, max_depth=5, random_state=42)
