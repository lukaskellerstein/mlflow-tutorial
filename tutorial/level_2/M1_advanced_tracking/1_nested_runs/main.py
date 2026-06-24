"""
L2-1.1 — Nested Runs and Run Hierarchies

Demonstrates how to use nested runs for hyperparameter grid search:
- A parent run groups the entire sweep
- Each child run (nested=True) logs one configuration
- Three models x two hyperparameter values = six nested runs
- Parent run records summary: best config, best accuracy
- search_runs() retrieves and ranks all child runs
"""

import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Grid definition: 3 models x 2 max_depth values = 6 combinations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "static_params": {"n_estimators": 100, "random_state": 42},
    },
    {
        "name": "GradientBoosting",
        "class": GradientBoostingClassifier,
        "static_params": {"n_estimators": 100, "random_state": 42},
    },
    {
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "static_params": {"max_iter": 5000, "random_state": 42},
    },
]

MAX_DEPTH_VALUES = [3, 7]


def load_data() -> tuple:
    """Load the Wine dataset and return train/test splits."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, wine.feature_names


def train_and_log_child(
    model_cfg: dict,
    max_depth: int | None,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    """Train one model configuration inside a nested child run."""

    model_name = model_cfg["name"]
    run_label = f"{model_name}_depth_{max_depth}"

    with mlflow.start_run(run_name=run_label, nested=True) as child_run:
        # -- Build params dict -----------------------------------------------
        params = dict(model_cfg["static_params"])
        params["model_type"] = model_name

        # LogisticRegression does not accept max_depth
        if max_depth is not None and model_name != "LogisticRegression":
            params["max_depth"] = max_depth

        mlflow.log_params(params)
        mlflow.set_tags({
            "model_family": model_name,
            "sweep_param": "max_depth",
            "sweep_value": str(max_depth),
        })

        # -- Train -----------------------------------------------------------
        constructor_params = dict(model_cfg["static_params"])
        if model_name != "LogisticRegression" and max_depth is not None:
            constructor_params["max_depth"] = max_depth

        model = model_cfg["class"](**constructor_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -- Evaluate --------------------------------------------------------
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)

        # -- Log model -------------------------------------------------------
        mlflow.sklearn.log_model(model, name="model")

        print(f"  {run_label:40s} accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

        return {
            "run_id": child_run.info.run_id,
            "run_name": run_label,
            "model": model_name,
            "max_depth": max_depth,
            **metrics,
        }


def run_grid_search() -> None:
    """Execute the full nested-run hyperparameter sweep."""

    X_train, X_test, y_train, y_test, feature_names = load_data()

    print("=" * 60)
    print("Step 1: Loading the Wine dataset")
    print("=" * 60)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Features:         {len(feature_names)}")
    print()

    # -- Parent run ----------------------------------------------------------
    print("=" * 60)
    print("Step 2: Running hyperparameter grid search (nested runs)")
    print("=" * 60)
    results: list[dict] = []

    with mlflow.start_run(run_name="hyperparameter_sweep") as parent_run:
        mlflow.set_tags({
            "sweep_type": "grid_search",
            "dataset": "wine",
            "num_configs": str(len(MODEL_CONFIGS) * len(MAX_DEPTH_VALUES)),
        })

        for model_cfg in MODEL_CONFIGS:
            for max_depth in MAX_DEPTH_VALUES:
                result = train_and_log_child(
                    model_cfg, max_depth, X_train, X_test, y_train, y_test
                )
                results.append(result)

        # -- Parent summary --------------------------------------------------
        print()
        print("=" * 60)
        print("Step 3: Logging parent-run summary")
        print("=" * 60)

        best = max(results, key=lambda r: r["accuracy"])
        mlflow.log_params({
            "best_model": best["model"],
            "best_max_depth": best["max_depth"],
        })
        mlflow.log_metrics({
            "best_accuracy": best["accuracy"],
            "best_f1": best["f1"],
        })
        mlflow.set_tag("best_child_run_id", best["run_id"])

        print(f"  Best config:   {best['run_name']}")
        print(f"  Best accuracy: {best['accuracy']:.4f}")
        print(f"  Best F1:       {best['f1']:.4f}")
        print(f"  Parent run ID: {parent_run.info.run_id}")
        print()

    # -- Query nested runs with search_runs() --------------------------------
    print("=" * 60)
    print("Step 4: Querying child runs with search_runs()")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name(
        "L2/M1_advanced_tracking/1_nested_runs"
    )
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        order_by=["metrics.accuracy DESC"],
    )

    # Build a clean summary table
    summary_cols = ["run_id", "tags.model_family", "params.max_depth", "metrics.accuracy", "metrics.f1"]
    available_cols = [c for c in summary_cols if c in child_runs.columns]
    summary = child_runs[available_cols].copy()
    summary.columns = [c.split(".")[-1] for c in available_cols]

    print()
    print(summary.to_string(index=False))
    print()

    print("=" * 60)
    print("Done! View the nested run hierarchy in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("  Expand the 'hyperparameter_sweep' parent run to see children.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M1_advanced_tracking/1_nested_runs")
    run_grid_search()
