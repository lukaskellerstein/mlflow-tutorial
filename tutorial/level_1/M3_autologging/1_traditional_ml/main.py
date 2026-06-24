"""
L1-3.1 — Traditional ML Autologging

Demonstrates MLflow's autologging for traditional ML frameworks:
- sklearn autologging (params, metrics, model, artifacts)
- XGBoost autologging (params, metrics, model, feature importance)
- Universal autolog (enables all frameworks at once)
"""

import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def print_run_info(run_id: str, label: str) -> None:
    """Query a completed run and print what was auto-logged."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    print(f"\n{'- ' * 30}")
    print(f"  Auto-logged details for: {label}")
    print(f"  Run ID: {run_id}")
    print(f"{'- ' * 30}")

    # Parameters
    params = run.data.params
    print(f"\n  Parameters ({len(params)}):")
    for k, v in sorted(params.items()):
        print(f"    {k}: {v}")

    # Metrics
    metrics = run.data.metrics
    print(f"\n  Metrics ({len(metrics)}):")
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.4f}")

    # Artifacts
    artifacts = client.list_artifacts(run_id)
    print(f"\n  Artifacts ({len(artifacts)}):")
    for a in artifacts:
        print(f"    {a.path} {'(dir)' if a.is_dir else ''}")


def part1_sklearn_autolog(X_train, X_test, y_train, y_test) -> None:
    """Part 1: sklearn autologging."""
    print("=" * 60)
    print("Part 1: sklearn Autologging")
    print("=" * 60)
    print("  Enabling mlflow.sklearn.autolog() ...")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="sklearn_autolog") as run:
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42
        )
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"  RandomForest accuracy: {score:.4f}")

    print_run_info(run.info.run_id, "sklearn RandomForest")
    mlflow.sklearn.autolog(disable=True)


def part2_xgboost_autolog(X_train, X_test, y_train, y_test) -> None:
    """Part 2: XGBoost autologging."""
    print("\n" + "=" * 60)
    print("Part 2: XGBoost Autologging")
    print("=" * 60)
    print("  Enabling mlflow.xgboost.autolog() ...")

    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="xgboost_autolog") as run:
        xgb = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train)
        score = xgb.score(X_test, y_test)
        print(f"  XGBoost accuracy: {score:.4f}")

    print_run_info(run.info.run_id, "XGBoost XGBClassifier")
    mlflow.xgboost.autolog(disable=True)


def part3_universal_autolog(X_train, X_test, y_train, y_test) -> None:
    """Part 3: Universal autolog — enables all frameworks at once."""
    print("\n" + "=" * 60)
    print("Part 3: Universal Autolog (mlflow.autolog)")
    print("=" * 60)
    print("  Enabling mlflow.autolog() — covers ALL supported frameworks ...")

    # Re-enable sklearn (was disabled in Part 1) so autolog() takes effect.
    # In practice you'd call mlflow.autolog() at the start of your script
    # and never need to toggle individual frameworks.
    mlflow.sklearn.autolog(disable=False)
    mlflow.autolog()

    with mlflow.start_run(run_name="universal_autolog") as run:
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X_train, y_train)
        score = lr.score(X_test, y_test)
        print(f"  LogisticRegression accuracy: {score:.4f}")

    print_run_info(run.info.run_id, "Universal autolog — LogisticRegression")
    mlflow.autolog(disable=True)


def compare_results() -> None:
    """Print a summary comparing what each framework auto-logs."""
    print("\n" + "=" * 60)
    print("Comparison: What Each Framework Auto-Logs")
    print("=" * 60)
    print("""
  sklearn:  all constructor params, training metrics (accuracy,
            f1, precision, recall, roc_auc, log_loss),
            serialized model, confusion matrix plot

  XGBoost:  all constructor params, test score,
            serialized model, feature importance (plot + JSON)

  TIP: mlflow.autolog() enables ALL frameworks at once —
  the easiest way to get started.
""")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M3_autologging/1_traditional_ml")

    # Load data once, reuse across all parts
    print("=" * 60)
    print("Loading Iris dataset")
    print("=" * 60)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    part1_sklearn_autolog(X_train, X_test, y_train, y_test)
    part2_xgboost_autolog(X_train, X_test, y_train, y_test)
    part3_universal_autolog(X_train, X_test, y_train, y_test)
    compare_results()

    print("=" * 60)
    print("Done! View all three runs in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)
