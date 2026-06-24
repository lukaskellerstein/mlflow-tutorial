"""
L1-M1.3 — Search and Query API

Demonstrates MLflow's search and query capabilities:
- mlflow.search_runs() with various filters
- mlflow.search_experiments() to list experiments
- MlflowClient for programmatic access
- Exporting results to pandas DataFrames
"""

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_core_platform/3_search_query_api"
COLS = ["run_id", "params.model_type", "metrics.accuracy"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def create_sample_runs() -> str:
    """Train several models on the Wine dataset and log each as an MLflow run."""
    section("Step 1: Creating sample runs with different models")

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=7)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    models = [
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=5, random_state=42)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=80, random_state=42)),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("svm", SVC(kernel="linear", random_state=42)),
        ("decision_tree", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ]

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    for model_type, model in models:
        with mlflow.start_run(run_name=f"{model_type}_{model.__class__.__name__}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dataset", "wine")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", round(acc, 4))
            mlflow.log_metric("f1_score", round(f1, 4))
            mlflow.set_tag("lesson", "L1-M1.3")
            print(f"  Logged {model_type:25s}  accuracy={acc:.4f}  f1={f1:.4f}")

    return experiment.experiment_id


def demo_search_runs(experiment_id: str) -> None:
    """Show various search_runs() queries."""

    # All runs (no filter)
    section("Step 2: search_runs -- all runs (no filter)")
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    print(f"  Total runs found: {len(all_runs)}")
    print(all_runs[COLS].to_string(index=False))

    # Filter by metric
    section("Step 3: search_runs -- metrics.accuracy > 0.9")
    high_acc = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string="metrics.accuracy > 0.9",
    )
    print(f"  Runs with accuracy > 0.9: {len(high_acc)}")
    if not high_acc.empty:
        print(high_acc[COLS].to_string(index=False))

    # Filter by param
    section("Step 4: search_runs -- params.model_type = 'random_forest'")
    rf_runs = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string="params.model_type = 'random_forest'",
    )
    print(f"  Random forest runs: {len(rf_runs)}")
    if not rf_runs.empty:
        print(rf_runs[COLS].to_string(index=False))

    # Order by metric
    section("Step 5: search_runs -- order by accuracy DESC")
    ordered = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"],
    )
    print("  Runs ranked by accuracy (best first):")
    print(ordered[COLS].to_string(index=False))

    # Combined filters
    section("Step 6: Combined filter -- accuracy > 0.8 AND random_forest")
    combined = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.accuracy > 0.8 AND params.model_type = 'random_forest'",
    )
    print(f"  Matching runs: {len(combined)}")
    if not combined.empty:
        print(combined[COLS + ["metrics.f1_score"]].to_string(index=False))


def demo_search_experiments() -> None:
    """List experiments on the tracking server."""
    section("Step 7: search_experiments -- list all experiments")
    experiments = mlflow.search_experiments()
    print(f"  Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"    [{exp.experiment_id}] {exp.name}")


def demo_mlflow_client(experiment_id: str) -> None:
    """Use MlflowClient for programmatic access."""
    section("Step 8: MlflowClient -- programmatic access")
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment(experiment_id)
    print(f"  Experiment name : {experiment.name}")
    print(f"  Experiment ID   : {experiment.experiment_id}")
    print(f"  Artifact location: {experiment.artifact_location}\n")

    best_runs = client.search_runs(
        experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"], max_results=1,
    )
    if best_runs:
        best = best_runs[0]
        print(f"  Best run ID     : {best.info.run_id}")
        print(f"  Best run name   : {best.info.run_name}")
        print(f"  Model type      : {best.data.params.get('model_type')}")
        print(f"  Accuracy        : {best.data.metrics.get('accuracy')}")
        print(f"  F1 score        : {best.data.metrics.get('f1_score')}")
        print(f"  Status          : {best.info.status}")


def demo_dataframe_export(experiment_id: str) -> None:
    """Export search results to a pandas DataFrame and summarize."""
    section("Step 9: DataFrame export -- summary statistics")
    df = mlflow.search_runs(experiment_ids=[experiment_id])

    summary = (
        df.groupby("params.model_type")["metrics.accuracy"]
        .agg(["count", "mean", "max"])
        .rename(columns={"count": "runs", "mean": "avg_accuracy", "max": "best_accuracy"})
        .sort_values("best_accuracy", ascending=False)
    )
    print("  Accuracy summary by model type:")
    print(summary.to_string())

    best_row = df.loc[df["metrics.accuracy"].idxmax()]
    print(f"\n  Overall best run: {best_row['run_id']}")
    print(f"    model_type = {best_row['params.model_type']}")
    print(f"    accuracy   = {best_row['metrics.accuracy']}")
    print(f"    f1_score   = {best_row['metrics.f1_score']}")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_id = create_sample_runs()
    demo_search_runs(experiment_id)
    demo_search_experiments()
    demo_mlflow_client(experiment_id)
    demo_dataframe_export(experiment_id)
    section(f"Done! Open MLflow UI at {TRACKING_URI}")
    print(f"Look for experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
