"""
L2-1.4 — MlflowClient: Programmatic Access

Demonstrates the MlflowClient low-level API for full CRUD control:
- Creating experiments and runs programmatically
- Logging params, metrics, and tags via the client API
- Querying experiments and runs with search/get methods
- Updating, deleting, and restoring runs
- Building a comparison report from queried data

Use MlflowClient when you need programmatic control beyond what the
fluent API (mlflow.log_param, mlflow.start_run, etc.) provides — e.g.,
managing runs across experiments, building dashboards, or scripting
batch operations on existing runs.
"""

import time

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "L2/M1_advanced_tracking/4_mlflow_client"


def train_and_log_model(
    client: MlflowClient,
    experiment_id: str,
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
) -> str:
    """Train a model and log everything using MlflowClient (not fluent API)."""
    run = client.create_run(experiment_id, run_name=model_name)
    run_id = run.info.run_id

    # Log model parameters
    params = model.get_params()
    for key, value in params.items():
        client.log_param(run_id, key, value)
    client.log_param(run_id, "model_type", type(model).__name__)

    # Train and time it
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    client.log_metric(run_id, "accuracy", acc)
    client.log_metric(run_id, "f1_score", f1)
    client.log_metric(run_id, "training_time", training_time)

    # Mark the run as finished
    client.update_run(run_id, status="FINISHED")

    print(f"  {model_name}: accuracy={acc:.4f}, f1={f1:.4f}, time={training_time:.4f}s")
    return run_id


def main() -> None:
    client = MlflowClient()

    # ================================================================
    # Part 1: Create experiment and runs using MlflowClient
    # ================================================================
    print("=" * 60)
    print("Part 1: Create experiment and runs via MlflowClient")
    print("=" * 60)

    # Create or get experiment — create_experiment raises if it already exists,
    # so we check first with get_experiment_by_name.
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment and experiment.lifecycle_stage == "active":
        experiment_id = experiment.experiment_id
        print(f"  Using existing experiment: {EXPERIMENT_NAME} (id={experiment_id})")
    else:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "mlflow-tutorial", "level": "2"},
        )
        print(f"  Created experiment: {EXPERIMENT_NAME} (id={experiment_id})")

    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42,
    )

    # Train 3 models, logging each with the client API
    models = [
        (LogisticRegression(max_iter=200, random_state=42), "logistic_regression"),
        (RandomForestClassifier(n_estimators=50, random_state=42), "random_forest"),
        (GradientBoostingClassifier(n_estimators=50, random_state=42), "gradient_boosting"),
    ]

    run_ids: list[str] = []
    print("\n  Training and logging 3 models...")
    for model, name in models:
        rid = train_and_log_model(
            client, experiment_id, model, name, X_train, X_test, y_train, y_test,
        )
        run_ids.append(rid)

    # ================================================================
    # Part 2: Query operations
    # ================================================================
    print("\n" + "=" * 60)
    print("Part 2: Query operations")
    print("=" * 60)

    # 2a. search_experiments — find our experiment by name filter
    experiments = client.search_experiments(
        filter_string=f"name = '{EXPERIMENT_NAME}'",
    )
    print(f"\n  search_experiments found {len(experiments)} matching experiment(s):")
    for exp in experiments:
        print(f"    id={exp.experiment_id}, name={exp.name}")

    # 2b. get_experiment / get_experiment_by_name
    exp_by_id = client.get_experiment(experiment_id)
    exp_by_name = client.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"\n  get_experiment(id={experiment_id}): name={exp_by_id.name}")
    print(f"  get_experiment_by_name('{EXPERIMENT_NAME}'): id={exp_by_name.experiment_id}")

    # 2c. search_runs — all runs in this experiment, ordered by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
    )
    print(f"\n  search_runs found {len(runs)} run(s) (ordered by accuracy DESC):")
    for r in runs:
        acc = r.data.metrics.get("accuracy", 0)
        print(f"    {r.info.run_name:<25s} accuracy={acc:.4f}  (id={r.info.run_id[:8]}...)")

    # 2d. search_runs with a filter — only runs with accuracy > 0.9
    good_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.accuracy > 0.9",
    )
    print(f"\n  Runs with accuracy > 0.9: {len(good_runs)}")
    for r in good_runs:
        print(f"    {r.info.run_name}: {r.data.metrics['accuracy']:.4f}")

    # 2e. get_run — detailed info for a single run
    detail_run = client.get_run(run_ids[0])
    print(f"\n  get_run({run_ids[0][:8]}...) details:")
    print(f"    name:   {detail_run.info.run_name}")
    print(f"    status: {detail_run.info.status}")
    print(f"    params: {len(detail_run.data.params)} logged")
    print(f"    metrics: {detail_run.data.metrics}")

    # ================================================================
    # Part 3: Update/manage operations
    # ================================================================
    print("\n" + "=" * 60)
    print("Part 3: Update and manage operations")
    print("=" * 60)

    # 3a. set_tag — add metadata tags to runs
    for rid in run_ids:
        client.set_tag(rid, "tutorial_lesson", "L2-M1.4")
        client.set_tag(rid, "dataset", "iris")
    print("\n  Added tags 'tutorial_lesson' and 'dataset' to all runs.")

    # 3b. update_run — rename a run
    old_name = client.get_run(run_ids[0]).info.run_name
    client.update_run(run_ids[0], name="lr_renamed")
    new_name = client.get_run(run_ids[0]).info.run_name
    print(f"  Renamed run: '{old_name}' -> '{new_name}'")

    # Rename it back for clarity in the comparison report
    client.update_run(run_ids[0], name=old_name)

    # 3c. delete_run and restore_run
    print(f"\n  Deleting run {run_ids[2][:8]}...")
    client.delete_run(run_ids[2])

    deleted_run = client.get_run(run_ids[2])
    print(f"    lifecycle_stage after delete: {deleted_run.info.lifecycle_stage}")

    # Verify it is excluded from active searches
    active_runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ALL,
    )
    print(f"    Active runs: {len(active_runs)}, All runs (incl. deleted): {len(all_runs)}")

    # Restore it
    client.restore_run(run_ids[2])
    restored = client.get_run(run_ids[2])
    print(f"    lifecycle_stage after restore: {restored.info.lifecycle_stage}")

    # ================================================================
    # Part 4: Build a comparison report
    # ================================================================
    print("\n" + "=" * 60)
    print("Part 4: Model comparison report")
    print("=" * 60)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.model_type != ''",
        order_by=["metrics.accuracy DESC"],
    )

    header = f"  {'Model':<28s} {'Accuracy':>10s} {'F1':>10s} {'Time (s)':>10s}"
    separator = f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}"
    print(f"\n{header}")
    print(separator)
    for r in runs:
        name = r.info.run_name or "unnamed"
        acc = r.data.metrics.get("accuracy", 0.0)
        f1 = r.data.metrics.get("f1_score", 0.0)
        t = r.data.metrics.get("training_time", 0.0)
        print(f"  {name:<28s} {acc:>10.4f} {f1:>10.4f} {t:>10.4f}")

    # Identify the best model
    if runs:
        best = runs[0]
        print(f"\n  Best model: {best.info.run_name} "
              f"(accuracy={best.data.metrics.get('accuracy', 0):.4f})")

    # ================================================================
    # Fluent API vs MlflowClient — when to use which
    # ================================================================
    print("\n" + "=" * 60)
    print("Fluent API vs MlflowClient")
    print("=" * 60)
    print("""
  Fluent API (mlflow.log_param, mlflow.start_run, etc.):
    - Simple, concise — great for interactive work and single-run scripts
    - Manages "active run" state automatically
    - Best for: notebooks, single experiments, quick prototyping

  MlflowClient:
    - Full CRUD control — create, read, update, delete any entity
    - No global state — you pass run_id explicitly
    - Can manage runs across experiments, rename/delete/restore runs
    - Best for: automation scripts, dashboards, CI/CD pipelines,
      batch operations, admin tools, multi-experiment workflows
""")

    print("=" * 60)
    print("Done! View results in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    main()
