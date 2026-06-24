"""
L1-7.1 — Dataset Logging and Lineage

Demonstrates MLflow's dataset tracking capabilities:
- Creating MLflow datasets from pandas DataFrames
- Logging datasets with context (training / validation)
- Querying dataset info from completed runs
- Understanding data lineage — linking data to models
"""

import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # Step 1: Prepare data
    print("=" * 60)
    print("Step 1: Loading the Wine dataset")
    print("=" * 60)
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Total: {len(df)}  Train: {len(train_df)}  Val: {len(val_df)}\n")

    # Step 2: Create MLflow datasets
    print("=" * 60)
    print("Step 2: Creating MLflow datasets from DataFrames")
    print("=" * 60)
    train_dataset = mlflow.data.from_pandas(
        train_df, source="sklearn.datasets.load_wine", targets="target", name="wine_train"
    )
    val_dataset = mlflow.data.from_pandas(
        val_df, source="sklearn.datasets.load_wine", targets="target", name="wine_val"
    )
    print(f"  Train: name={train_dataset.name}, digest={train_dataset.digest}")
    print(f"  Val:   name={val_dataset.name}, digest={val_dataset.digest}")
    print(f"  Schema: {train_dataset.schema}\n")

    # Step 3: Train model and log datasets with lineage
    print("=" * 60)
    print("Step 3: Logging datasets and model in an MLflow run")
    print("=" * 60)
    with mlflow.start_run(run_name="wine_with_dataset_lineage") as run:
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")
        print("  Logged training dataset (context='training')")
        print("  Logged validation dataset (context='validation')")

        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_val, y_val = val_df.drop(columns=["target"]), val_df["target"]
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, clf.predict(X_val))

        mlflow.log_params({"n_estimators": 50, "max_depth": 5})
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.sklearn.log_model(clf, name="model")
        print(f"  Validation accuracy: {accuracy:.4f}")
        print(f"  Run ID: {run.info.run_id}\n")

    # Step 4: Query dataset lineage from the run
    print("=" * 60)
    print("Step 4: Querying dataset lineage from the completed run")
    print("=" * 60)
    run_data = mlflow.get_run(run.info.run_id)
    for ds_input in run_data.inputs.dataset_inputs:
        ds = ds_input.dataset
        ctx = {t.key: t.value for t in ds_input.tags}.get("mlflow.data.context", "N/A")
        print(f"  Dataset: {ds.name}  Digest: {ds.digest}  Context: {ctx}")
        print(f"    Source: {ds.source}")

    print("\n" + "=" * 60)
    print("Done! View dataset lineage in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M7_data_datasets/1_dataset_logging")
    main()
