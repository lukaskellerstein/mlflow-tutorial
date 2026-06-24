"""
L1-8.1 -- Model Serving Basics

Trains a model, registers it, and demonstrates serving commands,
programmatic prediction, and batch prediction workflows.
"""

import json
import os
import tempfile

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_NAME = "iris-classifier-serving-demo"
FEATURE_COLS = [f"feature_{i}" for i in range(4)]


def part1_prepare_model(X_train, X_test, y_train, y_test) -> None:
    """Train, log, and register a model for serving."""
    print("=" * 60)
    print("Part 1: Prepare a Model for Serving")
    print("=" * 60)
    with mlflow.start_run(run_name="train_for_serving") as run:
        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        mlflow.log_params({"n_estimators": 50, "max_depth": 4})
        mlflow.log_metric("accuracy", accuracy)
        # Signature and input example define the serving request/response format.
        signature = infer_signature(X_test, clf.predict(X_test))
        input_example = pd.DataFrame(X_test[:2], columns=FEATURE_COLS)
        mlflow.sklearn.log_model(
            clf, name="model", signature=signature,
            input_example=input_example, registered_model_name=MODEL_NAME,
        )
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Model URI: runs:/{run.info.run_id}/model")
        print(f"  Registered as: {MODEL_NAME}")


def part2_serving_commands() -> None:
    """Print CLI commands and endpoint details for model serving."""
    print("\n" + "=" * 60)
    print("Part 2: Serving Commands and Endpoints")
    print("=" * 60)
    print(f'\n  Serve locally:')
    print(f'    mlflow models serve -m "models:/{MODEL_NAME}/1" --port 5001 --no-conda')
    print("\n  Endpoints:")
    print("    POST /invocations  -- run predictions")
    print("    GET  /ping         -- health check (returns 200 OK)")
    print("    GET  /version      -- MLflow version info")
    sample = {"dataframe_split": {
        "columns": FEATURE_COLS,
        "data": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]],
    }}
    print("\n  Prediction (dataframe_split format):")
    print("    curl -X POST http://127.0.0.1:5001/invocations \\")
    print('      -H "Content-Type: application/json" \\')
    print(f"      -d '{json.dumps(sample)}'")
    alt = {"instances": [dict(zip(FEATURE_COLS, [5.1, 3.5, 1.4, 0.2]))]}
    print(f"\n  Alternative (instances format):")
    print(f"    -d '{json.dumps(alt)}'")


def part3_programmatic_prediction(X_test) -> None:
    """Load the model and run predictions without serving."""
    print("\n" + "=" * 60)
    print("Part 3: Programmatic Prediction (No Server Needed)")
    print("=" * 60)
    model_uri = f"models:/{MODEL_NAME}/1"
    print(f"  Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    sample = pd.DataFrame(X_test[:3], columns=FEATURE_COLS)
    predictions = model.predict(sample)
    print(f"  Input shape: {sample.shape}")
    print(f"  Predictions: {predictions.tolist()}")
    print("\n  Serving  -- REST API, language-agnostic, production use")
    print("  Loading  -- in-process, Python only, scripts and notebooks")


def part4_batch_prediction(X_test) -> None:
    """Demonstrate batch prediction workflow."""
    print("\n" + "=" * 60)
    print("Part 4: Batch Prediction and Docker")
    print("=" * 60)
    sample_df = pd.DataFrame(X_test[:5], columns=FEATURE_COLS)
    csv_path = os.path.join(tempfile.gettempdir(), "iris_batch_input.csv")
    sample_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="batch_inputs")
    print(f"  Sample input CSV: {csv_path}")
    print(f'\n  Batch predict:')
    print(f'    mlflow models predict -m "models:/{MODEL_NAME}/1" -i {csv_path}')
    print(f'\n  Docker containerization:')
    print(f'    mlflow models build-docker -m "models:/{MODEL_NAME}/1" -n "iris-server"')
    print(f'    docker run -p 5001:8080 iris-server')


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M8_deployment/1_model_serving")

    print("=" * 60)
    print("Loading Iris dataset")
    print("=" * 60)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    print(f"  Training: {len(X_train)} samples | Test: {len(X_test)} samples")

    part1_prepare_model(X_train, X_test, y_train, y_test)
    part2_serving_commands()

    with mlflow.start_run(run_name="batch_and_programmatic"):
        part3_programmatic_prediction(X_test)
        part4_batch_prediction(X_test)

    print("\n" + "=" * 60)
    print("Done! Try serving with the command from Part 2.")
    print("View runs in MLflow UI: http://127.0.0.1:5000")
    print("=" * 60)
