"""
L1-M2.1 — MLflow Models and Flavors

Learn how MLflow packages models, what flavors are, how signatures
document the expected input/output schema, and how input examples
make models self-documenting.
"""

import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/1_models_flavors")


def main() -> None:
    # ------------------------------------------------------------------
    # Concepts
    # ------------------------------------------------------------------
    print("=" * 60)
    print("L1-M2.1 — MLflow Models and Flavors")
    print("=" * 60)

    print("""
WHAT IS AN MLFLOW MODEL?
  An MLflow Model is a directory containing:
  - MLmodel        YAML manifest listing available flavors
  - model.pkl      (or equivalent) serialized model artifact
  - conda.yaml     Conda environment specification
  - requirements.txt  pip dependencies
  - input_example.json  (optional) sample input for documentation

WHAT ARE FLAVORS?
  Flavors are named interfaces a model can be loaded through.
  Every model has the generic 'python_function' (pyfunc) flavor
  so it can always be loaded and served uniformly.
  Framework-specific flavors provide native access.

  Built-in flavors include:
    sklearn, pytorch, transformers, langchain, openai,
    tensorflow, xgboost, lightgbm, spark, onnx, pyfunc ...

WHY SIGNATURES?
  A ModelSignature records column names, types, and shapes of
  inputs and outputs.  MLflow uses it to:
    - Validate data before inference (catch schema errors early)
    - Generate REST API docs when serving
    - Display schema in the MLflow UI

INPUT EXAMPLES
  An input_example is a small sample saved alongside the model.
  It serves as living documentation and is used by MLflow to
  validate that the model can be loaded and called successfully.
""")

    # ------------------------------------------------------------------
    # Step 1 — Prepare data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Preparing Iris dataset")
    print("=" * 60)

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Features         : {list(X.columns)}")

    # ------------------------------------------------------------------
    # Step 2 — Train and log the model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Training model and logging to MLflow")
    print("=" * 60)

    with mlflow.start_run(run_name="sklearn_iris_rf") as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        print(f"  Training accuracy on test set: {accuracy:.4f}")

        # Infer signature from training data and predictions
        predictions = clf.predict(X_train)
        signature = infer_signature(X_train, predictions)
        print(f"\n  Model signature:\n{signature}\n")

        # Log the model with signature and input example
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="iris_model",
            signature=signature,
            input_example=input_example,
        )
        print("  Model logged with signature and input example.")
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")

    # ------------------------------------------------------------------
    # Step 3 — Load the model back and predict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Loading model and running predictions")
    print("=" * 60)

    model_uri = f"runs:/{run_id}/iris_model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"  Loaded model from: {model_uri}")

    test_preds = loaded_model.predict(X_test)
    target_names = iris.target_names

    results = pd.DataFrame({
        "predicted": [target_names[p] for p in test_preds],
        "actual": [target_names[a] for a in y_test],
    })
    results["correct"] = results["predicted"] == results["actual"]

    print(f"\n  Predictions (first 10 of {len(results)}):")
    print(results.head(10).to_string(index=False))
    print(f"\n  Overall accuracy: {results['correct'].mean():.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done!  Check the MLflow UI at http://127.0.0.1:5000")
    print("Navigate to the run to inspect the logged model artifact,")
    print("its MLmodel file, signature, and input example.")
    print("=" * 60)


if __name__ == "__main__":
    main()
