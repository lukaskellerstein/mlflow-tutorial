"""
L2-M2.1 — Model Signatures Deep Dive

Explore MLflow model signatures in depth: column-based signatures,
tensor-based signatures, manual construction, inference-time params,
and signature enforcement during prediction.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L2/M2_advanced_models/1_signatures_deep_dive")


def part1_column_based_signature() -> str:
    """Infer a column-based signature from tabular training data."""
    print("=" * 60)
    print("Part 1: Column-Based Signature (inferred)")
    print("=" * 60)

    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_train)

    # infer_signature captures column names, types, and shapes
    signature = infer_signature(X_train, predictions)
    print(f"\n  Inferred signature:\n{signature}\n")
    print(f"  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")

    with mlflow.start_run(run_name="part1_column_based"):
        mlflow.log_param("signature_type", "column-based (inferred)")
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            name="iris_inferred_sig",
            signature=signature,
            input_example=X_train.head(3),
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model with inferred column signature. Run: {run_id}")

    return run_id


def part2_manual_signature() -> str:
    """Build a model signature manually with explicit column specs."""
    print("\n" + "=" * 60)
    print("Part 2: Manual Signature Construction")
    print("=" * 60)

    # Manually specify the schema for a house price model
    input_schema = Schema([
        ColSpec(DataType.double, "square_feet"),
        ColSpec(DataType.integer, "bedrooms"),
        ColSpec(DataType.integer, "bathrooms"),
        ColSpec(DataType.string, "neighborhood"),
        ColSpec(DataType.boolean, "has_garage"),
    ])
    output_schema = Schema([
        ColSpec(DataType.double, "predicted_price"),
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    print(f"\n  Manual signature:\n{signature}\n")
    print(f"  Input columns:")
    for col in input_schema.inputs:
        print(f"    - {col.name}: {col.type} (required={col.required})")
    print(f"\n  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")

    # Log a simple sklearn model with this manual signature
    iris = load_iris(as_frame=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(iris.data, iris.target)

    with mlflow.start_run(run_name="part2_manual_signature"):
        mlflow.log_param("signature_type", "column-based (manual)")
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="house_price_manual_sig",
            signature=signature,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model with manual column signature. Run: {run_id}")

    return run_id


def part3_tensor_signature() -> str:
    """Create a tensor-based signature for numpy array inputs/outputs."""
    print("\n" + "=" * 60)
    print("Part 3: Tensor-Based Signature")
    print("=" * 60)

    # Generate synthetic data shaped like image features
    np.random.seed(42)
    X = np.random.rand(100, 4).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    # Infer tensor signature from numpy arrays
    tensor_sig = infer_signature(X, preds)
    print(f"\n  Inferred tensor signature:\n{tensor_sig}\n")

    # Also build one manually with TensorSpec
    manual_input = Schema([
        TensorSpec(np.dtype("float64"), shape=(-1, 4), name="features"),
    ])
    manual_output = Schema([
        TensorSpec(np.dtype("int64"), shape=(-1,), name="predictions"),
    ])
    manual_tensor_sig = ModelSignature(inputs=manual_input, outputs=manual_output)
    print(f"  Manual tensor signature:\n{manual_tensor_sig}\n")
    print(f"  Signature JSON:\n{json.dumps(manual_tensor_sig.to_dict(), indent=2)}\n")

    with mlflow.start_run(run_name="part3_tensor_signature"):
        mlflow.log_param("signature_type", "tensor-based")
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="tensor_sig_model",
            signature=manual_tensor_sig,
            input_example=X[:2],
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model with tensor signature. Run: {run_id}")

    return run_id


def part4_signature_with_params() -> str:
    """Create a signature that includes inference-time parameters."""
    print("\n" + "=" * 60)
    print("Part 4: Signature with Inference Params")
    print("=" * 60)

    # Define input/output schemas
    input_schema = Schema([
        ColSpec(DataType.string, "question"),
    ])
    output_schema = Schema([
        ColSpec(DataType.string, "answer"),
    ])

    # Define inference-time parameters the model accepts
    param_schema = ParamSchema([
        ParamSpec("temperature", DataType.double, default=0.7),
        ParamSpec("max_tokens", DataType.long, default=256),
        ParamSpec("top_p", DataType.double, default=0.9),
        ParamSpec("stop_sequences", DataType.string, default=["###"], shape=(-1,)),
    ])

    signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
        params=param_schema,
    )

    print(f"\n  Signature with params:\n{signature}\n")
    print(f"  Parameters:")
    for p in param_schema.params:
        shape_info = f", shape={p.shape}" if p.shape else ""
        print(f"    - {p.name}: {p.dtype} (default={p.default}{shape_info})")
    print(f"\n  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")

    # Log a dummy model with this signature to show it in the UI
    iris = load_iris(as_frame=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(iris.data, iris.target)

    with mlflow.start_run(run_name="part4_params_signature"):
        mlflow.log_param("signature_type", "with inference params")
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="llm_params_sig_model",
            signature=signature,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model with param signature. Run: {run_id}")

    return run_id


def part5_signature_enforcement(run_id: str) -> None:
    """Demonstrate signature enforcement during prediction."""
    print("\n" + "=" * 60)
    print("Part 5: Signature Enforcement")
    print("=" * 60)

    # Load the model from Part 1 (column-based, inferred from Iris)
    model_uri = f"runs:/{run_id}/iris_inferred_sig"
    loaded = mlflow.pyfunc.load_model(model_uri)
    sig = loaded.metadata.signature

    print(f"\n  Loaded model signature:\n{sig}\n")

    # --- Correct input ---
    correct_input = pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
    })
    pred = loaded.predict(correct_input)
    print(f"  Correct input prediction: {pred}")

    # --- Wrong column names ---
    print("\n  Testing with wrong column names...")
    wrong_cols = pd.DataFrame({
        "feat_a": [5.1],
        "feat_b": [3.5],
        "feat_c": [1.4],
        "feat_d": [0.2],
    })
    try:
        loaded.predict(wrong_cols)
        print("  Result: prediction succeeded (schema may allow flexible columns)")
    except Exception as e:
        print(f"  Result: caught error - {type(e).__name__}: {e}")

    # --- Wrong number of columns ---
    print("\n  Testing with wrong number of columns...")
    too_few = pd.DataFrame({"sepal length (cm)": [5.1], "sepal width (cm)": [3.5]})
    try:
        loaded.predict(too_few)
        print("  Result: prediction succeeded (model handled missing columns)")
    except Exception as e:
        print(f"  Result: caught error - {type(e).__name__}: {e}")

    # --- Wrong data types ---
    print("\n  Testing with wrong data types (strings instead of floats)...")
    wrong_types = pd.DataFrame({
        "sepal length (cm)": ["not_a_number"],
        "sepal width (cm)": ["bad"],
        "petal length (cm)": ["data"],
        "petal width (cm)": ["here"],
    })
    try:
        loaded.predict(wrong_types)
        print("  Result: prediction succeeded (types were coerced)")
    except Exception as e:
        print(f"  Result: caught error - {type(e).__name__}: {e}")

    print()


def main() -> None:
    print("=" * 60)
    print("L2-M2.1 — Model Signatures Deep Dive")
    print("=" * 60)
    print()

    run_id_part1 = part1_column_based_signature()
    part2_manual_signature()
    part3_tensor_signature()
    part4_signature_with_params()
    part5_signature_enforcement(run_id_part1)

    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("Explore each run's model artifacts to see the signatures")
    print("in the MLmodel file and the signature tab.")
    print("=" * 60)


if __name__ == "__main__":
    main()
