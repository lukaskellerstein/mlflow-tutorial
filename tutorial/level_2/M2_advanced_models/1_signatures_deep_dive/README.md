# L2-M2.1 — Model Signatures Deep Dive

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

Model signatures are the contract between a model and its consumers. They
describe the exact shape, types, and names of inputs, outputs, and
inference-time parameters. In this lesson you will learn how to infer
signatures automatically, construct them manually, use tensor-based schemas,
attach inference parameters, and see how MLflow enforces signatures at
prediction time.

## Prerequisites

- Completed: L1-M2.1 (Models and Flavors — basic signature usage)
- MLflow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (we use scikit-learn)

## Concepts

### Signature Types

MLflow supports two fundamental signature types:

| Type | Use Case | Spec Class |
|------|----------|------------|
| **Column-based** | Tabular / DataFrame data | `ColSpec` |
| **Tensor-based** | NumPy arrays, images, embeddings | `TensorSpec` |

You cannot mix `ColSpec` and `TensorSpec` in the same schema.

### Column-Based Signatures

Column-based signatures describe named columns with data types drawn from
`mlflow.types.DataType`: `boolean`, `integer`, `long`, `float`, `double`,
`string`, `binary`, `datetime`.

```python
from mlflow.types import Schema, ColSpec, DataType

schema = Schema([
    ColSpec(DataType.double, "price"),
    ColSpec(DataType.string, "category"),
])
```

### Tensor-Based Signatures

Tensor-based signatures describe multi-dimensional arrays using numpy dtypes
and shape tuples. Use `-1` for the batch dimension.

```python
from mlflow.types import Schema, TensorSpec

schema = Schema([
    TensorSpec(np.dtype("float32"), shape=(-1, 768), name="embeddings"),
])
```

### Inference Parameters (ParamSpec)

Starting with MLflow 2.6, signatures can include parameter specifications.
These describe runtime configuration that consumers can pass at inference
time — like `temperature`, `max_tokens`, or `top_p` for LLM models.

```python
from mlflow.types import ParamSchema, ParamSpec, DataType

params = ParamSchema([
    ParamSpec("temperature", DataType.double, default=0.7),
    ParamSpec("max_tokens", DataType.long, default=256),
])
```

### Signature Enforcement

When you load a model via `mlflow.pyfunc.load_model()` and call `.predict()`,
MLflow validates the input against the stored signature. It will:

- Warn or error on missing required columns
- Warn on extra columns
- Attempt type coercion (e.g., int to float) where possible
- Raise errors when types are incompatible

This catch-errors-early behavior is especially valuable in production serving.

## Step-by-Step

### Part 1: Column-Based Signature (Inferred)

We train a RandomForest on Iris and let `infer_signature()` capture column
names and types automatically from the DataFrame.

```python
signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(clf, name="model", signature=signature)
```

### Part 2: Manual Signature Construction

We build a signature by hand using `Schema`, `ColSpec`, and `DataType` for
a hypothetical house-price model with mixed types (double, integer, string,
boolean).

```python
input_schema = Schema([
    ColSpec(DataType.double, "square_feet"),
    ColSpec(DataType.integer, "bedrooms"),
    ColSpec(DataType.string, "neighborhood"),
    ColSpec(DataType.boolean, "has_garage"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

### Part 3: Tensor-Based Signature

We create a model that takes numpy arrays and build a tensor signature with
`TensorSpec`, specifying dtype and shape (using `-1` for the batch dimension).

```python
Schema([TensorSpec(np.dtype("float64"), shape=(-1, 4), name="features")])
```

### Part 4: Signature with Inference Params

We attach `ParamSpec` entries to describe inference-time parameters. This
is the pattern used for LLM models where consumers need to control
temperature, max tokens, stop sequences, etc.

```python
ParamSchema([
    ParamSpec("temperature", DataType.double, default=0.7),
    ParamSpec("max_tokens", DataType.long, default=256),
    ParamSpec("stop_sequences", DataType.string, default=["###"], shape=(-1,)),
])
```

### Part 5: Signature Enforcement

We load the Part 1 model via `pyfunc` and test prediction with:
- Correct input (succeeds)
- Wrong column names (may warn or error)
- Too few columns (may error)
- Wrong data types (may error or coerce)

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/1_signatures_deep_dive
uv sync
uv run python main.py
```

## Expected Output

You should see five sections, each printing:
- The signature in human-readable form
- The signature as JSON (showing the internal representation)
- Confirmation that the model was logged

For Part 5 (enforcement), you will see whether MLflow accepts or rejects
each malformed input. The exact behavior depends on your MLflow version —
newer versions are stricter.

In the MLflow UI at http://127.0.0.1:5000:
- Open each run under the `L2/M2_advanced_models/1_signatures_deep_dive` experiment
- Click into the model artifact and inspect the **MLmodel** file
- The signature section shows inputs, outputs, and params schemas

## Key Takeaways

- **Column-based** signatures are for tabular DataFrames; **tensor-based** are for numpy arrays.
- Use `infer_signature()` for quick automatic signatures, or build them manually with `Schema`, `ColSpec`, and `TensorSpec` for precise control.
- `ParamSpec` lets you declare inference-time parameters (temperature, max_tokens, etc.) as part of the model contract.
- Signature enforcement catches schema mismatches at prediction time — a safety net for production serving.
- Always include a signature when logging models; it serves as documentation, validation, and REST API schema generation.

## Next Steps

In the next lesson (L2-M2.2 — Custom PyFunc Models) you will learn how to
build fully custom models using `mlflow.pyfunc.PythonModel`, including
loading external artifacts, custom predict logic, and advanced signatures
for non-standard model types.
