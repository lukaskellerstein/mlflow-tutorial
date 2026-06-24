# L1-M2.1 — MLflow Models and Flavors

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Learn how MLflow packages trained models into a portable, self-describing
format.  You will understand what *flavors* are, how *signatures* document
input/output schemas, and how *input examples* make models self-documenting.

## Prerequisites

- Completed: L1-M1 (Core Platform — tracking basics)
- MLflow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (we use scikit-learn)

## Concepts

### What is an MLflow Model?

An MLflow Model is a standard directory layout that stores everything needed
to reproduce and serve a model:

```
iris_model/
  MLmodel              # YAML manifest — lists flavors, signature, etc.
  model.pkl            # Serialized model artifact
  conda.yaml           # Conda environment spec
  requirements.txt     # pip dependencies
  input_example.json   # Sample input (optional)
```

The `MLmodel` file is the key — it tells MLflow *how* to load the model.

### What are Flavors?

A **flavor** is a named interface through which a model can be saved and
loaded.  Every model gets the generic `python_function` (pyfunc) flavor so
it can always be loaded and served the same way.  Framework-specific flavors
give you access to the native model object.

Built-in flavors include:

| Flavor | Framework |
|--------|-----------|
| `sklearn` | scikit-learn |
| `pytorch` | PyTorch |
| `transformers` | Hugging Face Transformers |
| `langchain` | LangChain / LangGraph |
| `openai` | OpenAI API |
| `tensorflow` | TensorFlow / Keras |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `spark` | PySpark |
| `onnx` | ONNX Runtime |
| `pyfunc` | Any Python code (custom models) |

### Model Signatures

A `ModelSignature` records column names, data types, and shapes for inputs
and outputs.  MLflow uses signatures to:

- **Validate** data before inference — catch schema errors early.
- **Generate** REST API documentation when serving.
- **Display** schema in the MLflow UI.

You create a signature with `mlflow.models.infer_signature(inputs, outputs)`.

### Input Examples

An *input example* is a small sample saved alongside the model.  It serves
as living documentation — anyone can look at the model artifact and
immediately see what data the model expects.

## Step-by-Step

### Step 1: Prepare data

We load the Iris dataset and split it into training and test sets.

```python
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
```

### Step 2: Train and log the model

Inside an MLflow run we train a `RandomForestClassifier`, infer the model
signature, and log the model with both the signature and an input example.

```python
signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(
    sk_model=clf,
    artifact_path="iris_model",
    signature=signature,
    input_example=X_train.head(3),
)
```

### Step 3: Load and predict

We load the model back using its run artifact URI and run predictions on the
test set to verify everything round-trips correctly.

```python
model_uri = f"runs:/{run_id}/iris_model"
loaded_model = mlflow.sklearn.load_model(model_uri)
preds = loaded_model.predict(X_test)
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_models_registry/1_models_flavors
uv sync
uv run python main.py
```

## Expected Output

You should see:
- Dataset statistics (120 training / 30 test samples)
- Training accuracy (~1.0 on Iris)
- The inferred model signature showing feature columns and output type
- A table of predicted vs. actual class names
- Overall accuracy on the test set

In the MLflow UI at http://127.0.0.1:5000 you can:
- Open the run and inspect the **Artifacts** tab
- Click into `iris_model/` to see the `MLmodel` file, signature, and input example

## Key Takeaways

- An MLflow Model is a portable directory with an `MLmodel` manifest.
- **Flavors** let the same model be loaded natively or through the generic pyfunc interface.
- **Signatures** document and enforce the expected input/output schema.
- **Input examples** make models self-documenting — always include one.
- Use `infer_signature()` to auto-generate signatures from training data.

## Next Steps

In the next lesson (L1-M2.2 — Model Registry) you will learn how to
register models, manage versions, and assign lifecycle aliases like
`champion` and `challenger`.
