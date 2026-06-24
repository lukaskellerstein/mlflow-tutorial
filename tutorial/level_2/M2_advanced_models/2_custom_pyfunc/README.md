# L2-M2.2 — Custom PyFunc Models

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

MLflow's `PythonModel` base class lets you wrap arbitrary prediction logic as a first-class MLflow model. This lesson explores three advanced patterns: loading external artifacts at model-load time with `load_context()`, accepting runtime parameters via `params`, and packaging a multi-model ensemble as a single deployable PyFunc.

## Prerequisites

- Completed: L1-M2.3 (PyFunc basics)
- MLFlow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why Custom PyFunc?

MLflow's built-in flavors (sklearn, pytorch, etc.) cover common cases, but real-world deployments often need custom logic:

- **Pre/post-processing** baked into the model artifact
- **Loading external resources** (files, databases, other models) at startup
- **Runtime configurability** — callers pass parameters that change behavior without reloading the model
- **Ensemble or composite models** — multiple models behind a single prediction API

All of these are handled by subclassing `mlflow.pyfunc.PythonModel`.

### Key Methods

| Method | Purpose |
|---|---|
| `load_context(self, context)` | Called once when the model is loaded. Use it to load heavy artifacts (models, lookup tables, tokenizers) from `context.artifacts`. |
| `predict(self, context, model_input, params=None)` | Called on every prediction request. `model_input` is a DataFrame or dict; `params` is an optional dict of runtime overrides. |

### The artifacts Dict

When you call `mlflow.pyfunc.log_model(..., artifacts={"name": "/path/to/file"})`, MLflow copies each file into the model artifact store. At load time, `context.artifacts["name"]` returns the local path to the restored file.

## Step-by-Step

### Step 1: PythonModel with load_context()

We train a scikit-learn RandomForest, save it to a temporary file, then wrap it in a custom `SklearnWrapper` that loads the model inside `load_context()`. This pattern is useful when you want to add custom pre/post-processing around an existing model.

```python
class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)
```

The model is logged with the sklearn artifact:

```python
mlflow.pyfunc.log_model(
    name="sklearn_wrapper",
    python_model=SklearnWrapper(),
    artifacts={"sklearn_model": str(model_path)},
    signature=signature,
)
```

### Step 2: PyFunc with params Support

The `predict()` method accepts an optional `params` dict, letting callers change behavior at inference time without reloading. Here we build a text processor that calls Ollama's `gemma4:e2b` model with configurable `temperature` and `style`:

```python
result = loaded.predict(
    test_input,
    params={"temperature": 0.3, "style": "concise"}
)
```

The params schema is captured in the model signature so MLflow can validate inputs at serving time.

### Step 3: Multi-Model Ensemble

We train three classifiers (RandomForest, GradientBoosting, LogisticRegression), save each as a separate artifact, and wrap them in an `EnsembleModel` PyFunc. The ensemble averages class probabilities and returns the argmax:

```python
class EnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        for name in self.manifest["model_names"]:
            self.models[name] = joblib.load(context.artifacts[name])

    def predict(self, context, model_input, params=None):
        all_probas = [m.predict_proba(model_input) for m in self.models.values()]
        return np.argmax(np.mean(all_probas, axis=0), axis=1)
```

A JSON manifest tracks which models belong to the ensemble, making the pattern extensible.

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/2_custom_pyfunc
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: PythonModel with load_context()
============================================================
  Trained RandomForest — accuracy: 0.9778
  Saved sklearn model to /tmp/.../rf_model.joblib
  [load_context] Loaded sklearn model from ...
  Logged custom PyFunc model: runs:/.../sklearn_wrapper
  Predictions on 5 samples: [1 0 2 1 1]
  Expected:                 [1 0 2 1 1]

============================================================
Part 2: PyFunc with params support
============================================================
  Logged LLMTextProcessor model: runs:/.../text_processor

  --- Test 1: concise style, temperature=0.3 ---
  Result: <rewritten text in concise style>...

  --- Test 2: formal style, temperature=0.9 ---
  Result: <rewritten text in formal style>...

============================================================
Part 3: Multi-model ensemble as a single PyFunc
============================================================
  Trained random_forest             — accuracy: 0.9778
  Trained gradient_boosting         — accuracy: 0.9778
  Trained logistic_regression       — accuracy: 0.9778
  [load_context] Loaded model 'random_forest' from ...
  [load_context] Loaded model 'gradient_boosting' from ...
  [load_context] Loaded model 'logistic_regression' from ...

  Individual model accuracies:
    random_forest             0.9778
    gradient_boosting         0.9778
    logistic_regression       0.9778
    ENSEMBLE                  0.9778

============================================================
Done!
============================================================
Open MLflow UI at http://127.0.0.1:5000
```

## Key Takeaways

- **`load_context()`** is the right place to load heavy artifacts once, rather than on every `predict()` call.
- **`artifacts` dict** in `log_model()` tells MLflow which files to bundle with the model; they are restored automatically on load.
- **`params`** in `predict()` enables runtime configurability without retraining or relogging the model.
- **Ensemble models** can be packaged as a single PyFunc, simplifying deployment of composite prediction logic.
- Custom PyFunc models are served via `mlflow models serve` just like any built-in flavor.

## Next Steps

Continue to L2-M2.3 (Registry Workflows) to learn how to manage model lifecycle stages, aliases, and promotion workflows in the MLflow Model Registry.
