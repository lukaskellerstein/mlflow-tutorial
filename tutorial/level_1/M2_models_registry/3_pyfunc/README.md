# L1-M2.3 — PyFunc: The Universal Model Wrapper

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

PyFunc (`mlflow.pyfunc`) is MLflow's universal model interface. Every MLflow model exposes a `python_function` flavor, which means any model — regardless of framework — can be loaded and called via the same `predict()` API. In this lesson you will build a custom `PythonModel` that wraps a prompt template and an Ollama LLM call, log it to MLflow, load it back, and run inference through the standard interface.

## Prerequisites

- Completed: L1-M2.1 (Models and Flavors), L1-M2.2 (Model Registry)
- MLflow server running at http://127.0.0.1:5000
- Ollama running locally with the `gemma4:e2b` model pulled

## Concepts

### The PyFunc Flavor

Every model saved by MLflow includes a `python_function` (pyfunc) flavor in its `MLmodel` manifest. This flavor provides a single, framework-agnostic entry point:

```python
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)
```

Whether the underlying model is scikit-learn, PyTorch, a Hugging Face transformer, or completely custom logic, the calling code stays the same.

### Why PyFunc Matters

- **Uniform serving**: `mlflow models serve` works with any pyfunc model.
- **Evaluation**: `mlflow.evaluate()` accepts pyfunc models directly.
- **Registry**: custom models can be versioned and registered just like framework models.
- **Flexibility**: wrap API calls, rule engines, ensembles, or LLM pipelines.

### Custom PythonModel

To wrap custom logic, subclass `mlflow.pyfunc.PythonModel` and implement `predict()`:

```python
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        # Your custom logic here
        return results
```

- `context` — a `PythonModelContext` giving access to saved artifacts.
- `model_input` — a pandas DataFrame (by default).
- `params` — optional inference-time parameters.

You can also override `load_context(self, context)` to load heavy resources (model weights, config files) once when the model is loaded rather than on every predict call.

## Step-by-Step

### Step 1: Define a Prompt-Template Model

We create a `PromptTemplateModel` that stores a prompt template string and an Ollama model name. The `predict()` method formats the template with each input row and calls the LLM:

```python
class PromptTemplateModel(mlflow.pyfunc.PythonModel):
    def __init__(self, template: str, model_name: str = "gemma4:e2b"):
        self.template = template
        self.model_name = model_name

    def predict(self, context, model_input, params=None):
        import ollama
        results = []
        for _, row in model_input.iterrows():
            prompt = self.template.format(**row.to_dict())
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            results.append(response["message"]["content"])
        return results
```

### Step 2: Log the Model to MLflow

We call `mlflow.pyfunc.log_model()` with our custom instance, a signature inferred from a sample run, and an input example:

```python
mlflow.pyfunc.log_model(
    artifact_path="prompt_model",
    python_model=model,
    signature=signature,
    input_example=test_input,
)
```

### Step 3: Load and Predict

Load the model back using the standard pyfunc interface and call `predict()` with new inputs:

```python
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/prompt_model")
predictions = loaded_model.predict(new_input)
```

The loaded model is framework-agnostic — the caller does not need to know it wraps Ollama.

## Running the Lesson

```bash
cd tutorial/level_1/M2_models_registry/3_pyfunc
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L1-M2.3 — PyFunc: The Universal Model Wrapper
============================================================

WHAT IS PYFUNC?
  ...

============================================================
Step 1: Defining a prompt-template PyFunc model
============================================================
  Template : Explain {topic} in one sentence for a {audience}.
  LLM      : gemma4:e2b

============================================================
Step 2: Logging the PyFunc model to MLflow
============================================================
  Model logged. Run ID: <run_id>
  Signature:
  inputs: ['topic': string, 'audience': string]
  outputs: [string]

============================================================
Step 3: Loading model back and running predictions
============================================================
  Loaded model from: runs:/<run_id>/prompt_model

  Inputs:
    topic='black holes', audience='teenager'
    topic='recursion', audience='beginner programmer'
    topic='democracy', audience='ten-year-old'

  LLM Outputs:
    [1] <one-sentence explanation of black holes for a teenager>
    [2] <one-sentence explanation of recursion for a beginner>
    [3] <one-sentence explanation of democracy for a ten-year-old>

============================================================
Done!  Check the MLflow UI at http://127.0.0.1:5000
...
```

In the MLflow UI, navigate to the run to see the model artifact, its `MLmodel` manifest listing the `python_function` flavor, the saved signature, and the input example.

## Key Takeaways

- **PyFunc is the universal interface** — every MLflow model exposes `python_function`, so `load_model` + `predict` always works.
- **`PythonModel` wraps any logic** — subclass it to turn API calls, LLM pipelines, or rule engines into standard MLflow models.
- **Signatures and input examples** still work with custom models, enabling schema validation and documentation.
- **The loaded model hides implementation details** — callers just pass a DataFrame and get results.

## Next Steps

In L1-M3 (Autologging) you will see how MLflow can automatically capture models, metrics, and parameters for supported frameworks — removing the need for explicit `log_model` calls.
