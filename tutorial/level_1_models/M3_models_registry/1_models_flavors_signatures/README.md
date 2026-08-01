# L1-M3.1 -- Models, Flavors, and Signatures

**Level:** Essentials
**Duration:** 35m

## Overview

Learn how MLflow packages models into a portable, self-describing format. This lesson covers two model flavors (pyfunc and openai), three approaches to defining signatures (inferred, manual, and parameterized), and how MLflow enforces signatures at prediction time.

## Prerequisites

- Completed: L1-M2 (Tracing)
- MLflow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### What is an MLflow Model?

An MLflow Model is a standard directory layout containing:

| File | Purpose |
|------|---------|
| `MLmodel` | YAML manifest listing available flavors and signature |
| Model artifacts | Serialized model (pickle, weights, config) |
| `conda.yaml` | Conda environment specification |
| `requirements.txt` | pip dependencies |
| `input_example.json` | Sample input for documentation (optional) |

### Flavors

A **flavor** is a named interface for saving and loading a model. Every model gets the generic `python_function` (pyfunc) flavor so it can always be loaded and served the same way. Framework-specific flavors provide native access.

| Flavor | Use Case |
|--------|----------|
| `pyfunc` | Any Python code (custom models, API wrappers) |
| `openai` | OpenAI-compatible chat/completion models (including LMStudio) |
| `transformers` | Hugging Face Transformers |

### Signatures

A `ModelSignature` records input/output schemas. MLflow uses signatures to validate data before inference, generate REST API docs when serving, and display schema in the UI.

Three approaches:
- **Inferred** -- `infer_signature(inputs, outputs)` auto-detects schema from sample data
- **Manual** -- `ModelSignature(inputs=Schema([ColSpec(...)]), outputs=...)` for precise control
- **Parameterized** -- add `ParamSchema` with `ParamSpec` entries for runtime-configurable parameters like temperature and max_tokens

### Signature Enforcement

When you load a model with `mlflow.pyfunc.load_model()` and call `.predict()`, MLflow validates the input against the stored signature. Missing columns, wrong types, and extra columns are flagged.

## Step-by-Step

### Step 1: Log with PyFunc Flavor

Wrap a direct OpenAI SDK call in a `PythonModel` subclass:

```python
class LLMModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        questions = model_input["question"].tolist()
        # ... call the LLM for each question

mlflow.pyfunc.log_model(name="pyfunc_llm", python_model=LLMModel(), ...)
```

### Step 2: Log with OpenAI Flavor

The `openai` flavor is declarative -- just specify the model name, task, and a message template:

```python
mlflow.openai.log_model(
    model="google/gemma-4-e4b",
    task="chat.completions",
    name="openai_llm",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{question}"},
    ],
)
```

### Step 3: Infer Signatures

```python
signature = infer_signature(input_df, output)
```

### Step 4: Build Manual Signatures

```python
input_schema = Schema([ColSpec(DataType.string, "text")])
output_schema = Schema([ColSpec(DataType.string, "json_output")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

### Step 5: Add Inference Parameters

```python
param_schema = ParamSchema(
    [
        ParamSpec("temperature", DataType.double, default=0.7),
        ParamSpec("max_tokens", DataType.long, default=256),
    ]
)
signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)
```

### Step 6: Test Signature Enforcement

Load a model and test with wrong column names, wrong types, and extra columns to see how MLflow validates inputs.

## Running the Lesson

```bash
cd tutorial/level_1_models/M3_models_registry/1_models_flavors_signatures
uv sync
uv run python main.py
```

## Expected Output

Six parts will run sequentially:

1. A PyFunc model wrapping a direct LLM call, logged and loaded back
2. An OpenAI-flavor model logged declaratively, loaded natively and via pyfunc
3. An inferred signature displayed with its JSON representation
4. A manual signature with explicit Schema and ColSpec
5. A parameterized signature with temperature and max_tokens ParamSpecs
6. Signature enforcement tests showing how MLflow handles malformed input

## Key Takeaways

- An MLflow Model is a portable directory with an `MLmodel` manifest listing available flavors.
- The **pyfunc** flavor wraps any Python code -- ideal for custom LLM wrappers with full control.
- The **openai** flavor logs OpenAI-compatible models declaratively -- no custom code needed.
- Use `infer_signature()` for quick automatic signatures, or build manually with `Schema`/`ColSpec` for precise control.
- `ParamSpec` declares inference-time parameters as part of the model contract.
- Signature enforcement catches schema mismatches at prediction time.
- Every model can be loaded via `mlflow.pyfunc.load_model()` regardless of its original flavor.

## Next Steps

Continue to **L1-M3.3 (Registry Workflows)** to learn how to register models, manage versions with aliases, evaluate candidates, and promote the best to champion.
