# L1-M3.1 -- MLflow Models and Flavors

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Learn how MLflow packages models into a portable, self-describing format. You
will understand what *flavors* are, how *signatures* document input/output
schemas, and how *input examples* make models self-documenting -- demonstrated
by logging the same LLM two different ways: as a PyFunc model and with the
OpenAI flavor.

## Prerequisites

- Completed: L1-M2 (Tracing)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### What is an MLflow Model?

An MLflow Model is a standard directory layout that stores everything needed
to reproduce and serve a model:

```
model_artifact/
  MLmodel              # YAML manifest -- lists flavors, signature, etc.
  model artifacts      # Serialized model (pickle, graph, weights, etc.)
  conda.yaml           # Conda environment spec
  requirements.txt     # pip dependencies
  input_example.json   # Sample input (optional)
```

The `MLmodel` file is the key -- it tells MLflow *how* to load the model.

### What are Flavors?

A **flavor** is a named interface through which a model can be saved and
loaded. Every model gets the generic `python_function` (pyfunc) flavor so
it can always be loaded and served the same way. Framework-specific flavors
give you access to the native model object.

Key flavors for LLM work:

| Flavor | Use Case |
|--------|----------|
| `pyfunc` | Any Python code (custom models, API wrappers) |
| `openai` | OpenAI-compatible chat/completion models (including LMStudio) |
| `transformers` | Hugging Face Transformers |

### Model Signatures

A `ModelSignature` records input/output schemas. MLflow uses signatures to:

- **Validate** data before inference -- catch schema errors early.
- **Generate** REST API documentation when serving.
- **Display** schema in the MLflow UI.

You create a signature with `mlflow.models.infer_signature(inputs, outputs)`.

### Input Examples

An *input example* is a small sample saved alongside the model. It serves
as living documentation -- anyone can look at the model artifact and
immediately see what data the model expects.

## Step-by-Step

### Step 1: Log an LLM with the PyFunc flavor

We wrap a direct OpenAI SDK call to LMStudio inside a `PythonModel` subclass.
This is the most flexible approach -- it can wrap any Python code.

```python
class DirectLLMModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        questions = model_input["question"].tolist()
        answers = []
        for q in questions:
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[{"role": "user", "content": q}],
            )
            answers.append(resp.choices[0].message.content)
        return answers

mlflow.pyfunc.log_model(
    name="pyfunc_llm",
    python_model=DirectLLMModel(),
    signature=pyfunc_signature,
    input_example=pyfunc_input,
)
```

### Step 2: Load and test the PyFunc model

Load through the generic `pyfunc` interface and call `predict()`.

```python
loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc_llm")
result = loaded.predict(pd.DataFrame({"question": ["What is MLflow?"]}))
```

### Step 3: Log an LLM with the OpenAI flavor

The `openai` flavor is declarative -- just specify the model name, task, and
a message template with `{variable}` placeholders.

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

The pyfunc wrapper reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the
environment at prediction time, so set them before loading.

### Step 4: Load the OpenAI model two ways

Load natively to get the raw config dict, or through the generic pyfunc
interface. Both work because every flavor includes `python_function` as a
base.

```python
# Native load -- returns the saved config as a dict
raw = mlflow.openai.load_model(uri)

# Generic pyfunc load -- works for any MLflow model
model = mlflow.pyfunc.load_model(uri)
result = model.predict(pd.DataFrame({"question": ["What is an API?"]}))
```

## Running the Lesson

```bash
cd tutorial/level_1/M3_models/1_models_flavors
uv sync
uv run python main.py
```

## Expected Output

You should see:
- A PyFunc model wrapping a direct LLM call, logged and loaded back
- An OpenAI-flavor model logged declaratively, loaded natively (as a dict) and via pyfunc
- Both models producing LLM responses when loaded through `mlflow.pyfunc.load_model()`

In the MLflow UI at http://127.0.0.1:5000 you can:
- Open each run and inspect the **Artifacts** tab
- Compare the `MLmodel` files to see different flavors listed
- View signatures and input examples for each model

## Key Takeaways

- An MLflow Model is a portable directory with an `MLmodel` manifest.
- **Flavors** let the same model be loaded natively or through the generic pyfunc interface.
- The **pyfunc** flavor wraps any Python code -- ideal for custom LLM wrappers with full control.
- The **openai** flavor logs OpenAI-compatible models declaratively -- no custom code needed.
- **Signatures** document and enforce the expected input/output schema.
- **Input examples** make models self-documenting -- always include one.
- Every model can be loaded via `mlflow.pyfunc.load_model()` regardless of its original flavor.

## Next Steps

In the next lesson (L1-M3.2 -- Model Registry) you will learn how to
register models, manage versions, and assign lifecycle aliases like
`champion` and `challenger`.
