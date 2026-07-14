# L2-M2.1 — Model Signatures Deep Dive

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

Model signatures define the contract between an LLM-backed model and its
consumers. This lesson explores how to build signatures for chat completion
models, structured JSON output, and inference-time parameters like temperature
and max_tokens -- the patterns you will use whenever you wrap an LLM as a
logged MLflow model.

## Prerequisites

- Completed: L1-M3.1 (Models and Flavors -- basic signature usage)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Signature Types for LLMs

When wrapping LLM calls as MLflow `PythonModel` objects, signatures use
column-based schemas (`ColSpec`) to describe string inputs and outputs.
The most common patterns are:

| Pattern | Input | Output |
|---------|-------|--------|
| **Chat completion** | `question` (string) | list of strings |
| **Structured output** | `text` (string) | JSON string |
| **Configurable chat** | `question` (string) + params | list of strings |

Unlike traditional ML models that take numeric arrays, LLM signatures
typically operate on string columns -- a question, prompt, or text passage
goes in, and a generated string comes out.

### Inference Parameters (ParamSpec)

Real LLM applications need runtime control over generation behavior.
`ParamSpec` lets you declare these parameters as part of the model
signature so that consumers know what knobs are available:

```python
from mlflow.types import ParamSchema, ParamSpec, DataType

params = ParamSchema([
    ParamSpec("temperature", DataType.double, default=0.7),
    ParamSpec("max_tokens", DataType.long, default=256),
])
```

When a consumer calls `model.predict(data, params={"temperature": 0.2})`,
MLflow validates the parameter names and types against the schema before
forwarding them to the model's `predict` method.

### Signature Enforcement

When you load a model with `mlflow.pyfunc.load_model()` and call
`.predict()`, MLflow validates the input against the stored signature:

- Missing required columns trigger warnings or errors.
- Extra columns are flagged (and typically ignored).
- Type mismatches may be coerced (e.g., int to string) or rejected.

This catch-errors-early behavior is especially valuable when serving
models behind a REST API, where malformed requests should fail fast.

## Step-by-Step

### Part 1: Chat Completion Signature

We wrap an LLM call in a `ChatModel` (subclass of `PythonModel`) and use
`infer_signature()` to automatically capture the input/output schema from
sample data.

```python
input_df = pd.DataFrame({"question": ["What is MLflow?"]})
output = ["MLflow is an open-source platform..."]
signature = infer_signature(input_df, output)
```

The inferred signature records that the model expects a DataFrame with a
`question` string column and returns a list of strings. We log the model
with this signature and then load it back to verify prediction works.

### Part 2: Structured Output Signature

For models that return JSON, we build the signature manually using
`Schema` and `ColSpec`. This gives explicit control over column names and
types on both input and output sides.

```python
input_schema = Schema([ColSpec(DataType.string, "text")])
output_schema = Schema([ColSpec(DataType.string, "json_output")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

The `StructuredOutputModel` instructs the LLM to return JSON with specific
keys (`summary`, `key_points`, `confidence`), and the signature documents
this contract for downstream consumers.

### Part 3: Signature with Inference Params

We attach `ParamSpec` entries so that consumers can pass `temperature` and
`max_tokens` at inference time. The `ConfigurableChatModel` reads these
from the `params` dict in its `predict` method.

```python
param_schema = ParamSchema([
    ParamSpec("temperature", DataType.double, default=0.7),
    ParamSpec("max_tokens", DataType.long, default=256),
])
signature = ModelSignature(
    inputs=input_schema, outputs=output_schema, params=param_schema,
)
```

When calling the loaded model, the consumer passes params explicitly:

```python
model.predict(data, params={"temperature": 0.2, "max_tokens": 64})
```

### Part 4: Signature Enforcement

We load the Part 1 chat model and test it with several kinds of
malformed input to see how MLflow enforces the signature:

- **Wrong column name** (`query` instead of `question`) -- may error or warn.
- **Wrong data type** (integer instead of string) -- MLflow may coerce.
- **Extra columns** -- typically ignored with a warning.

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/1_signatures_deep_dive
uv sync
uv run python main.py
```

## Expected Output

You should see four sections, each printing:

- The signature in human-readable form
- The signature as JSON (showing the internal representation)
- Confirmation that the model was logged
- A test prediction from the loaded model

For Part 4 (enforcement), you will see whether MLflow accepts or rejects
each malformed input. The exact behavior depends on your MLflow version --
newer versions tend to be stricter about schema validation.

In the MLflow UI at http://127.0.0.1:5000:

- Open each run under the `L2/M2_advanced_models/1_signatures_deep_dive`
  experiment
- Click into the model artifact and inspect the **MLmodel** file
- The signature section shows inputs, outputs, and params schemas

## Key Takeaways

- LLM model signatures use column-based schemas (`ColSpec`) with string types for text-in, text-out patterns.
- Use `infer_signature()` for quick automatic signatures from sample data, or build them manually with `Schema` and `ColSpec` for precise control.
- `ParamSpec` declares inference-time parameters (temperature, max_tokens) as part of the model contract, enabling runtime configuration.
- Signature enforcement catches schema mismatches at prediction time, acting as a safety net for production serving.
- Always include a signature when logging models; it serves as documentation, validation, and REST API schema generation.

## Next Steps

In the next lesson (L2-M2.2 -- Custom PyFunc Models) you will learn how to
build fully custom models using `mlflow.pyfunc.PythonModel`, including
loading external artifacts, custom predict logic, and advanced signatures
for non-standard model types.
