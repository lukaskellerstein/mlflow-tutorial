# L1-M2.1 -- MLflow Models and Flavors

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Learn how MLflow packages models into a portable, self-describing format. You
will understand what *flavors* are, how *signatures* document input/output
schemas, and how *input examples* make models self-documenting -- all
demonstrated with LLM and agent examples.

## Prerequisites

- Completed: L1-M1 (Tracking)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### What is an MLflow Model?

An MLflow Model is a standard directory layout that stores everything needed
to reproduce and serve a model:

```
agent_model/
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
| `langchain` | LangChain chains, agents, and LangGraph graphs |
| `openai` | Direct OpenAI API models |
| `transformers` | Hugging Face Transformers |
| `pyfunc` | Any Python code (custom models, API wrappers) |

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

### Step 1: Create a LangChain agent

We set up a `ChatOpenAI` model connected to LMStudio and create an agent
with two tools using `create_agent`.

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)

agent = create_agent(
    model=llm,
    tools=[get_word_length, reverse_string],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)
```

### Step 2: Log with the langchain flavor

Inside an MLflow run, we infer the signature from sample input/output and
log the agent with `mlflow.langchain.log_model()`.

```python
signature = infer_signature(sample_input, result)
mlflow.langchain.log_model(
    lc_model=agent,
    name="agent_model",
    signature=signature,
    input_example=sample_input,
)
```

### Step 3: Load and test

We load the model back using its run URI and verify it still works.

```python
model_uri = f"runs:/{run_id}/agent_model"
loaded_agent = mlflow.langchain.load_model(model_uri)
result = loaded_agent.invoke(test_input)
```

### Step 4: Log with the pyfunc flavor

For comparison, we wrap a raw LLM API call in a `PythonModel` and log
it with `mlflow.pyfunc.log_model()`.

```python
class SimpleLLMModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        # ... call the LLM and return results

mlflow.pyfunc.log_model(
    name="pyfunc_llm",
    python_model=SimpleLLMModel(),
    signature=pyfunc_signature,
    input_example=pyfunc_input,
)
```

### Step 5: Load the pyfunc model

The pyfunc model loads through the generic interface -- any MLflow model
can be loaded this way regardless of its original flavor.

```python
loaded_pyfunc = mlflow.pyfunc.load_model(pyfunc_uri)
result = loaded_pyfunc.predict(test_df)
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_models_registry/1_models_flavors
uv sync
uv run python main.py
```

## Expected Output

You should see:
- The agent running and answering questions using its tools
- The inferred model signature for the langchain model
- The agent loaded back from MLflow and producing new answers
- A PyFunc-wrapped LLM model logged, loaded, and tested

In the MLflow UI at http://127.0.0.1:5000 you can:
- Open each run and inspect the **Artifacts** tab
- Compare the `MLmodel` files to see different flavors listed
- View signatures and input examples for each model

## Key Takeaways

- An MLflow Model is a portable directory with an `MLmodel` manifest.
- **Flavors** let the same model be loaded natively or through the generic pyfunc interface.
- The **langchain** flavor logs LangChain chains and agents with their tools and prompts.
- The **pyfunc** flavor wraps any Python code -- useful for custom LLM integrations.
- **Signatures** document and enforce the expected input/output schema.
- **Input examples** make models self-documenting -- always include one.

## Next Steps

In the next lesson (L1-M2.2 -- Model Registry) you will learn how to
register models, manage versions, and assign lifecycle aliases like
`champion` and `challenger`.
