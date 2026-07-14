# L1-M3.2 -- Model Registry

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

The MLflow Model Registry is a centralized hub for managing the full lifecycle
of models. In this lesson you will register LLM models with different
configurations, assign version aliases like "champion" and "challenger", add
metadata, and load a model by alias for inference.

## Prerequisites

- Completed: L1-M3.1 (Models and Flavors)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why a Model Registry?

When teams iterate on LLM configurations -- trying different system prompts,
temperatures, or model sizes -- they need a single place to:

1. **Track versions** -- every registered model gets an auto-incrementing
   version number.
2. **Assign aliases** -- labels like `champion` (production) and `challenger`
   (next candidate) that point to specific versions.
3. **Add metadata** -- descriptions and tags that document what each version
   is and how it was configured.
4. **Load by name** -- downstream services load
   `models:/MyModel@champion` instead of hard-coding run IDs.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Registered Model** | A named entry in the registry (e.g., `L1-llm-assistant`). |
| **Model Version** | An immutable snapshot tied to a specific run and artifact path. |
| **Alias** | A mutable pointer (e.g., `champion`) that can be moved between versions. |
| **Tags** | Key-value metadata on a model or version (e.g., `config=concise_assistant`). |
| **Description** | Free-text documentation on the model or a specific version. |

### Aliases vs. Stages (Legacy)

Older MLflow versions used "stages" (`Staging`, `Production`, `Archived`).
Modern MLflow replaces stages with **aliases**, which are more flexible --
you can define any alias name and a model can have multiple aliases.

## Step-by-Step

### Step 1: Define two LLM configurations

We create two model variants with different system prompts and temperatures:
a concise assistant (low temperature, brief answers) and a detailed assistant
(higher temperature, thorough answers).

```python
configs = {
    "concise_assistant": {
        "system_prompt": "You are a concise assistant. Answer in 1-2 sentences.",
        "temperature": 0.3,
    },
    "detailed_assistant": {
        "system_prompt": "You are a thorough assistant. Provide detailed answers.",
        "temperature": 0.7,
    },
}
```

### Step 2: Run and log both models

Each configuration is wrapped in a `PythonModel` that calls LMStudio
directly via the OpenAI SDK, then logged to its own MLflow run.

```python
class LLMAssistant(mlflow.pyfunc.PythonModel):
    def __init__(self, system_prompt, temperature):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def predict(self, context, model_input, params=None):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        # ... call the LLM with self.system_prompt and self.temperature

mlflow.pyfunc.log_model(name="model", python_model=model, ...)
```

### Step 3: Register models

`mlflow.register_model()` creates a new version under a named model. Both
configurations are registered under the same name.

```python
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "L1-llm-assistant")
```

### Step 4: Set aliases

Point `champion` at the detailed assistant and `challenger` at the concise one.

```python
client.set_registered_model_alias("L1-llm-assistant", "champion", version)
```

### Step 5: Load by alias and compare

Load models by their alias and compare outputs on the same question.

```python
champion = mlflow.pyfunc.load_model("models:/L1-llm-assistant@champion")
result = champion.predict(pd.DataFrame({"question": ["What is an LLM?"]}))
```

## Running the Lesson

```bash
cd tutorial/level_1/M3_models/2_model_registry
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Defining two LLM configurations
============================================================
  concise_assistant:
    temperature:   0.3
    system_prompt: You are a concise assistant...
  detailed_assistant:
    temperature:   0.7
    system_prompt: You are a thorough assistant...

============================================================
Step 2: Running and logging both models
============================================================
  concise_assistant          avg_len=85 chars   run_id=...
  detailed_assistant         avg_len=350 chars  run_id=...

============================================================
Step 3: Registering models in the Model Registry
============================================================
  Registered concise_assistant as L1-llm-assistant version 1
  Registered detailed_assistant as L1-llm-assistant version 2

...

============================================================
Done! View the Model Registry in the MLflow UI:
  http://127.0.0.1:5000/#/models/L1-llm-assistant
============================================================
```

In the MLflow UI, navigate to **Models** to see the registered model, its
versions, aliases, and tags.

## Key Takeaways

- The **Model Registry** provides a centralized, versioned catalog for models.
- Different LLM configurations (system prompts, temperatures) become different
  **versions** of the same registered model.
- **Aliases** (`champion`, `challenger`) are mutable pointers -- move them
  between versions to promote or roll back.
- **Descriptions** and **tags** document what each version does and how it
  was configured.
- Load models by name and alias (`models:/name@alias`) so downstream code
  never hard-codes run IDs.

## Next Steps

Continue to **L1-M4 Evaluations** to learn how to assess LLM output quality
with `mlflow.evaluate()`. In Level 2, we will explore advanced PyFunc models
(wrapping RAG pipelines and agents) and registry workflows including CI/CD
promotion.
