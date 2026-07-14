# L1-M3.3 -- Registry Workflows

**Level:** Essentials
**Duration:** 35m

## Overview

Walk through the full LLM model registry lifecycle: build two model versions with different system prompts, register them, evaluate on test prompts, promote the best to champion via aliases, and load the champion for serving. This lesson combines registry basics with the evaluate-and-promote workflow.

## Prerequisites

- Completed: L1-M3.1 (Models, Flavors, and Signatures)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Model Registry

The MLflow Model Registry is a centralized hub for managing model versions. Key concepts:

| Concept | Description |
|---------|-------------|
| **Registered Model** | A named entry (e.g., `L1-llm-assistant`) |
| **Model Version** | An immutable snapshot tied to a specific run |
| **Alias** | A mutable pointer (e.g., `champion`) that can be moved between versions |
| **Tags** | Key-value metadata on a model or version |
| **Description** | Free-text documentation |

### LLM Versions Are Configurations

LLM "model versions" are not retrained weights -- they are different configurations of the same base model. A system prompt change, a temperature tweak, or a RAG pipeline adjustment all produce meaningfully different behavior worth tracking as separate versions.

### Alias-Based Deployment

Instead of hard-coding version numbers, reference models by alias:

```python
model = mlflow.pyfunc.load_model("models:/L1-llm-assistant@champion")
```

When you promote a new version, reassign the alias. Downstream consumers automatically pick up the new model on the next load -- no code changes needed.

### The Lifecycle

1. **Build** -- create model versions with different system prompts
2. **Register** -- log each version under a single registered model name
3. **Evaluate** -- test on standard prompts, measuring quality metrics
4. **Promote** -- assign `champion` alias to the best version
5. **Serve** -- load by alias so deployment code never changes
6. **Retire** -- when a new champion is promoted, the alias moves automatically

## Step-by-Step

### Step 1: Define and Log Two Configurations

A concise assistant (low temperature, brief) and a detailed assistant (higher temperature, thorough), both wrapped in the same `LLMAssistant` PyFunc class.

```python
class LLMAssistant(mlflow.pyfunc.PythonModel):
    def __init__(self, system_prompt: str, temperature: float = 0.7):
        self.system_prompt = system_prompt
        self.temperature = temperature
    def predict(self, context, model_input, params=None):
        # ... call LLM with self.system_prompt
```

### Step 2: Register as Versions

Both models are registered under the same name, creating version 1 and version 2.

```python
mv = mlflow.register_model(f"runs:/{run_id}/model", "L1-llm-assistant")
```

### Step 3: Evaluate on Test Prompts

Both models answer ML-related questions. For each response we measure response length, latency, and a quality heuristic (`avg_length / (1 + avg_latency)`). Metrics are logged back to the original runs.

### Step 4: Promote Best to Champion

The version with the highest quality score gets the `champion` alias; the other gets `challenger`. Descriptions and tags are set on each version.

```python
client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
```

### Step 5: Load and Serve by Alias

```python
champion = mlflow.pyfunc.load_model("models:/L1-llm-assistant@champion")
predictions = champion.predict(test_df)
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M3_models_registry/3_registry_workflows
uv sync
uv run python main.py
```

## Expected Output

Six parts run sequentially:

1. Two LLM configurations defined and logged to separate runs
2. Both registered as versions of `L1-llm-assistant`
3. Evaluation results showing response length, latency, and quality scores
4. Champion/challenger aliases assigned based on evaluation
5. Champion and challenger loaded by alias with sample predictions
6. Comparison summary table

In the MLflow UI, navigate to **Models** to see the registered model, versions, aliases, and tags.

## Key Takeaways

- The Model Registry provides a centralized, versioned catalog for models.
- Different LLM configurations become different versions of the same registered model.
- Aliases (`champion`, `challenger`) are mutable pointers -- move them between versions to promote or roll back.
- Evaluation drives promotion: automated metrics determine which version gets the `champion` alias.
- Alias-based loading (`models:/name@alias`) decouples deployment from training -- no code changes needed.
- Descriptions and tags document what each version does and how it was configured.

## Next Steps

Continue to **L1-M4.1 (Evaluation Fundamentals)** to learn how to assess LLM output quality with `mlflow.genai.evaluate()`, built-in scorers, and LLM-as-judge evaluation.
