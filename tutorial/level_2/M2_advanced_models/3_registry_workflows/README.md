# L2-2.3 -- Model Registry Workflows

**Level:** Practitioner
**Duration:** 45 min

## Overview

This lesson walks through the full LLM model registry lifecycle: build two LLM model versions with different system prompts (precise vs. creative), register them, evaluate on test prompts, promote the best to champion via aliases, and load the champion model for serving. You will use `MlflowClient` for programmatic registry operations and see how alias-based deployment enables safe model promotion in CI/CD pipelines.

## Prerequisites

- Completed: L1-2.2 (Model Registry basics), L2-2.1 (Signatures Deep Dive), L2-2.2 (Custom PyFunc)
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Model Lifecycle (for LLMs)

LLM "model versions" are not retrained weights -- they are different configurations of the same base model. A system prompt change, a temperature tweak, or a RAG pipeline adjustment all produce meaningfully different behavior worth tracking as separate versions. The lifecycle is:

1. **Build** -- create model versions with different system prompts wrapped in `mlflow.pyfunc.PythonModel`
2. **Register** -- log each version to the model registry under a single registered model name
3. **Test** -- evaluate on a standard set of test prompts, measuring response length, latency, and a quality heuristic
4. **Promote** -- assign the `champion` alias to the best version based on evaluation metrics
5. **Serve** -- load the model by alias (`models:/MyModel@champion`) so deployment code never changes
6. **Retire** -- when a new champion is promoted, the alias is reassigned automatically; the old version remains available for rollback

### Alias-Based Deployment

Instead of hard-coding version numbers, you reference models by alias:

```python
model = mlflow.pyfunc.load_model("models:/L2-llm-assistant@champion")
```

When you promote a new version, you reassign the alias. Every downstream consumer automatically picks up the new model on the next load -- no code changes, no redeployments.

### CI/CD Promotion Pattern

A typical CI/CD pipeline for LLM model promotion:

1. Build a candidate model version (new system prompt, temperature, etc.)
2. Register it under the same registered model name
3. Run automated evaluation against a standard test set
4. Compare evaluation metrics against the current champion
5. If the candidate beats the champion, reassign the `champion` alias
6. Tag the old champion for rollback if needed

This lesson implements steps 1-5 programmatically using `MlflowClient`.

## Step-by-Step

### Step 1: Build Two LLM Model Versions

We create two instances of `LLMAssistant` (a custom `PythonModel`) with different system prompts. The "precise" assistant gives concise, factual answers. The "creative" assistant uses analogies and vivid language.

```python
class LLMAssistant(mlflow.pyfunc.PythonModel):
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt

    def predict(self, context, model_input, params=None):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        # ... call LLM with self.system_prompt for each question
```

### Step 2: Log Both to MLflow

Each model is logged in its own MLflow run with `mlflow.pyfunc.log_model()`. We also infer the model signature from a sample input/output pair so the registry knows the expected schema.

```python
signature = infer_signature(sample_input, sample_output)
mlflow.pyfunc.log_model(
    name="model",
    python_model=model,
    signature=signature,
)
```

### Step 3: Register as Model Versions

Both logged models are registered as versions of a single registered model name (`L2-llm-assistant`). This groups them for side-by-side comparison in the Model Registry UI.

```python
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "L2-llm-assistant")
```

### Step 4: Evaluate on Test Prompts

Both models answer five ML-related test prompts. For each response we measure:

- **Response length** -- longer, more detailed answers score higher
- **Latency** -- how long the LLM took to respond
- **Quality heuristic** -- `avg_length / (1 + avg_latency)`, rewarding detailed answers with low latency

These metrics are logged back to the original runs for tracking.

```python
quality_score = avg_length / (1.0 + avg_latency)
mlflow.log_metrics({
    "eval_avg_response_length": avg_length,
    "eval_avg_latency": avg_latency,
    "eval_quality_score": quality_score,
})
```

### Step 5: Promote Best to Champion

The version with the highest quality score gets the `champion` alias; the other gets `challenger`. We also set descriptions and role tags on each version.

```python
client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
client.set_registered_model_alias(MODEL_NAME, "challenger", runner_up_version)
```

### Step 6: Load Champion and Serve

Downstream serving code loads the model by alias. When a new champion is promoted, this code automatically uses the new version with no changes.

```python
champion_uri = f"models:/{MODEL_NAME}@champion"
champion_model = mlflow.pyfunc.load_model(champion_uri)
predictions = champion_model.predict(test_df)
```

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/3_registry_workflows
uv sync
uv run python main.py
```

## Expected Output

```
======================================================================
Step 1-2: Build two LLM model versions and log to MLflow
======================================================================
  Logged: precise_assistant (run a1b2c3d4...)
  Logged: creative_assistant (run e5f6g7h8...)

======================================================================
Step 3: Register both as versions of L2-llm-assistant
======================================================================
  precise_assistant          -> L2-llm-assistant v1
  creative_assistant         -> L2-llm-assistant v2

======================================================================
Step 4: Evaluate both models on test prompts
======================================================================

  Evaluating: precise_assistant
    [2.1s] What is machine learning?              -> 312 chars
    [1.8s] Explain the concept of overfitting.     -> 287 chars
    ...
    Summary: avg_length=298  avg_latency=1.95s  quality=101.0

  Evaluating: creative_assistant
    [2.3s] What is machine learning?              -> 485 chars
    [2.1s] Explain the concept of overfitting.     -> 421 chars
    ...
    Summary: avg_length=452  avg_latency=2.20s  quality=141.3

======================================================================
Step 5: Promote best to champion, runner-up to challenger
======================================================================
  champion   -> v2 (creative_assistant, quality=141.3)
  challenger -> v1 (precise_assistant, quality=101.0)

======================================================================
Step 6: Load champion by alias and demonstrate serving
======================================================================
  Loaded: models:/L2-llm-assistant@champion
  Q1: What is reinforcement learning?
  A1: Imagine you are training a dog -- you give it a treat when it sits...
  Q2: Why is data preprocessing important?
  A2: Think of data preprocessing like preparing ingredients before cooking...

======================================================================
Lifecycle Summary: All registered versions
======================================================================
 Version                Style Avg Length Avg Latency Quality     Alias
      v1   precise_assistant       298       1.95s   101.0 challenger
      v2  creative_assistant       452       2.20s   141.3   champion

======================================================================
Done! View the Model Registry in the MLflow UI:
  http://127.0.0.1:5000/#/models/L2-llm-assistant
======================================================================
```

Note: Exact values depend on LLM responses and server performance. Version numbers depend on whether previous versions of `L2-llm-assistant` exist in your registry.

## Key Takeaways

- **LLM versions are configurations, not retrained weights**: different system prompts, temperatures, or pipeline setups produce distinct model versions worth tracking.
- **One registered model, many versions**: group related LLM configurations under a single name for organized comparison.
- **Aliases replace stages**: MLflow 2.x uses `champion` / `challenger` aliases instead of the deprecated `Staging` / `Production` stages.
- **Evaluation drives promotion**: automated metrics (response quality, latency) determine which version gets the `champion` alias -- no manual judgment needed.
- **Alias-based loading decouples deployment from training**: serving code references `@champion`, so promotions require zero code changes.

## Next Steps

Move on to **L2-M3: Deep Evaluation** to learn how to build custom metrics and evaluation pipelines. In L2-3.1 you will create domain-specific evaluation functions that go beyond the simple quality heuristic used here, including LLM-as-judge scoring for open-ended response quality.
