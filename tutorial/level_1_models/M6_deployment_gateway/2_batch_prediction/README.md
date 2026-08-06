# L1-M6.2 -- Batch LLM Inference Pipeline

**Level:** Practitioner
**Duration:** 45 min

## Overview

Batch LLM inference lets you score a collection of prompts in one pipeline run rather than handling them one at a time through a REST endpoint. This lesson wraps a local LLM inside an MLflow PyFunc model, runs a batch of diverse prompts, and tracks latency, token usage, and cost estimates -- the same pattern you would use with a paid API in production.

## Prerequisites

- Completed: L1-M6.1 (Model Serving)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway up (`cd infra && podman compose up -d`), with LMStudio
  serving `google/gemma-4-26b-a4b` behind the `gemma-chat` alias

## Concepts

### Batch vs. Real-Time for LLMs

| Aspect | Real-Time (Serving) | Batch |
|--------|---------------------|-------|
| Latency | Per-request SLA | Total wall-clock time |
| Input | Single prompt | DataFrame of prompts |
| Trigger | API call / user action | Schedule or event |
| Use case | Chatbots, assistants | Bulk classification, report generation, data enrichment |
| Cost control | Hard to predict | Predictable per-batch budget |

Batch inference is the right choice when results are not needed immediately -- nightly content moderation, weekly sentiment scoring, bulk document summarization, or offline evaluation of prompt variants.

### Cost Tracking

Even with a free local model, tracking token usage and latency per prompt builds the habit you need when switching to a paid API. The lesson logs `total_tokens`, `cost_estimate_usd`, and `avg_latency_per_prompt_ms` so you can set budgets and catch regressions.

### PyFunc for Batch Scoring

`mlflow.pyfunc.PythonModel` provides a framework-agnostic wrapper. By implementing `load_context` (initialize the OpenAI client) and `predict` (loop over prompts, call the LLM, collect results), the model can be loaded anywhere with `mlflow.pyfunc.load_model()` and scored with a single `.predict(df)` call -- locally, in a notebook, or via the `mlflow models predict` CLI.

## Step-by-Step

### Step 1: Create and Log the LLM PyFunc Model

We define `LLMModel(mlflow.pyfunc.PythonModel)` with two methods:

- `load_context` -- initializes the OpenAI client pointing at the LiteLLM gateway.
- `predict` -- iterates over the `prompt` column, calls the LLM, and returns a DataFrame with `response`, `latency_ms`, and `tokens_used`.

The model is logged with an inferred signature so MLflow validates inputs at prediction time.

```python
class LLMModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from openai import OpenAI

        self.client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-litellm-master")

    def predict(self, context, model_input, params=None):
        # Loop over prompts, call LLM, collect responses + timing + tokens
        ...


mlflow.pyfunc.log_model(name="llm_model", python_model=LLMModel(), signature=signature)
```

### Step 2: Build Batch Prompts and Run Inference

A DataFrame of eight diverse prompts (summarization, translation, Q&A, classification, creative writing, explanation, listing, rewriting) is scored through the loaded model.

```python
model = mlflow.pyfunc.load_model(model_uri)
results = model.predict(batch_prompts)
```

Each prompt is timed individually so you can see which task types are faster or slower.

### Step 3: Log Results to MLflow

Batch metrics are logged to a dedicated run:

- `batch_size` -- number of prompts
- `total_latency_sec` -- wall-clock time for the full batch
- `avg_latency_per_prompt_ms` -- mean per-prompt latency
- `total_tokens` -- sum of tokens across all completions
- `cost_estimate_usd` -- estimated cost at a configurable rate

The full prompts-and-responses table is saved as a CSV artifact.

```python
mlflow.log_metrics({...})
mlflow.log_artifact("batch_results.csv", artifact_path="results")
```

### Step 4: CLI Batch Prediction

The lesson prints ready-to-use `mlflow models predict` commands for offline scoring, along with scheduling examples for cron and workflow orchestrators.

```bash
mlflow models predict -m "runs:/<run_id>/llm_model" \
  -i prompts.csv -o responses.csv --content-type csv
```

## Running the Lesson

```bash
cd tutorial/level_2/M7_deployment/2_batch_prediction
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
L1-M6.2 -- Batch LLM Inference Pipeline
============================================================

============================================================
Part 1: Create and Log the LLM PyFunc Model
============================================================
  Model URI: runs:/<run_id>/llm_model
  Signature: inputs: ['prompt': string] -> outputs: ['response': string, ...]

============================================================
Part 2: Batch LLM Inference
============================================================
  Batch size: 8 prompts
  Model loaded from: runs:/<run_id>/llm_model
  [1] 1200ms | 85 tok | Renewable energy reduces greenhouse gas emissions and ...
  [2]  950ms | 42 tok | Le temps est magnifique aujourd'hui. ...
  ...
  Total time: 8.50s

============================================================
Part 3: Log Batch Results to MLflow
============================================================
  batch_size              : 8
  total_latency_sec       : 8.50
  avg_latency_per_prompt  : 1062.5 ms
  total_tokens            : 520
  cost_estimate_usd       : $0.000052
  Logged artifact: results/batch_results.csv

============================================================
Part 4: CLI Batch Prediction
============================================================
  ...CLI commands and scheduling examples...

============================================================
Done!
============================================================
```

In the MLflow UI you will see two runs under the experiment:
- **log_llm_model** -- the logged PyFunc model with parameters
- **batch_inference_results** -- batch metrics and the CSV artifact

## Key Takeaways

- Wrapping an LLM in `mlflow.pyfunc.PythonModel` gives you a reusable, framework-agnostic batch scoring interface.
- Tracking per-prompt latency and token usage lets you set cost budgets and catch performance regressions early.
- Logging the full prompts-and-responses CSV as an artifact creates an audit trail for every batch run.
- The same PyFunc model works with programmatic `.predict()` calls and the `mlflow models predict` CLI.
- Batch inference is the right pattern for offline tasks like bulk classification, content moderation, and data enrichment where real-time latency is not required.

## Next Steps

Move on to L2-M8 (Framework Integrations) to see how MLflow integrates with PyTorch, Hugging Face, and LangChain for deeper model tracking and evaluation. In Level 3, you will build production batch pipelines with scheduling, monitoring, and CI/CD quality gates.
