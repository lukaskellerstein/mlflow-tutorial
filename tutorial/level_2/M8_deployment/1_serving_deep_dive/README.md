# L2-8.1 -- LLM Model Serving Deep Dive

**Level:** Practitioner
**Duration:** 1 hour

## Overview

Learn how to serve LLM models through MLflow by wrapping an OpenAI-compatible LLM in a custom PyFunc model. This lesson covers the full serving lifecycle: defining a reusable chat model wrapper, logging multiple versions with different configurations, Docker containerization, and production health checks. You will work with a local LMStudio-hosted model, but the patterns apply to any OpenAI-compatible endpoint.

## Prerequisites

- Completed: L1-8.1 Model Serving, L1-2.1 Models & Flavors, L2-2.2 Custom PyFunc Models
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded at http://localhost:1234

## Concepts

### PyFunc Wrapping for LLMs

LLMs do not fit neatly into traditional ML flavors like `sklearn` or `pytorch`. By wrapping an LLM client in `mlflow.pyfunc.PythonModel`, you get a standardized serving interface: clients send a DataFrame with prompts, and the model returns a DataFrame with responses. The `__init__` method stores configuration (model name, default temperature), `load_context` initializes the LLM client at serving time, and `predict` handles the chat completion loop.

### Chat Interface Patterns

The PyFunc model accepts a DataFrame with a required `prompt` column and optional `temperature` and `max_tokens` columns. This pattern lets callers override generation parameters per request while keeping sensible defaults. The response DataFrame includes both the generated text and token usage, enabling downstream cost tracking.

### Docker Containerization

`mlflow models build-docker` packages the PyFunc model, its pip dependencies, and the MLflow serving infrastructure into a single container image. For LLM models, the container needs network access to the LLM provider (LMStudio, OpenAI, etc.), so you must configure host networking or pass the provider URL as an environment variable.

### Health Checks and Monitoring

Every MLflow serving endpoint exposes standard health check routes that integrate with Kubernetes probes and load balancers. For LLM workloads, monitoring token usage per request is especially important for cost control, alongside the standard latency and error rate metrics.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/invocations` | POST | Prediction requests (send prompts, get responses) |
| `/ping` | GET | Liveness probe (returns 200 when server is ready) |
| `/health` | GET | Alias for `/ping` |
| `/version` | GET | MLflow version info |

## Step-by-Step

### Step 1: Define and log the LLM PyFunc model

Create an `LLMChatModel` class that extends `mlflow.pyfunc.PythonModel`. The class stores the model name and default temperature, initializes an OpenAI client in `load_context`, and runs chat completions in `predict`:

```python
class LLMChatModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name="google/gemma-4-e4b", default_temperature=0.7):
        self.model_name = model_name
        self.default_temperature = default_temperature

    def load_context(self, context):
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def predict(self, context, model_input, params=None):
        # Iterate rows, call chat completion, return DataFrame
        ...
```

Log the model with a proper signature inferred from sample input/output DataFrames. The signature tells MLflow (and clients) exactly what columns to send and what to expect back.

### Step 2: Load and test locally

Load the logged model back using `mlflow.pyfunc.load_model()` and run sample prompts through it. This verifies the full serialize-deserialize cycle works before attempting to serve the model:

```python
loaded_model = mlflow.pyfunc.load_model(model_uri)
results = loaded_model.predict(test_prompts)
```

### Step 3: Log a second version and register both

Log another version of the model with a lower temperature (0.3) for more deterministic responses. Register both versions in the Model Registry and assign aliases (`champion` for creative, `challenger` for deterministic):

```python
client.set_registered_model_alias(reg_name, "champion", mv1.version)
client.set_registered_model_alias(reg_name, "challenger", mv2.version)
```

This lets you serve different configurations side by side and A/B test them.

### Step 4: Docker containerization

Review the `mlflow models build-docker` command for packaging the LLM model into a container. For LLM models, the container needs network access to the LLM provider:

```bash
mlflow models build-docker \
    --model-uri models:/llm-serving-demo@champion \
    --name mlflow-llm-server

podman run -p 5001:8080 \
    --add-host=host.containers.internal:host-gateway \
    mlflow-llm-server
```

### Step 5: Health checks and monitoring

Review the standard MLflow serving endpoints and monitoring best practices. For LLM workloads, token usage tracking is a critical addition to standard latency and error rate monitoring.

## Running the Lesson

```bash
cd tutorial/level_2/M8_deployment/1_serving_deep_dive
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Define and log LLM PyFunc model (temp=0.7)
============================================================
  Model logged: runs:/<id>/model
  Default temperature: 0.7

============================================================
Part 2: Load and test locally with sample prompts
============================================================
  Loading model from: runs:/<id>/model
  Running predictions...

  Prompt 1: What is MLflow model serving in one sentence?
  Response: MLflow model serving wraps logged models in a REST API...
  Tokens used: 42

  Prompt 2: Name three benefits of containerized ML deployments.
  Response: 1. Reproducibility - containers package all dependencies...
  Tokens used: 67

============================================================
Part 3: Log second version (temp=0.3) and register both
============================================================
  Model logged: runs:/<id>/model
  Default temperature: 0.3
  Registered version 1 (temp=0.7) as 'llm-serving-demo'
  Registered version 2 (temp=0.3) as 'llm-serving-demo'
  Alias 'champion' -> version 1 (creative, temp=0.7)
  Alias 'challenger' -> version 2 (deterministic, temp=0.3)

  Serve each version:
    mlflow models serve -m models:/llm-serving-demo@champion -p 5001
    mlflow models serve -m models:/llm-serving-demo@challenger -p 5002

============================================================
Part 4: Docker containerization
============================================================
  Build a Docker image for the champion model:
    mlflow models build-docker \
      --model-uri models:/llm-serving-demo@champion \
      --name mlflow-llm-server

  Run the container (pass LMStudio host for LLM access):
    podman run -p 5001:8080 \
      --add-host=host.containers.internal:host-gateway \
      mlflow-llm-server

  Send a prediction request:
    curl -X POST http://localhost:5001/invocations \
      -H "Content-Type: application/json" \
      -d '{"dataframe_split": {
            "columns": ["prompt", "temperature", "max_tokens"],
            "data": [["What is MLflow?", 0.7, 128]]
          }}'

  Input format options:
    dataframe_split: {"columns": [...], "data": [[...]]}
    dataframe_records: [{"prompt": "...", "temperature": 0.7}]

============================================================
Part 5: Health check endpoints and monitoring
============================================================
  GET /ping                 Liveness check. Returns 200 when server is ready.
  GET /health               Alias for /ping. Use in Kubernetes readiness probes.
  GET /version              Returns MLflow version. Useful for debugging.
  POST /invocations         Prediction endpoint. Accepts JSON or CSV.

  Example health check commands:
    curl http://localhost:5001/ping
    curl http://localhost:5001/health
    curl http://localhost:5001/version

  Kubernetes probe configuration:
    livenessProbe:
      httpGet: { path: /ping, port: 8080 }
    readinessProbe:
      httpGet: { path: /health, port: 8080 }

  Monitoring best practices:
    1. Track request latency (p50, p95, p99) via reverse proxy or sidecar.
    2. Log prediction counts and error rates to Prometheus.
    3. Monitor token usage per request to control LLM costs.
    4. Set up alerts for latency spikes or error rate > threshold.
    5. Track model version staleness -- when was the served version updated?

============================================================
Done! Check the MLflow UI at http://127.0.0.1:5000
  Experiment: L2/M8_deployment/1_serving_deep_dive
  Registered model: llm-serving-demo (2 versions)
============================================================
```

In the MLflow UI you will see:

- Experiment **L2/M8_deployment/1_serving_deep_dive** with three runs
- Two LLM model runs with different default temperatures (0.7 and 0.3)
- Both versions registered under **llm-serving-demo** with `champion` and `challenger` aliases
- Model signatures showing the prompt/temperature/max_tokens input schema and response/tokens_used output schema

## Key Takeaways

- Wrapping an LLM in `mlflow.pyfunc.PythonModel` gives you a standardized serving interface that works with `mlflow models serve` and Docker deployment out of the box.
- Use `load_context` to initialize expensive resources (like the OpenAI client) once at model load time, not on every prediction call.
- Logging multiple model versions with different configurations (temperature, model name) lets you A/B test and roll back via the Model Registry.
- For containerized LLM models, ensure the container has network access to the LLM provider -- use host networking or pass the provider URL as a configuration.
- Monitor token usage per request alongside latency and error rates to control LLM serving costs in production.

## Next Steps

In **L2-8.2 -- Batch Prediction Pipelines** you will build automated batch inference scripts that load models from the registry, process large datasets, and log prediction results back to MLflow for tracking and auditing.
