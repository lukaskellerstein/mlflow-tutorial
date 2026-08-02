# L1-M6.1 — Model Serving

**Level:** Essentials
**Duration:** 35 min

## Overview

MLflow can serve any logged model as a REST API with a single CLI command. This lesson covers the full serving lifecycle: wrapping an LLM in a PythonModel with efficient initialization via `load_context`, logging multiple versions with different configurations, testing locally, Docker containerization, and production health checks.

## Prerequisites

- Completed: L1-M3 (Models and Flavors, Model Registry)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### PythonModel with `load_context`

By placing client initialization in `load_context()` instead of `predict()`, the LLM client is created once at model load time rather than on every request. This is critical for serving performance.

### Serving Architecture

```
Client (curl / app)       MLflow Serving Process
   |                          |
   |  POST /invocations       |
   | -----------------------> |
   |                          |  load_context() (once at startup)
   |                          |  predict(input) per request
   |  JSON response           |
   | <----------------------- |
```

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/invocations` | POST | Run predictions |
| `/ping` | GET | Health check (returns 200) |
| `/health` | GET | Alias for /ping |
| `/version` | GET | MLflow version info |

### Input Formats

The `/invocations` endpoint accepts JSON in two formats:

**dataframe_split** (recommended):

```json
{"dataframe_split": {"columns": ["question"], "data": [["What is MLflow?"]]}}
```

**instances**:

```json
{"instances": [{"question": "What is MLflow?"}]}
```

## Step-by-Step

### Step 1: Define a PythonModel with `load_context`

```python
class LLMChatModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name, default_temperature):
        self.model_name = model_name
        self.default_temperature = default_temperature

    def load_context(self, context):
        from openai import OpenAI

        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def predict(self, context, model_input, params=None):
        # Iterate rows, call LLM, return list of answers
        ...
```

### Step 2: Log and Register the Model

```python
mlflow.pyfunc.log_model(
    name="model",
    python_model=LLMChatModel(model_name="google/gemma-4-e4b", default_temperature=0.7),
    signature=signature,
    input_example=input_example,
    registered_model_name="L1-llm-serving-demo",
    pip_requirements=["openai>=1.0", "mlflow>=2.0"],
)
```

### Step 3: Test Locally

Verify the model works by loading it in-process before serving:

```python
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(pd.DataFrame({"question": ["What is AI?"]}))
```

### Step 4: Register Multiple Versions with Aliases

Log a second version with a different temperature and assign aliases:

```python
client.set_registered_model_alias("L1-llm-serving-demo", "champion", v1)
client.set_registered_model_alias("L1-llm-serving-demo", "challenger", v2)
```

### Step 5: Serve and Deploy

```bash
mlflow models serve -m "models:/L1-llm-serving-demo@champion" --port 5001 --no-conda
```

For Docker:

```bash
mlflow models build-docker --model-uri models:/L1-llm-serving-demo@champion --name mlflow-llm-server
podman run -p 5001:8080 --add-host=host.containers.internal:host-gateway mlflow-llm-server
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M6_deployment_gateway/1_model_serving
uv sync
uv run python main.py
```

Note: The script logs models, tests locally, and prints serving commands. It does not start a server process. Follow the printed instructions to try serving.

## Expected Output

```
============================================================
Part 1: Log LLM Model (temperature=0.7)
============================================================
  Model signature:
  ...
  Model logged: runs:/<id>/model
  Registered as: L1-llm-serving-demo

============================================================
Part 2: Test the Model Locally (No Server)
============================================================
  Q: What is MLflow?
  A: MLflow is an open-source platform...

============================================================
Part 3: Model Versions and Aliases
============================================================
  Alias 'champion'   -> v1 (creative, temp=0.7)
  Alias 'challenger' -> v2 (deterministic, temp=0.3)

============================================================
Part 4: Serving as REST API + Docker Deployment
============================================================
  Start serving:
    mlflow models serve ...
  Example prediction request:
    curl ...

============================================================
Part 5: Health Checks and Monitoring
============================================================
  curl http://localhost:5001/ping
  ...
```

## Key Takeaways

- `mlflow.pyfunc.PythonModel` wraps any Python code -- including LLM API calls -- as a servable MLflow model
- Use `load_context()` to initialize expensive resources (LLM client) once at model load time
- `mlflow models serve` turns any logged model into a REST API with zero application code
- Multiple model versions with different configurations enable A/B testing via the Model Registry
- `mlflow models build-docker` packages the model into a container for cloud deployment
- Standard health check endpoints integrate with Kubernetes probes and load balancers

## Next Steps

Continue to L1-M6.2 (Batch Prediction) to learn about running models against large datasets from the command line, or skip to L1-M6.3 (AI Gateway) for unified multi-provider LLM management.
