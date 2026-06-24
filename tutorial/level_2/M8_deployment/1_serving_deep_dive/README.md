# L2-8.1 — Model Serving Deep Dive

**Level:** Practitioner
**Duration:** 1 hour

## Overview

Go beyond basic `mlflow models serve` and learn production serving patterns: advanced CLI configuration, custom PyFunc models with embedded preprocessing, Docker containerization, and health check endpoints. This lesson prepares models and documents patterns without starting a live server, so you can study the full serving surface area.

## Prerequisites

- Completed: L1-8.1 Model Serving, L1-2.1 Models & Flavors, L2-2.2 Custom PyFunc Models
- MLFlow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (uses scikit-learn only)

## Concepts

### Serving Architecture

MLflow model serving wraps your logged model in a REST API powered by Flask (default) or MLServer. The server loads the model artifact, validates incoming requests against the model signature, runs inference, and returns predictions. Understanding the full configuration surface lets you tune performance, handle complex inputs, and deploy reliably.

### Serving Configurations

The `mlflow models serve` CLI accepts options that control performance and behavior:

| Option | Purpose |
|--------|---------|
| `--host` | Bind address (default `127.0.0.1`, use `0.0.0.0` for containers) |
| `--port` | Port number (default `5000`) |
| `--workers` | Number of Gunicorn workers for parallel request handling |
| `--timeout` | Request timeout in seconds |
| `--no-conda` | Skip Conda environment creation (use current env) |
| `--enable-mlserver` | Use MLServer (Seldon) instead of Flask for advanced features |

### Custom PyFunc for Serving

When your model needs preprocessing (scaling, feature engineering) or custom output formatting, wrap everything in a `mlflow.pyfunc.PythonModel`. The `predict()` method receives raw client input and can transform it before passing to the underlying model. This keeps the serving contract clean: clients send raw data, the model handles the rest.

### Docker Containerization

`mlflow models build-docker` packages your model, its dependencies, and the serving infrastructure into a Docker image. The resulting container exposes the same REST endpoints and can be deployed to Kubernetes, ECS, or any container runtime.

### Health Check Endpoints

Every MLflow model server exposes endpoints for orchestration:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/invocations` | POST | Prediction requests |
| `/ping` | GET | Liveness probe (returns 200) |
| `/health` | GET | Alias for `/ping` |
| `/version` | GET | MLflow version info |

## Step-by-Step

### Step 1: Prepare multiple models for serving

Train a RandomForest and GradientBoosting classifier on the wine dataset. Log each with a proper signature and input example, then register them in the model registry:

```python
signature = infer_signature(df_test, preds)
input_example = df_test.head(3)

info = mlflow.sklearn.log_model(
    model, name="model",
    signature=signature,
    input_example=input_example,
)
mlflow.register_model(info.model_uri, f"serving-demo-{name}")
```

The signature enables request validation at serving time, and the input example generates a sample request in the MLflow UI.

### Step 2: Explore serving configurations

Review the CLI options for `mlflow models serve` and the equivalent environment variables. A serving configuration JSON is logged as an artifact for reference:

```python
SERVING_CONFIG = {
    "model_uri": "models:/serving-demo-RandomForest/1",
    "host": "0.0.0.0",
    "port": 5001,
    "workers": 4,
    "timeout": 120,
}
```

### Step 3: Build a custom PyFunc with preprocessing

Create a `PythonModel` that bundles a StandardScaler and a classifier. The `predict()` method scales raw inputs before classification and returns both class IDs and human-readable names:

```python
class WineClassifierWithPreprocessing(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.scaler = pickle.load(open(context.artifacts["scaler"], "rb"))
        self.classifier = pickle.load(open(context.artifacts["classifier"], "rb"))

    def predict(self, context, model_input, params=None):
        scaled = self.scaler.transform(model_input)
        class_ids = self.classifier.predict(scaled)
        return pd.DataFrame({"class_id": class_ids, "class_name": ...})
```

### Step 4: Docker containerization

Review the `mlflow models build-docker` command and the input format options (dataframe_split, dataframe_records, instances). A deployment guide is logged as an artifact.

### Step 5: Health checks and monitoring

Review the four serving endpoints and monitoring best practices for production deployments (latency tracking, error rates, model staleness).

## Running the Lesson

```bash
cd tutorial/level_2/M8_deployment/1_serving_deep_dive
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Preparing multiple models for serving
============================================================
  RandomForest              accuracy=1.0000  uri=runs:/<id>/model
  GradientBoosting          accuracy=0.9444  uri=runs:/<id>/model

============================================================
Part 2: Serving configurations
============================================================
  CLI command:
    mlflow models serve \
      --model-uri models:/serving-demo-RandomForest/1 \
      --host 0.0.0.0 \
      --port 5001 \
      --workers 4 \
      --timeout 120 \
      --no-conda

  Environment variable equivalents:
    export MLFLOW_MODEL_URI=models:/serving-demo-RandomForest/1
    export MLFLOW_HOST=0.0.0.0
    export MLFLOW_PORT=5001
    export MLFLOW_WORKERS=4

  Serving config logged as artifact: serving_config.json

============================================================
Part 3: Custom PyFunc with preprocessing
============================================================
  Custom PyFunc model URI: runs:/<id>/custom_model
  Sample prediction:
  class_id class_name
         0    class_0
         ...

============================================================
Part 4: Docker containerization
============================================================
  Build a Docker image for a registered model:
    mlflow models build-docker \
      --model-uri models:/serving-demo-RandomForest/1 \
      --name mlflow-wine-server

  Run the container:
    podman run -p 5001:8080 mlflow-wine-server

  Deployment guide logged as artifact: deployment_guide.md

============================================================
Part 5: Health checks and monitoring endpoints
============================================================
  GET /ping                 Liveness check. Returns 200 ...
  GET /health               Alias for /ping. ...
  GET /version              Returns MLflow version. ...
  POST /invocations         Prediction endpoint. ...

  Monitoring best practices:
    1. Track request latency ...
    2. Log prediction counts ...
    ...

============================================================
Done! Check the MLflow UI at http://127.0.0.1:5000
  Experiment: L2/M8_deployment/1_serving_deep_dive
  Registered models: serving-demo-RandomForest,
    serving-demo-GradientBoosting, serving-demo-CustomPyFunc
============================================================
```

In the MLflow UI you will see:

- Experiment **L2/M8_deployment/1_serving_deep_dive** with four runs
- Two sklearn model runs (RandomForest, GradientBoosting) with signatures and input examples
- A serving_config run with a JSON artifact
- A custom_pyfunc_serving run with the bundled scaler+classifier model
- A deployment_guide run with a markdown deployment reference
- Three registered models in the Model Registry

## Key Takeaways

- Always log models with signatures and input examples — they enable request validation and self-documenting APIs at serving time.
- Use `--workers` to scale serving throughput via multiple Gunicorn workers.
- Custom PyFunc models let you embed preprocessing, postprocessing, and multi-model ensembles behind a single serving endpoint.
- `mlflow models build-docker` creates production-ready containers with all dependencies baked in.
- The `/ping` and `/health` endpoints integrate directly with Kubernetes liveness and readiness probes.

## Next Steps

In **L2-8.2 -- Batch Prediction Pipelines** you will build automated batch inference scripts that load models from the registry, process large datasets, and log prediction results back to MLflow for tracking and auditing.
