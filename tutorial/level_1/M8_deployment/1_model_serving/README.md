# L1-8.1 — Model Serving Basics

**Level:** Essentials
**Duration:** 30 min

## Overview

MLflow can serve any logged model as a REST API with a single CLI command. This lesson covers how to prepare a model for serving, the available endpoints and input formats, batch prediction, and Docker containerization. You will also compare serving against programmatic (in-process) prediction.

## Prerequisites

- Completed: L1-M2 (Models and Registry)
- MLflow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson (uses scikit-learn)

## Concepts

### Why Model Serving?

Training a model is only half the story. To use it in an application you need to make it accessible. MLflow provides two main approaches:

1. **Real-time serving** — expose the model as a REST API so any language or service can call it over HTTP.
2. **Batch prediction** — run the model against a file of inputs from the command line.

Both approaches work with any MLflow model flavor (sklearn, PyTorch, pyfunc, LangChain, etc.) without writing any serving code.

### Serving Architecture

```
Client (curl / app)       MLflow Serving Process
   |                          |
   |  POST /invocations       |
   | -----------------------> |
   |                          |  Load model from artifact store
   |                          |  Run model.predict(input)
   |  JSON predictions        |
   | <----------------------- |
```

The serving process loads the model once at startup and handles prediction requests.

### Endpoints

| Endpoint        | Method | Purpose                    |
|-----------------|--------|----------------------------|
| `/invocations`  | POST   | Run predictions            |
| `/ping`         | GET    | Health check (returns 200) |
| `/version`      | GET    | MLflow version info        |

### Input Formats

The `/invocations` endpoint accepts JSON in two formats:

**dataframe_split** (recommended):
```json
{
  "dataframe_split": {
    "columns": ["feature_0", "feature_1", "feature_2", "feature_3"],
    "data": [[5.1, 3.5, 1.4, 0.2]]
  }
}
```

**instances**:
```json
{
  "instances": [
    {"feature_0": 5.1, "feature_1": 3.5, "feature_2": 1.4, "feature_3": 0.2}
  ]
}
```

Both produce the same result. The `dataframe_split` format is more compact for many rows.

## Step-by-Step

### Step 1: Train and Register a Model

We train a RandomForest on the Iris dataset and log it with a **signature** and **input example**. The signature tells the serving layer what input shape and types to expect. The input example gives documentation and testing data.

```python
signature = infer_signature(X_test, clf.predict(X_test))
mlflow.sklearn.log_model(
    clf,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
    registered_model_name="iris-classifier-serving-demo",
)
```

### Step 2: Serve the Model

After logging and registering, serve it with one command:

```bash
mlflow models serve -m "models:/iris-classifier-serving-demo/1" --port 5001 --no-conda
```

This starts a local REST server on port 5001. The `--no-conda` flag skips Conda environment creation (use your current environment instead).

### Step 3: Call the Serving Endpoints

Health check:
```bash
curl http://127.0.0.1:5001/ping
```

Prediction:
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["feature_0","feature_1","feature_2","feature_3"], "data": [[5.1,3.5,1.4,0.2]]}}'
```

### Step 4: Programmatic Prediction (Alternative)

For Python scripts and notebooks, you can skip serving entirely and load the model directly:

```python
model = mlflow.pyfunc.load_model("models:/iris-classifier-serving-demo/1")
predictions = model.predict(sample_dataframe)
```

This is simpler but only works within Python.

### Step 5: Batch Prediction

For processing a file of inputs without writing code:

```bash
mlflow models predict -m "models:/iris-classifier-serving-demo/1" -i input.csv
```

### Step 6: Docker Containerization

Package the model as a Docker image for deployment anywhere:

```bash
mlflow models build-docker -m "models:/iris-classifier-serving-demo/1" -n "iris-server"
docker run -p 5001:8080 iris-server
```

The container includes the model, its dependencies, and the serving layer. It exposes the same `/invocations`, `/ping`, and `/version` endpoints.

## Running the Lesson

```bash
cd tutorial/level_1/M8_deployment/1_model_serving
uv sync
uv run python main.py
```

Note: The script trains, logs, registers the model, and runs programmatic predictions. It prints the CLI commands for serving and batch prediction but does not start a server process. Follow the printed instructions to try serving yourself.

## Expected Output

In the terminal you will see:
- Model training with accuracy around 1.0 (Iris is a simple dataset)
- The model URI and registered model name
- CLI commands for serving, curl requests, and batch prediction
- Programmatic predictions (class labels 0, 1, or 2)

In the MLflow UI at http://127.0.0.1:5000:
- Experiment "L1/M8_deployment/1_model_serving" with two runs
- The first run contains the logged model with signature and input example
- The second run contains the batch input CSV artifact

## Key Takeaways

- `mlflow models serve` turns any logged model into a REST API with zero application code.
- Models need a **signature** and **input example** for reliable serving.
- The `/invocations` endpoint accepts JSON in `dataframe_split` or `instances` format.
- `mlflow.pyfunc.load_model()` is the in-process alternative when you don't need HTTP.
- `mlflow models build-docker` packages the model into a portable container.

## Next Steps

Continue to L1-8.2 (AI Gateway Overview) to learn how MLflow can route requests across multiple LLM providers with rate limiting, fallbacks, and unified API access. In Level 2, we will explore advanced serving patterns including custom endpoints and deployment strategies.
