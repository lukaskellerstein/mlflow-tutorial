# L1-1.1 — What is MLflow? Architecture Overview

**Level:** Essentials
**Duration:** 20 min

## Overview

This lesson introduces MLflow's architecture and the five core pillars that make up the platform. You will connect to a running MLflow server, create your first experiment run, and learn the key building blocks (experiments, runs, parameters, metrics, artifacts, and tags) that every subsequent lesson builds on.

## Prerequisites

- MLFlow server running at http://127.0.0.1:5000 (start with `podman compose up -d` from `infra/`)
- Python 3.10+
- `uv` package manager installed

## Concepts

### The 5 Pillars of MLflow

MLflow is an open-source platform for managing the complete machine learning and AI lifecycle. It is organized around five core pillars:

1. **Tracking** — Record and query parameters, metrics, code versions, and artifacts for every experiment run. This is the foundation everything else builds on.

2. **Models** — Package ML and AI models from any framework (scikit-learn, PyTorch, LangChain, custom Python functions) in a standard format that MLflow can deploy.

3. **Model Registry** — A centralized model store with versioning, stage transitions (e.g., staging to production), and access control. Think of it as "git for models."

4. **Evaluation** — Evaluate model quality using built-in metrics (accuracy, toxicity, faithfulness) and custom metrics. Essential for LLM and agent evaluation.

5. **Deployment** — Serve models as REST APIs for real-time inference or run batch predictions. Includes the AI Gateway for routing requests across LLM providers.

### Key Concepts

| Concept | Description |
|------------|-------------|
| **Experiment** | A named collection of runs — typically one per project or task. |
| **Run** | A single execution that stores parameters, metrics, artifacts, and tags. |
| **Parameters** | Input configuration values (model name, temperature, learning rate). |
| **Metrics** | Numeric results to track over time (accuracy, latency, cost). |
| **Artifacts** | Output files — trained models, plots, data snapshots. |
| **Tags** | Free-form key/value metadata for organizing and filtering runs. |

### Architecture at a Glance

```
 Your Code (main.py)
       |
       v
 MLflow Client (Python SDK)
       |
       v
 Tracking Server (http://127.0.0.1:5000)
       |
       v
 Backend Store (PostgreSQL)  +  Artifact Store (local / S3)
```

The MLflow client in your Python code sends data to the tracking server over HTTP. The server persists run metadata in a backend store (PostgreSQL in our setup) and stores artifacts (models, files) in an artifact store.

## Step-by-Step

### Step 1: Connect to MLflow

We set the tracking URI so the MLflow client knows where to send data, then create (or reuse) an experiment:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M1_core_platform/1_architecture_overview")
```

### Step 2: Create a Run and Log Data

A run is the fundamental unit of work in MLflow. We log parameters (inputs), metrics (outputs), and tags (metadata):

```python
with mlflow.start_run(run_name="architecture_overview") as run:
    mlflow.log_params({"framework": "mlflow", "version": mlflow.__version__, "lesson": "L1-1.1"})
    mlflow.log_metric("setup_complete", 1.0)
    mlflow.set_tags({"level": "1", "module": "core_platform", "lesson": "architecture_overview"})
```

### Step 3: Verify in the UI

Open http://127.0.0.1:5000 in your browser. Navigate to the experiment `L1/M1_core_platform/1_architecture_overview`. Click on the run to see its parameters, metrics, and tags.

## Running the Lesson

```bash
cd tutorial/level_1/M1_core_platform/1_architecture_overview
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
MLflow Architecture — The 5 Pillars
============================================================
  1. Tracking
    Record parameters, metrics, and artifacts for every experiment run.
  2. Models
    Package ML/AI models in a standard format for any framework.
  3. Model Registry
    Centralized model store with versioning and lifecycle management.
  4. Evaluation
    Evaluate model quality with built-in and custom metrics.
  5. Deployment
    Serve models as REST APIs or batch-process predictions.

============================================================
Key Concepts
============================================================
  Experiment   — A named collection of runs (e.g., one per project or task).
  Run          — A single execution — stores params, metrics, artifacts, and tags.
  Parameters   — Input configuration values (model name, learning rate, ...).
  Metrics      — Numeric results you want to track (accuracy, latency, ...).
  Artifacts    — Output files — models, plots, data snapshots.
  Tags         — Free-form key/value metadata for organizing and filtering runs.

============================================================
Creating a run on the MLflow server
============================================================
  Tracking URI : http://127.0.0.1:5000
  Experiment   : L1/M1_core_platform/1_architecture_overview
  MLflow version: 2.x.x

  Run ID   : <generated-run-id>
  Status   : RUNNING
  Artifact URI: <server-artifact-path>

============================================================
Done!
============================================================
  Open the MLflow UI at http://127.0.0.1:5000
  Navigate to experiment: L1/M1_core_platform/1_architecture_overview
  You should see the run with parameters, metrics, and tags.
```

## Key Takeaways

- MLflow is built on five pillars: Tracking, Models, Model Registry, Evaluation, and Deployment.
- The **experiment** groups related runs; the **run** is the unit of work that stores params, metrics, artifacts, and tags.
- The MLflow Python SDK communicates with a tracking server over HTTP -- you just set the URI and start logging.
- Every lesson in this tutorial follows the same pattern: set the tracking URI, create an experiment, and log results.

## Next Steps

Continue to **L1-1.2 Tracking Basics** where you will dive deeper into the Tracking pillar -- logging parameters, metrics, and artifacts across multiple runs, and learning to compare results in the MLflow UI.
