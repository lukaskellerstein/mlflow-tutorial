# L1-M1.1 — Your First MLflow Run

**Level:** Essentials
**Duration:** 15 min

## Overview

This lesson introduces MLflow's five core pillars and the key concepts behind experiment tracking. You will connect to a running MLflow server, call a local LLM via LMStudio, and log everything — parameters, metrics, and tags — as your first MLflow run.

## Prerequisites

- MLflow server running at http://127.0.0.1:5000 (start with `podman compose up -d` from `infra/`)
- LMStudio running with `google/gemma-4-e4b` loaded (`lms load google/gemma-4-e4b --gpu max -y`)
- LMStudio server started (`lms server start`)
- Python 3.10+
- `uv` package manager installed

## Concepts

### The 5 Pillars of MLflow

MLflow is an open-source platform for managing the complete machine learning and AI lifecycle:

1. **Tracking** — Record and query parameters, metrics, code versions, and artifacts for every experiment run. This is the foundation everything else builds on.
2. **Models** — Package ML and AI models from any framework in a standard format.
3. **Model Registry** — A centralized model store with versioning and lifecycle management.
4. **Evaluation** — Evaluate model quality with built-in and custom metrics.
5. **Deployment** — Serve models as REST APIs or run batch predictions.

### Key Concepts

| Concept | Description |
|------------|-------------|
| **Experiment** | A named collection of runs — typically one per project or task. |
| **Run** | A single execution that stores parameters, metrics, artifacts, and tags. |
| **Parameters** | Input configuration values (model name, temperature, max tokens). |
| **Metrics** | Numeric results to track (response time, token count, latency). |
| **Artifacts** | Output files — models, responses, data snapshots. |
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

## Step-by-Step

### Step 1: Connect to MLflow

Set the tracking URI and create (or reuse) an experiment:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M1_tracking/1_first_run")
```

### Step 2: Call the LLM and Log Everything

Start a run, log the LLM configuration as parameters, make the call, then log the results as metrics:

```python
with mlflow.start_run(run_name="first_llm_call") as run:
    mlflow.log_params({"model": MODEL, "temperature": 0.7, "max_tokens": 256})
    result = call_llm(client, prompt, temperature, max_tokens)
    mlflow.log_metrics({
        "response_time_seconds": result["response_time_seconds"],
        "total_tokens": result["total_tokens"],
    })
    mlflow.set_tags({"level": "1", "module": "tracking", "lesson": "first_run"})
```

### Step 3: Verify in the UI

Open http://127.0.0.1:5000 in your browser. Navigate to the experiment `L1/M1_tracking/1_first_run`. Click on the run to see its parameters, metrics, and tags.

## Running the Lesson

```bash
cd tutorial/level_1/M1_tracking/1_first_run
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
MLflow — The 5 Pillars
============================================================
  1. Tracking
    Record parameters, metrics, and artifacts for every experiment run.
  ...

============================================================
Creating your first MLflow run
============================================================
  Tracking URI : http://127.0.0.1:5000
  Experiment   : L1/M1_tracking/1_first_run
  Model        : google/gemma-4-e4b
  Prompt       : Explain what MLflow is in 2 sentences.

  Run ID          : <generated-run-id>
  Response time   : 1.234s
  Tokens (total)  : 42
  Finish reason   : stop

  LLM response:
    <LLM's response about MLflow>

============================================================
Done!
============================================================
```

## Key Takeaways

- MLflow is built on five pillars: Tracking, Models, Model Registry, Evaluation, and Deployment.
- The **experiment** groups related runs; the **run** stores params, metrics, artifacts, and tags.
- LLM call configuration (model, temperature, max_tokens) maps naturally to MLflow **parameters**.
- LLM call results (response time, token counts) map naturally to MLflow **metrics**.
- The MLflow Python SDK communicates with a tracking server over HTTP — set the URI and start logging.

## Next Steps

Continue to **L1-M1.2 Tracking Basics** where you will track multiple LLM calls, compare configurations, log artifacts, and learn step-based metric logging.
