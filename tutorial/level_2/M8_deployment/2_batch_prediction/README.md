# L2-8.2 -- Batch Prediction Pipelines

**Level:** Practitioner
**Duration:** 45 min

## Overview

Batch prediction is how most ML models deliver value in production -- scoring large datasets on a schedule rather than one request at a time. This lesson builds a complete batch prediction pipeline with MLflow: loading a logged model, running predictions over a DataFrame, tracking results as artifacts and metrics, handling errors, and generating CLI commands for offline scoring.

## Prerequisites

- Completed: L1-8.1 (Model Serving Basics)
- Completed: L2-8.1 (Serving Deep Dive)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Batch vs. Real-Time Prediction

| Aspect | Real-Time (Serving) | Batch |
|--------|-------------------|-------|
| Latency | Milliseconds | Minutes to hours |
| Input | Single request | Large dataset |
| Trigger | API call | Schedule / event |
| Use case | User-facing apps | Reports, ETL, scoring |
| Scaling | Horizontal (replicas) | Vertical (bigger machine) |

Batch prediction is ideal when you need to score an entire dataset periodically -- nightly fraud scoring, weekly churn prediction, daily recommendation refresh.

### mlflow.pyfunc for Batch Scoring

`mlflow.pyfunc.load_model()` loads any MLflow model (regardless of flavor) into a unified prediction interface. This makes batch scripts framework-agnostic -- the same pipeline works whether the model is scikit-learn, XGBoost, PyTorch, or a custom PyFunc.

### Pipeline Pattern

A production batch pipeline follows a consistent flow:

1. **Load** -- read input data from a source (CSV, database, API)
2. **Validate** -- check for missing values, schema mismatches, data quality
3. **Predict** -- run the model, handle errors per-row or per-batch
4. **Log** -- save predictions, metrics, and metadata to MLflow

Tracking every batch run in MLflow gives you an audit trail: what model version was used, how many rows were scored, how long it took, and what the prediction distribution looked like.

## Step-by-Step

### Step 1: Train and Log a Model

We train a GradientBoostingClassifier on the wine dataset and log it with an inferred signature. The signature ensures that batch inputs are validated against the expected schema.

```python
signature = infer_signature(X_test, clf.predict(X_test))
mlflow.sklearn.log_model(clf, name="model", signature=signature)
```

### Step 2: Batch Prediction with mlflow.pyfunc

Load the model by its run URI and predict on a 60-row batch. We time the prediction to track throughput.

```python
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(batch_df)
```

### Step 3: Result Tracking

Log batch metrics (size, latency, throughput) and save predictions as a CSV artifact plus a JSON summary with class distribution.

```python
mlflow.log_metrics({"batch_size": 60, "predictions_per_second": 1500.0})
mlflow.log_artifact("batch_predictions.csv", artifact_path="predictions")
```

### Step 4: Pipeline Pattern

A four-step pipeline wraps the entire flow with validation and error handling:

1. Load data and log input row count
2. Validate inputs (drop rows with missing values)
3. Predict with try/except to catch model failures
4. Log output CSV and set a `pipeline_status` tag

### Step 5: CLI Batch Prediction

Generate `mlflow models predict` commands for offline scoring and create a sample input CSV as an artifact.

```bash
mlflow models predict -m "runs:/<run_id>/model" \
  -i sample_input.csv -o predictions.csv
```

## Running the Lesson

```bash
cd tutorial/level_2/M8_deployment/2_batch_prediction
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-8.2 -- Batch Prediction Pipelines
============================================================

  Wine dataset: 178 samples, 13 features
  Train: 142 | Test: 36

============================================================
Part 1: Train and Log a Model
============================================================
  Accuracy : 0.97xx
  Model URI: runs:/<run_id>/model

============================================================
Part 2: Batch Prediction with mlflow.pyfunc
============================================================
  Loaded model from: runs:/<run_id>/model
  Batch size: 60 rows
  Prediction time : 0.00xxs
  Predictions/sec : xxxxx.x
  Unique classes  : [0, 1, 2]

============================================================
Part 3: Result Tracking
============================================================
  Logged metrics: batch_size=60, time=0.00xxs, pps=xxxxx.x
  Logged artifact: predictions/batch_predictions.csv
  Logged artifact: predictions/prediction_summary.json
  Class distribution: {0: xx, 1: xx, 2: xx}

============================================================
Part 4: Complete Pipeline Pattern
============================================================
  [1/4] Loading data ...
  [2/4] Validating inputs ...
        178 valid, 0 skipped (0 missing values)
  [3/4] Running predictions ...
        178 predicted, 0 failed in 0.00xxs
  [4/4] Logging results ...
        Artifact: pipeline/pipeline_output.csv
        Status: success

============================================================
Part 5: CLI Batch Prediction
============================================================
  Logged sample input: cli/sample_input.csv
  ...CLI commands...

============================================================
Done!
============================================================
```

In the MLflow UI you will see four runs under the experiment:
- **train_wine_model** -- the trained model with accuracy metric
- **batch_prediction_results** -- batch metrics and prediction artifacts
- **batch_pipeline** -- the full pipeline run with validation metrics
- **cli_batch_setup** -- sample input file for CLI scoring

## Key Takeaways

- `mlflow.pyfunc.load_model()` provides a unified interface for batch scoring regardless of the underlying model flavor.
- Log batch metrics (size, latency, throughput) alongside predictions to build an operational audit trail.
- Save prediction results as CSV artifacts for downstream consumption and debugging.
- Wrap pipelines in validation and error handling -- production data is messy.
- Use `mlflow models predict` for simple CLI-based batch scoring without writing Python code.

## Next Steps

Move on to L2-M9 (Framework Integrations) to see how MLflow integrates with PyTorch, Hugging Face, and Sentence Transformers for deeper model tracking. In Level 3, you will build production batch pipelines with scheduling, monitoring, and CI/CD quality gates.
