# L1-4.1 — Traditional ML Evaluation

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Learn how to use `mlflow.models.evaluate()` to assess traditional ML models. Instead of manually computing metrics and logging them one by one, this API automatically calculates a comprehensive set of metrics and generates visual artifacts like confusion matrices and ROC curves -- all logged to MLflow in a single call.

> **Note:** In MLflow 3.x, the traditional ML evaluation API lives at `mlflow.models.evaluate()`. For LLM/GenAI evaluation, use `mlflow.genai.evaluate()` instead (covered in L1-4.2).

## Prerequisites

- Completed: L1-1.2 (Tracking Basics), L1-2.1 (Models & Flavors)
- MLFlow server running at http://127.0.0.1:5000
- Basic familiarity with scikit-learn classifiers

## Concepts

### Why Use mlflow.models.evaluate()?

In L1-1.2 you logged metrics manually with `mlflow.log_metric()`. That works, but it has drawbacks:

- You have to remember which metrics to compute.
- You have to create plots (confusion matrix, ROC) yourself.
- There is no standardized way to compare models across runs.

`mlflow.models.evaluate()` solves all of this. You pass it a logged model and evaluation data, and it automatically:

1. Runs predictions on the data.
2. Computes a standard set of metrics for the model type.
3. Generates evaluation artifacts (plots, tables).
4. Logs everything to the active MLflow run.

### Built-in Metrics for Classifiers

For **multiclass classifiers**, `mlflow.models.evaluate()` computes:

| Metric | Description |
|--------|-------------|
| `accuracy_score` | Fraction of correct predictions |
| `f1_score` | Weighted F1 score across classes |
| `precision_score` | Weighted precision across classes |
| `recall_score` | Weighted recall across classes |
| `log_loss` | Logarithmic loss (lower is better) |
| `roc_auc` | Area under the ROC curve |
| `example_count` | Number of evaluation samples |

It also generates **per-class metrics** (precision, recall, ROC AUC for each class) saved as a CSV artifact.

For **binary classifiers**, you additionally get: `true_positives`, `false_positives`, `true_negatives`, `false_negatives`, `precision_recall_auc`, plus dedicated ROC and precision-recall plots.

### Evaluation Artifacts

The default evaluator generates visual artifacts automatically:

- **Confusion matrix** — a heatmap of predicted vs. actual labels
- **ROC curves** — one curve per class (multiclass) or a single curve (binary)
- **Precision-Recall curves** — per-class or single curve
- **Per-class metrics CSV** — detailed metrics broken down by class

All artifacts are logged to the MLflow run and visible in the UI.

## Step-by-Step

### Step 1: Prepare Evaluation Data

`mlflow.models.evaluate()` expects a pandas DataFrame with both feature columns and a target column:

```python
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=42
)

eval_df = pd.DataFrame(X_test, columns=wine.feature_names)
eval_df["target"] = y_test
```

### Step 2: Train and Log a Model

Train the model and log it so `mlflow.models.evaluate()` can load it for predictions:

```python
with mlflow.start_run(run_name="random_forest"):
    model.fit(X_train, y_train)
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
```

### Step 3: Evaluate with mlflow.models.evaluate()

Pass the model URI, data, target column name, and model type:

```python
result = mlflow.models.evaluate(
    model=model_info.model_uri,
    data=eval_df,
    targets="target",
    model_type="classifier",
    evaluators="default",
)
```

### Step 4: Access Results

The `EvaluationResult` object contains metrics and artifacts:

```python
# Metrics as a dictionary
print(result.metrics)
# e.g. {'accuracy_score': 0.98, 'f1_score_macro': 0.97, ...}

# Artifacts (plots, tables)
for name, artifact in result.artifacts.items():
    print(name)
```

### Step 5: Compare Models

Run `mlflow.models.evaluate()` for each model variant in its own run, then compare metrics side by side in the MLflow UI or programmatically.

## Running the Lesson

```bash
cd tutorial/level_1/M4_evaluation/1_traditional_ml_eval
uv sync
uv run python main.py
```

## Expected Output

The script trains and evaluates two classifiers on the Wine dataset:

```
Step 1: Loading the Wine dataset
  Training samples: 124
  Evaluation samples: 54

Step 2: Evaluating RandomForest classifier
  Model logged: runs:/<run_id>/model
  Evaluation metrics for random_forest:
              accuracy_score: 0.9815
              example_count: 54
             f1_score_macro: 0.9802
             f1_score_micro: 0.9815
                   log_loss: 0.1234
                      score: 0.9815

Step 3: Evaluating GradientBoosting classifier
  ...similar output...

Step 4: Model comparison
  Metric                     RandomForest  GradientBoost
  ...table comparing key metrics with arrows showing the winner...
```

In the MLflow UI you will see:
- Two runs in the experiment, each with logged metrics
- Evaluation artifacts: confusion matrix, ROC curves, precision-recall curves
- A per-class metrics CSV file

## Key Takeaways

- `mlflow.models.evaluate()` automates metric computation and artifact generation for ML models.
- Set `model_type="classifier"` (or `"regressor"`) to get the appropriate built-in metrics.
- Evaluation data must be a pandas DataFrame with feature columns and a target column.
- Results include both a `metrics` dict and an `artifacts` dict with plots and tables.
- Use separate runs for each model to compare evaluation results in the MLflow UI.

## Next Steps

In L1-4.2 (LLM Evaluation Basics), you will apply `mlflow.evaluate()` to large language models, using LLM-specific metrics like toxicity and readability. In Level 2, we will explore this in more depth with custom metrics and RAG evaluation.
