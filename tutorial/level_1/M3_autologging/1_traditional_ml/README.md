# L1-3.1 — Traditional ML Autologging

**Level:** Essentials
**Duration:** ~20 minutes

## Overview

MLflow can automatically log parameters, metrics, models, and artifacts when you train models with supported frameworks. This lesson demonstrates autologging with scikit-learn and XGBoost -- two of the most popular traditional ML libraries -- and shows how `mlflow.autolog()` enables all frameworks at once.

## Prerequisites

- Completed: L1-1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000
- Ollama is **not** needed for this lesson (traditional ML only)

## Concepts

### What is Autologging?

Normally, you call `mlflow.log_param()`, `mlflow.log_metric()`, etc. manually. **Autologging** patches the training methods of supported frameworks so that MLflow captures everything automatically -- no manual logging calls needed.

### What Gets Auto-Logged?

When autologging is enabled, MLflow captures:

| Category | What's Logged |
|----------|---------------|
| **Parameters** | All constructor hyperparameters (e.g., `n_estimators`, `max_depth`) |
| **Metrics** | Training score and, where applicable, validation metrics |
| **Model** | The serialized model as an MLflow artifact |
| **Artifacts** | Framework-specific extras (e.g., feature importance plots for XGBoost) |

### Supported Frameworks

MLflow supports autologging for many frameworks, including:

- **scikit-learn** -- `mlflow.sklearn.autolog()`
- **XGBoost** -- `mlflow.xgboost.autolog()`
- **LightGBM** -- `mlflow.lightgbm.autolog()`
- **PyTorch** -- `mlflow.pytorch.autolog()`
- **TensorFlow/Keras** -- `mlflow.tensorflow.autolog()`
- **Hugging Face Transformers** -- `mlflow.transformers.autolog()`
- **LangChain** -- `mlflow.langchain.autolog()`
- **OpenAI** -- `mlflow.openai.autolog()`

Or enable all at once with `mlflow.autolog()`.

## Step-by-Step

### Step 1: Enable sklearn Autologging

One call to `mlflow.sklearn.autolog()` patches scikit-learn so all `fit()` calls are automatically tracked:

```python
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="sklearn_autolog"):
    clf = RandomForestClassifier(n_estimators=50, max_depth=4)
    clf.fit(X_train, y_train)
```

After `fit()`, MLflow has logged every hyperparameter, the training score, and the serialized model -- without any manual `log_param` or `log_metric` calls.

### Step 2: Enable XGBoost Autologging

The same pattern works for XGBoost:

```python
mlflow.xgboost.autolog()

with mlflow.start_run(run_name="xgboost_autolog"):
    xgb = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
    xgb.fit(X_train, y_train)
```

XGBoost autologging captures additional artifacts like feature importance.

### Step 3: Universal Autolog

If you use multiple frameworks, enable everything at once:

```python
mlflow.autolog()

with mlflow.start_run(run_name="universal_autolog"):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
```

`mlflow.autolog()` enables autologging for every supported framework in one call.

## Running the Lesson

```bash
cd tutorial/level_1/M3_autologging/1_traditional_ml
uv sync
uv run python main.py
```

## Expected Output

You should see three sections in the terminal, each showing the parameters, metrics, and artifacts that were auto-logged:

```
Part 1: sklearn Autologging
  RandomForest accuracy: 0.9667
  Auto-logged details for: sklearn RandomForest
  Parameters (7): n_estimators, max_depth, random_state, ...
  Metrics (1): training_score
  Artifacts: model/

Part 2: XGBoost Autologging
  XGBoost accuracy: 0.9667
  Auto-logged details for: XGBoost XGBClassifier
  Parameters (20+): n_estimators, max_depth, learning_rate, ...
  Metrics: training metrics
  Artifacts: model/, feature_importance_weight.png, ...

Part 3: Universal Autolog
  LogisticRegression accuracy: 1.0000
  Auto-logged details for: LogisticRegression
  Parameters, metrics, and model all captured automatically.
```

In the MLflow UI at http://127.0.0.1:5000, you will find all three runs under the experiment "L1/M3_autologging/1_traditional_ml", each with full parameters, metrics, and model artifacts.

## Key Takeaways

- **`mlflow.<framework>.autolog()`** enables automatic logging for a specific framework.
- **`mlflow.autolog()`** enables autologging for all supported frameworks at once -- the easiest way to get started.
- Autologging captures hyperparameters, training metrics, and the serialized model with zero manual logging code.
- Different frameworks log different extras (e.g., XGBoost logs feature importance).
- You can disable autologging with `mlflow.<framework>.autolog(disable=True)`.

## Next Steps

In the next lesson (**L1-3.2: LLM/GenAI Autologging**), you will see how autologging works for LLM frameworks like LangChain, where MLflow captures prompts, completions, and token usage. In Level 2, we will explore autologging in more depth.
