# L2-2.3 — Model Registry Workflows

**Level:** Practitioner
**Duration:** 45 min

## Overview

This lesson walks through the full model registry lifecycle: train multiple models, register them as versions of a single registered model, evaluate and compare versions, promote the best to production via aliases, and load the champion model for serving. You will use `MlflowClient` for programmatic registry operations and see how alias-based deployment enables safe model promotion in CI/CD pipelines.

## Prerequisites

- Completed: L1-2.2 (Model Registry basics), L2-2.1 (Signatures Deep Dive), L2-2.2 (Custom PyFunc)
- MLFlow server running at http://127.0.0.1:5000
- Familiarity with scikit-learn classifiers

## Concepts

### Model Lifecycle

In production, models go through a lifecycle:

1. **Train** — experiment with multiple algorithms and hyperparameters
2. **Register** — log the best candidates to the model registry as named versions
3. **Test** — evaluate each version on held-out data, attach metrics as tags
4. **Promote** — assign the `champion` alias to the best version; the runner-up becomes `challenger`
5. **Serve** — load the model by alias (`models:/MyModel@champion`) so deployment code never changes
6. **Retire** — when a new champion is crowned, the old one's alias is reassigned automatically

### Alias-Based Deployment

Instead of hard-coding version numbers, you reference models by alias:

```python
model = mlflow.pyfunc.load_model("models:/MyModel@champion")
```

When you promote a new version, you simply reassign the alias. Every downstream consumer automatically picks up the new model on the next load -- no code changes, no redeployments.

### CI/CD Promotion Pattern

A typical CI/CD pipeline looks like:

1. Train a candidate model and register it
2. Run automated evaluation against a held-out test set
3. Compare metrics against the current champion
4. If the candidate beats the champion, reassign the `champion` alias
5. Tag the old champion as `previous_champion` for rollback

This lesson implements steps 1-4 programmatically using `MlflowClient`.

## Step-by-Step

### Step 1: Train Three Models

We train logistic regression, random forest, and gradient boosting on the UCI Wine dataset. Each model gets its own MLflow run with logged parameters and four evaluation metrics (accuracy, F1, precision, recall).

```python
configs = {
    "logistic_regression": LogisticRegression(...),
    "random_forest": RandomForestClassifier(...),
    "gradient_boosting": GradientBoostingClassifier(...),
}
for name, clf in configs.items():
    with mlflow.start_run(run_name=f"train_{name}") as run:
        clf.fit(X_train, y_train)
        mlflow.sklearn.log_model(clf, name="model")
```

### Step 2: Register All Models

Each trained model is registered as a new version of a single registered model name (`L2-wine-classifier`). This groups all candidates under one umbrella for easy comparison.

```python
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "L2-wine-classifier")
```

### Step 3: Evaluate and Tag Versions

We attach evaluation metrics as version tags so they are visible in the Model Registry UI and queryable via the API.

```python
client.set_model_version_tag(MODEL_NAME, version, "eval_f1", "0.9876")
```

### Step 4: Promote Best to Champion

The version with the highest F1 score gets the `champion` alias; the runner-up gets `challenger`. We also set descriptions and role tags on every version.

```python
client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
client.set_registered_model_alias(MODEL_NAME, "challenger", runner_up_version)
```

### Step 5: Load Champion by Alias

Downstream serving code loads the model by alias, not version number. When a new champion is promoted, this code automatically picks it up.

```python
model = mlflow.pyfunc.load_model("models:/L2-wine-classifier@champion")
predictions = model.predict(X_test)
```

### Step 6: Compare All Versions

A comparison table displays accuracy, F1, precision, recall, and aliases for every registered version side by side.

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/3_registry_workflows
uv sync
uv run python main.py
```

## Expected Output

```
======================================================================
Step 1: Train three models on the Wine dataset
======================================================================
  logistic_regression        acc=0.9722  f1=0.9722
  random_forest              acc=1.0000  f1=1.0000
  gradient_boosting          acc=0.9722  f1=0.9725

======================================================================
Step 2: Register all models as versions of L2-wine-classifier
======================================================================
  logistic_regression        -> L2-wine-classifier v1
  random_forest              -> L2-wine-classifier v2
  gradient_boosting          -> L2-wine-classifier v3

======================================================================
Step 3: Evaluate each version and tag with metrics
======================================================================
  v1 (logistic_regression): accuracy=0.9722  f1=0.9722
  v2 (random_forest):       accuracy=1.0000  f1=1.0000
  v3 (gradient_boosting):   accuracy=0.9722  f1=0.9725

======================================================================
Step 4: Promote best to champion, runner-up to challenger
======================================================================
  champion   -> v2 (random_forest, f1=1.0000)
  challenger -> v3 (gradient_boosting, f1=0.9725)

======================================================================
Step 5: Load champion model by alias and predict
======================================================================
  Loaded: models:/L2-wine-classifier@champion
  Test accuracy: 1.0000
  Sample predictions: [0, 2, 1, 0, 1, 1, 0, 2]

======================================================================
Step 6: Comparison table of all registered versions
======================================================================
 Version              Algorithm Accuracy     F1 Precision Recall Aliases
      v1  logistic_regression   0.9722 0.9722    0.9735 0.9722       -
      v2        random_forest   1.0000 1.0000    1.0000 1.0000 champion
      v3   gradient_boosting   0.9722 0.9725    0.9735 0.9722 challenger
```

Note: Exact version numbers depend on whether previous versions of `L2-wine-classifier` exist in your registry. Metrics may vary slightly.

## Key Takeaways

- **One registered model, many versions**: group related model candidates under a single name for organized comparison.
- **Aliases replace stages**: MLflow 2.x uses `champion` / `challenger` aliases instead of the deprecated `Staging` / `Production` stages.
- **Tags store metadata**: attach evaluation metrics, algorithm names, and roles as version tags for filtering and querying.
- **Alias-based loading decouples deployment from training**: serving code references `@champion`, so promotions require zero code changes.
- **MlflowClient is your registry API**: use it for programmatic alias management, descriptions, tags, and version queries.

## Next Steps

Move on to **L2-M3: Deep Evaluation** to learn how to build custom metrics and evaluation pipelines. In L2-3.1 you will create domain-specific evaluation functions that go beyond the built-in metrics used here.
