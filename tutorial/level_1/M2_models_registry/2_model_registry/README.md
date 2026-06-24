# L1-2.2 — Model Registry

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

The MLflow Model Registry is a centralized hub for managing the full lifecycle of ML models. In this lesson you will register trained models, assign version aliases like "champion" and "challenger", add metadata, and load a model by alias for inference.

## Prerequisites

- Completed: L1-2.1 (Models and Flavors)
- MLflow server running at http://127.0.0.1:5000
- Infrastructure started (`cd infra && podman compose up -d`)

## Concepts

### Why a Model Registry?

When teams train many models across experiments, they need a single place to:

1. **Track versions** -- every registered model gets an auto-incrementing version number.
2. **Assign aliases** -- labels like `champion` (the production model) and `challenger` (the next candidate) that point to specific versions.
3. **Add metadata** -- descriptions and tags that document what each version is, how it was trained, and why it was promoted.
4. **Load by name** -- downstream services load `models:/MyModel@champion` instead of hard-coding run IDs.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Registered Model** | A named entry in the registry (e.g., `L1-iris-classifier`). |
| **Model Version** | An immutable snapshot tied to a specific run and artifact path. |
| **Alias** | A mutable pointer (e.g., `champion`) that can be moved between versions. |
| **Tags** | Key-value metadata on a model or version (e.g., `algorithm=random_forest`). |
| **Description** | Free-text documentation on the model or a specific version. |

### Aliases vs. Stages (Legacy)

Older MLflow versions used "stages" (`Staging`, `Production`, `Archived`). Modern MLflow replaces stages with **aliases**, which are more flexible -- you can define any alias name and a model can have multiple aliases simultaneously.

## Step-by-Step

### Step 1: Train two models

We train a RandomForestClassifier and a GradientBoostingClassifier on the Iris dataset, logging each to its own MLflow run.

```python
for name, clf in models.items():
    with mlflow.start_run(run_name=name) as run:
        clf.fit(X_train, y_train)
        mlflow.sklearn.log_model(clf, artifact_path="model")
```

### Step 2: Register models

`mlflow.register_model()` creates a new version under a named model. If the name does not exist yet, it is created automatically.

```python
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "L1-iris-classifier")
```

### Step 3: List versions

The `MlflowClient` provides programmatic access to the registry.

```python
client = MlflowClient()
versions = client.search_model_versions("name='L1-iris-classifier'")
```

### Step 4: Set aliases

Point `champion` at the better model and `challenger` at the other.

```python
client.set_registered_model_alias("L1-iris-classifier", "champion", best_version)
client.set_registered_model_alias("L1-iris-classifier", "challenger", other_version)
```

### Step 5: Add descriptions and tags

```python
client.update_registered_model(name, description="...")
client.update_model_version(name, version, description="...")
client.set_model_version_tag(name, version, "algorithm", "random_forest")
```

### Step 6: Load by alias and predict

```python
model = mlflow.sklearn.load_model("models:/L1-iris-classifier@champion")
predictions = model.predict(X_test)
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_models_registry/2_model_registry
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Loading the Iris dataset
============================================================
  Training samples: 120
  Test samples:     30

============================================================
Step 2: Training and logging two models
============================================================
  random_forest              accuracy=1.0000  run_id=...
  gradient_boosting          accuracy=1.0000  run_id=...

============================================================
Step 3: Registering models in the Model Registry
============================================================
  Registered random_forest as L1-iris-classifier version 1
  Registered gradient_boosting as L1-iris-classifier version 2

============================================================
Step 5: Setting aliases (champion / challenger)
============================================================
  champion  -> version 1 (random_forest, acc=1.0000)
  challenger -> version 2 (gradient_boosting, acc=1.0000)

...

============================================================
Done! View the Model Registry in the MLflow UI:
  http://127.0.0.1:5000/#/models/L1-iris-classifier
============================================================
```

Version numbers will increment if you run the lesson multiple times (the registry keeps all versions).

In the MLflow UI, navigate to **Models** to see the registered model, its versions, aliases, and tags.

## Key Takeaways

- The **Model Registry** provides a centralized, versioned catalog for your models.
- Every call to `register_model()` creates a new **version** under a named model.
- **Aliases** (`champion`, `challenger`) are mutable pointers -- move them between versions to promote or roll back.
- **Descriptions** and **tags** provide documentation and searchable metadata.
- Load models by name and alias (`models:/name@alias`) so downstream code never hard-codes run IDs.

## Next Steps

Continue to **L1-2.3 PyFunc** to learn how to wrap arbitrary Python code as an MLflow model using `PythonModel`. In Level 2, we will explore advanced registry workflows including CI/CD promotion and model lifecycle automation.
