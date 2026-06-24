# L1-7.1 — Dataset Logging and Lineage

**Level:** Essentials
**Duration:** ~20 minutes

## Overview

MLflow can track not just models and metrics, but also the datasets used to produce them. This lesson shows how to log datasets alongside your training runs, creating a clear lineage from data to model. Understanding data lineage is critical for reproducibility and auditing.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Why Track Datasets?

When you train a model, the resulting quality depends on the data that went in. Without dataset tracking, you lose the connection between "which data produced which model." MLflow's dataset logging solves this by:

- **Recording what data was used** — schema, size, and a content digest (hash)
- **Distinguishing contexts** — was this dataset used for training, validation, or testing?
- **Providing lineage** — given a model, you can trace back to the exact data that produced it

### The `mlflow.data` Module

MLflow provides dataset constructors for common formats:

| Constructor | Input Type |
|---|---|
| `mlflow.data.from_pandas()` | Pandas DataFrame |
| `mlflow.data.from_numpy()` | NumPy array |
| `mlflow.data.from_spark()` | Spark DataFrame |
| `mlflow.data.from_huggingface()` | Hugging Face Dataset |

Each constructor returns an `mlflow.data.Dataset` object that captures:

- **Name** — a human-readable label
- **Digest** — a hash of the data content (for change detection)
- **Schema** — column names and types
- **Source** — where the data came from (file path, URL, etc.)
- **Targets** — which column is the prediction target

### Logging and Lineage

Once you have a Dataset object, call `mlflow.log_input(dataset, context="training")` inside an active run. The `context` parameter is a free-form string that records how the dataset was used. Common values are `"training"`, `"validation"`, and `"testing"`.

After the run completes, the dataset info is visible in the MLflow UI under the run's "Datasets" section, and you can query it programmatically via `mlflow.get_run()`.

## Step-by-Step

### Step 1: Create DataFrames from scikit-learn

We load the Wine dataset, split it into training and validation sets, and store each as a pandas DataFrame.

```python
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
```

### Step 2: Wrap DataFrames as MLflow Datasets

`mlflow.data.from_pandas()` takes the DataFrame, an optional source string, the target column name, and a human-readable dataset name.

```python
train_dataset = mlflow.data.from_pandas(
    train_df, source="sklearn.datasets.load_wine", targets="target", name="wine_train"
)
```

### Step 3: Log Datasets with Context

Inside an MLflow run, log each dataset with a context string that describes its role.

```python
with mlflow.start_run(run_name="wine_with_dataset_lineage"):
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(val_dataset, context="validation")
```

### Step 4: Query Dataset Lineage

After the run completes, retrieve the logged datasets programmatically.

```python
run_data = mlflow.get_run(run_id)
for ds_input in run_data.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest)
```

## Running the Lesson

```bash
cd tutorial/level_1/M7_data_datasets/1_dataset_logging
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Loading the Wine dataset
============================================================
  Total samples:      178
  Training samples:   142
  Validation samples: 36

============================================================
Step 2: Creating MLflow datasets from DataFrames
============================================================
  Training dataset:   name=wine_train, digest=<hash>
  Validation dataset: name=wine_val, digest=<hash>
  Schema: ...

============================================================
Step 3: Logging datasets and model in an MLflow run
============================================================
  Logged training dataset with context='training'
  Logged validation dataset with context='validation'
  Validation accuracy: ~0.97
  Run ID: <run_id>

============================================================
Step 4: Querying dataset lineage from the completed run
============================================================
  Dataset: wine_train
    Digest:  <hash>
    Source:  sklearn.datasets.load_wine
    Context: training

  Dataset: wine_val
    Digest:  <hash>
    Source:  sklearn.datasets.load_wine
    Context: validation
```

In the MLflow UI, open the run and look for the **Datasets** section to see the logged datasets with their schemas and contexts.

## Key Takeaways

- `mlflow.data.from_pandas()` wraps a DataFrame into a trackable Dataset object with schema, digest, and source metadata.
- `mlflow.log_input(dataset, context="training")` links a dataset to the current run with a context label.
- Dataset lineage lets you trace any model back to the exact data that produced it.
- You can log multiple datasets per run with different contexts (training, validation, testing).
- The dataset digest acts as a fingerprint — if the data changes, the digest changes.

## Next Steps

Continue to **Level 2, M1** (Advanced Tracking) to explore nested runs, async logging, and rich artifact types. In Level 2, we'll explore datasets in more depth with evaluation workflows.
