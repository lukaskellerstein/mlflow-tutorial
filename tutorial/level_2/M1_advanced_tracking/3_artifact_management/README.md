# L2-1.3 — Artifact Management Deep Dive

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

MLflow tracks more than just numbers. You can log images, tables, figures, JSON configs, text files, and entire directories as artifacts attached to a run. This lesson explores every artifact type, shows how to organize them in folders, and discusses storage backend options for production.

## Prerequisites

- Completed: L1-1.2 (Tracking Basics), L2-1.1 (Nested Runs), L2-1.2 (Async & Batch Logging)
- MLflow server running at http://127.0.0.1:5000
- Familiarity with matplotlib and pandas

## Concepts

### What are artifacts?

Artifacts are files or directories attached to a run. Unlike parameters (key-value strings) and metrics (numeric time series), artifacts can be any file: plots, model weights, datasets, configs, reports, audio, video, or custom binary formats.

### Artifact APIs at a glance

| API | What it logs | Input type |
|-----|-------------|------------|
| `mlflow.log_artifact(path)` | A single local file | File path on disk |
| `mlflow.log_artifacts(dir)` | Every file in a local directory | Directory path |
| `mlflow.log_image(img, file)` | An image | PIL Image, numpy array, or `mlflow.Image` |
| `mlflow.log_figure(fig, file)` | A plot | matplotlib or plotly Figure |
| `mlflow.log_table(data, file)` | A table | pandas DataFrame or dict |

All of these accept an `artifact_path` (or embed the path in `artifact_file`) to organize artifacts into subdirectories within the run.

### Artifact storage backends

MLflow stores artifacts in a configurable backend. The choice depends on your deployment:

| Backend | URI scheme | When to use |
|---------|-----------|-------------|
| Local filesystem | `./mlartifacts` or absolute path | Development, single machine |
| S3 / MinIO | `s3://bucket/path` | AWS production, any S3-compatible store |
| GCS | `gs://bucket/path` | Google Cloud production |
| Azure Blob | `wasbs://container@account/path` | Azure production |
| HDFS | `hdfs://host:port/path` | On-prem Hadoop clusters |
| SFTP | `sftp://host/path` | Simple remote storage |

Our tutorial infrastructure uses a local artifact store. In production, point `--default-artifact-root` at an object store when starting the MLflow server.

### Organizing artifacts

Use `artifact_path` to create a clean folder structure inside each run:

```
run/
  artifacts/
    images/           # plots rendered as images
    figures/          # matplotlib/plotly figures
    tables/           # logged DataFrames
    config/           # JSON/YAML configuration files
      preprocessing/  # nested subfolders work too
      model/
    docs/             # text notes, reports
    reports/          # bulk-uploaded directory
```

A consistent naming convention makes it easy to browse artifacts in the UI and download them programmatically.

## Step-by-Step

### Step 1: Log images with `mlflow.log_image()`

You can pass a PIL Image or a numpy array (H x W x 3, uint8). The `artifact_file` argument sets the destination path inside the run.

```python
from PIL import Image
pil_img = Image.new("RGB", (200, 200), color=(30, 144, 255))
mlflow.log_image(pil_img, artifact_file="images/pil_sample.png")

# Numpy array from a matplotlib render
mlflow.log_image(img_array, artifact_file="images/numpy_plot.png")
```

### Step 2: Log tables with `mlflow.log_table()`

Pass a pandas DataFrame (or a plain dict). The table is stored as JSON, and MLflow can render it in the UI.

```python
results_df = pd.DataFrame({
    "model": ["RandomForest", "GradientBoosting", "SVM"],
    "accuracy": [0.92, 0.95, 0.88],
})
mlflow.log_table(data=results_df, artifact_file="tables/model_comparison.json")
```

### Step 3: Log figures with `mlflow.log_figure()`

Log a matplotlib or plotly figure directly, without saving to disk first. MLflow handles serialization.

```python
fig, ax = plt.subplots()
ax.plot(x, y)
mlflow.log_figure(fig, "figures/my_plot.png")
```

### Step 4: Log text and JSON with `mlflow.log_artifact()`

For arbitrary files, write them to a temporary location and then log them. The `artifact_path` parameter controls the destination subfolder.

```python
import json, tempfile, os

config = {"learning_rate": 0.001, "batch_size": 32}
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact(path, artifact_path="config")
```

### Step 5: Organize artifacts in nested subfolders

Stack `artifact_path` values to create deeper hierarchies:

```python
mlflow.log_artifact(preprocessing_path, artifact_path="config/preprocessing")
mlflow.log_artifact(hyperparams_path,   artifact_path="config/model")
```

### Step 6: Log an entire directory with `mlflow.log_artifacts()`

When you have a directory of files (reports, checkpoints, etc.), log them all at once:

```python
mlflow.log_artifacts(local_dir, artifact_path="reports")
```

Every file in `local_dir` appears under `reports/` in the artifact tree.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_tracking/3_artifact_management
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-1.3 — Artifact Management Deep Dive
============================================================

  Run ID: <run-id>

============================================================
Part 1: Logging images (PIL and numpy)
============================================================
  Logged PIL image  -> images/pil_sample.png
  Logged numpy image -> images/numpy_plot.png

============================================================
Part 2: Logging tables (pandas DataFrame)
============================================================
  Logged DataFrame   -> tables/model_comparison.json
  Table shape: 4 rows x 4 columns

============================================================
Part 3: Logging matplotlib figures directly
============================================================
  Logged figure      -> figures/trig_functions.png
  Logged figure      -> figures/accuracy_bars.png

============================================================
Part 4: Logging text and JSON artifacts
============================================================
  Logged JSON        -> config/training_config.json
  Logged text        -> docs/experiment_notes.txt

============================================================
Part 5: Organizing artifacts in subfolders
============================================================
  Logged -> config/preprocessing/preprocessing.json
  Logged -> config/model/hyperparameters.json

============================================================
Part 6: Logging an entire directory (log_artifacts)
============================================================
  Logged directory   -> reports/
    - reports/train_report.json
    - reports/validation_report.json
    - reports/test_report.json

============================================================
Done!
============================================================
```

In the MLflow UI, click on the run and open the **Artifacts** tab. You will see a folder tree with `images/`, `figures/`, `tables/`, `config/`, `docs/`, and `reports/`. Click on images and figures to preview them inline; click on JSON files to view their contents.

## Key Takeaways

- `mlflow.log_image()` accepts PIL images and numpy arrays — use it for rendered plots or generated images.
- `mlflow.log_table()` stores pandas DataFrames as JSON and renders them in the UI.
- `mlflow.log_figure()` logs matplotlib and plotly figures directly without manual file I/O.
- `mlflow.log_artifact()` handles any single file; use `artifact_path` to place it in a subfolder.
- `mlflow.log_artifacts()` uploads an entire directory at once.
- Consistent folder structure (images/, config/, reports/) makes artifacts easy to navigate and download programmatically.
- In production, configure an object store (S3, GCS, Azure Blob) as the artifact backend for durability and shared access.

## Next Steps

Continue to **L2-1.4 — MlflowClient: Programmatic Access** to learn how to use the low-level `MlflowClient` API for CRUD operations, artifact downloads, and building custom reports. In Level 1 you used the fluent API (`mlflow.log_*`); now you will see when the client API is the better choice.
