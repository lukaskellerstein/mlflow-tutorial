# L1-9.1 — MLflow Projects

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

MLflow Projects provide a standard format for packaging and reproducing ML experiments. In this lesson you will learn what an MLflow Project is, how to define one with an `MLproject` file, and how to run projects locally or from a Git repository.

## Prerequisites

- Completed: L1-M1 (Core Platform) lessons
- MLFlow server running at http://127.0.0.1:5000
- Ollama is **not** required for this lesson

## Concepts

### Why Reproducibility Matters

ML experiments involve code, data, dependencies, and hyperparameters. Changing any one of these can change the result. MLflow Projects capture all of these in a single, runnable package so that anyone can reproduce your experiment exactly.

### What is an MLflow Project?

An MLflow Project is a directory (or Git repo) that contains:

| File | Purpose |
|------|---------|
| `MLproject` | Declares entry points, parameters, and the environment spec |
| `python_env.yaml` / `conda.yaml` / `Dockerfile` | Specifies the runtime environment |
| `train.py` (or other scripts) | The actual code to run |

### The MLproject File

The `MLproject` file is a YAML document at the root of the project:

```yaml
name: iris-training

python_env: python_env.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth}"
```

Key elements:
- **name** — human-readable project name
- **python_env / conda_env / docker_env** — environment to create before running
- **entry_points** — named commands with typed, defaulted parameters

### Environment Options

| Option | When to Use |
|--------|-------------|
| `python_env` | Lightweight virtualenv; best for simple projects |
| `conda_env` | Full Conda environment; supports non-Python deps |
| `docker_env` | Complete container isolation; production deployments |

## Step-by-Step

### Step 1: Understand the Project Concept

The lesson begins by explaining what MLflow Projects are and how they enable reproducibility. No code runs here — just printed explanation.

### Step 2: Generate Project Files

`main.py` writes three files into the lesson directory:

- **MLproject** — declares an `iris-training` project with two hyperparameters
- **python_env.yaml** — pins Python 3.10 with mlflow and scikit-learn
- **train.py** — a standalone training script that parses CLI args, trains a RandomForest on Iris, and logs to MLflow

### Step 3: Review CLI Commands

The lesson prints the key commands for running projects:

```bash
# Run locally
mlflow run . -P n_estimators=200 -P max_depth=10

# Run from Git
mlflow run https://github.com/<user>/<repo>.git -P n_estimators=200

# Use current environment (skip virtualenv creation)
mlflow run . --env-manager local
```

### Step 4: Run Training Directly

Instead of invoking `mlflow run` (which creates a fresh virtualenv), the lesson trains a RandomForest directly with `n_estimators=150` and `max_depth=6`, logging parameters, metrics, and the model to MLflow.

## Running the Lesson

```bash
cd tutorial/level_1/M9_projects/1_mlflow_projects
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: What is an MLflow Project?
============================================================
<explanation text>

============================================================
Part 2: Creating MLflow Project Files
============================================================
  Wrote .../MLproject
  Wrote .../python_env.yaml
  Wrote .../train.py

============================================================
Part 3: Running MLflow Projects
============================================================
<CLI command examples>

============================================================
Part 4: Running the Training Directly
============================================================
  Params: {'n_estimators': 150, 'max_depth': 6}
  Accuracy:  1.0000
  F1 Score:  1.0000
  Run ID:    <run-id>

  Model and metrics logged to MLflow.
  View at http://127.0.0.1:5000

============================================================
Lesson complete!
============================================================
```

In the MLflow UI you will see the experiment **L1/M9_projects/1_mlflow_projects** with one run containing the logged parameters, metrics, and a scikit-learn model artifact.

## Key Takeaways

- An MLflow Project is a directory with an `MLproject` file that declares entry points and parameters.
- Projects enable reproducibility by bundling code, dependencies, and hyperparameters together.
- You can run projects locally, from Git repos, or in containers.
- Three environment backends are supported: `python_env`, `conda_env`, and `docker_env`.
- Even without `mlflow run`, the same tracking APIs work — Projects simply add a reproducible wrapper.

## Next Steps

Continue to **L1-M10.1 — Authentication & Permissions** to learn how to secure your MLflow server with user authentication and access controls. In Level 2, we will explore MLflow Projects in more depth with advanced workflows and CI/CD integration.
