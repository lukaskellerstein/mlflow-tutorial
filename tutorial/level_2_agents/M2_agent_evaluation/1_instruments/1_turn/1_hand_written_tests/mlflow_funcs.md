# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py` and its
`test_framework.py`.

## Imports

- `from mlflow.artifacts import download_artifacts`

## Function Calls

- `mlflow.artifacts.download_artifacts()` — reads the baseline back out of a run
- `mlflow.langchain.autolog()`
- `mlflow.log_artifact()` — the baseline JSON, via `test_framework.py`
- `mlflow.log_metrics()` — suite metrics, and per-case metrics in nested runs
- `mlflow.log_params()`
- `mlflow.log_table()` — the results DataFrame
- `mlflow.set_experiment()`
- `mlflow.set_tags()` — PASS/FAIL and difficulty, via `test_framework.py`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()` — parent per suite, `nested=True` per case

## Objects and Methods

- `ActiveRun` — `.info.run_id`, used to address the baseline artifact later
