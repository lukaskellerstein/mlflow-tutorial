# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.tracking import MlflowClient`
- `from mlflow.tracking.context.abstract_context import RunContextProvider`
- `from mlflow.tracking.context.registry import _run_context_provider_registry`

## Function Calls

- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_param()`
- `mlflow.log_table()`
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`
