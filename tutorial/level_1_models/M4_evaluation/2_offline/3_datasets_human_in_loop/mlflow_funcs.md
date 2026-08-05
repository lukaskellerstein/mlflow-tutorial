# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import AssessmentSourceType`

## Decorators

- `@mlflow.trace`

## Function Calls

- `mlflow.data.from_pandas()`
- `mlflow.flush_trace_async_logging()`
- `mlflow.get_last_active_trace_id()`
- `mlflow.get_run()`
- `mlflow.load_table()`
- `mlflow.log_expectation()`
- `mlflow.log_feedback()`
- `mlflow.log_input()`
- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_param()`
- `mlflow.log_table()`
- `mlflow.override_feedback()`
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`
- `mlflow.trace()`
