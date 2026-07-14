# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import AssessmentSourceType`

## Decorators

- `@mlflow.trace`

## Function Calls

- `mlflow.flush_trace_async_logging()`
- `mlflow.get_last_active_trace_id()`
- `mlflow.langchain.autolog()`
- `mlflow.log_artifact()`
- `mlflow.log_feedback()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.log_table()`
- `mlflow.set_experiment()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`
- `mlflow.trace()`
