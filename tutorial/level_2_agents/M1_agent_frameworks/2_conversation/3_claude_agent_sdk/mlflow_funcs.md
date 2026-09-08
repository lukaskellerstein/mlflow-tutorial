# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Function Calls

- `mlflow.flush_trace_async_logging()`
- `mlflow.get_experiment_by_name()`
- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_param()`
- `mlflow.log_params()`
- `mlflow.log_table()`
- `mlflow.search_sessions()`
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`
- `mlflow.start_span()`
- `mlflow.trace()`
- `mlflow.update_current_trace()`

There is no `autolog` call in this lesson. The Claude Agent SDK has no MLflow
integration, so every span here is hand-built — see L2-M1.1.3.
