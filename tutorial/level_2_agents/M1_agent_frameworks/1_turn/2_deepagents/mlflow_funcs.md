# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Auto-Tracing

- `mlflow.langchain.autolog()` — auto-traces DeepAgents (built on LangGraph)

## Function Calls

- `mlflow.get_experiment_by_name()`
- `mlflow.get_last_active_trace_id()` — the trace autolog just opened, kept for Part 3
- `mlflow.log_artifact()` — only works for a file with a real path, so only for `FilesystemBackend`
- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.log_table()`
- `mlflow.log_text()` — how a `StateBackend` file gets logged, since it has no path
- `mlflow.search_traces()` — with `filter_string="tags.session = ..."` and `flush=True`
- `mlflow.set_trace_tag()` — names a finished trace, so Part 3 can query it back
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## Entities

- `mlflow.entities.Span` — `span_id`, `parent_id`, `span_type`, `name`, timings
- `mlflow.entities.Trace` — `.data.spans` and `.info.token_usage`
