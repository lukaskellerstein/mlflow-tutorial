# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import Feedback`
- `from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer`
- `from mlflow.genai.scorers import scorer`

## Function Calls

- `mlflow.genai.list_scorers()`
- `mlflow.genai.make_judge()`
- `mlflow.get_experiment_by_name()`
- `mlflow.get_last_active_trace_id()`
- `mlflow.log_feedback()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.search_traces()`
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`
- `mlflow.trace()`

## Judge / Scorer Methods

- `Judge.align(traces, optimizer)` — re-derive instructions from human labels
- `Judge.register(name=...)` — store the judge on the tracking server
- `Scorer.register(name=...)` — **refused** for `@scorer` functions outside Databricks
