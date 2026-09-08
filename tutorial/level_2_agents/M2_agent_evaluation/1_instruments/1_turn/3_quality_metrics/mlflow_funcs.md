# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import Feedback`
- `from mlflow.entities import Trace`
- `from mlflow.genai.scorers import scorer`

### Session-level scorers

- `from mlflow.genai.scorers import ConversationCompleteness`
- `from mlflow.genai.scorers import ConversationalToolCallEfficiency`
- `from mlflow.genai.scorers import KnowledgeRetention`
- `from mlflow.genai.scorers import UserFrustration`

## Function Calls

- `mlflow.genai.evaluate()`
- `mlflow.get_last_active_trace_id()`
- `mlflow.get_trace()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## Session scorer contract

Session-level scorers are called with `session=list[Trace]` — the ordered turns
of one conversation — instead of `inputs`/`outputs`. `Scorer.is_session_level_scorer`
distinguishes them, and they reject the single-turn parameters entirely.
