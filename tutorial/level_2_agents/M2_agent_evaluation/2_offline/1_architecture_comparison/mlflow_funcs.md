# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import Feedback`
- `from mlflow.genai.scorers import scorer`

## Function Calls

- `mlflow.genai.evaluate()`
- `mlflow.genai.get_scorer()`
- `mlflow.genai.make_judge()`
- `mlflow.langchain.autolog()`
- `mlflow.log_artifact()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()` — nested, one child run per architecture

## Judge Methods

- `Judge.register(name=...)` — store the correctness judge server-side so every
  architecture, and every later run, is scored by the same versioned object

## Reading results back

`mlflow.genai.evaluate()` returns a result whose `.result_df` carries one row per
case with `<scorer_name>/value` and `<scorer_name>/rationale` columns. Those
values are **pandas** scalars — a boolean verdict arrives as `np.True_`, which is
not a Python `bool` and, under numpy 2.x, not an `int` — so unwrap with `.item()`
before type-checking, or every score silently reads as the default.
