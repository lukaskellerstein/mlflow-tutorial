# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import AssessmentSource`
- `from mlflow.entities import Feedback`
- `from mlflow.genai.optimize import MetaPromptOptimizer`
- `from mlflow.genai.scorers import scorer`

## Function Calls

- `mlflow.genai.load_prompt()`
- `mlflow.genai.optimize_prompts()`
- `mlflow.genai.register_prompt()`
- `mlflow.langchain.autolog()`
- `mlflow.log_artifact()`
- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## The optimize_prompts contract

`optimize_prompts(predict_fn=, train_data=, prompt_uris=, optimizer=, scorers=)`

- **`prompt_uris`** — the optimizer rewrites *registered prompt versions*, so the
  prompt must be in the registry (`register_prompt`), not a Python string.
- **`predict_fn`** — receives dataset `inputs` as keyword arguments and must call
  `PromptVersion.format()` during execution. That call is the hook the optimizer
  uses to inject a candidate template; a hardcoded prompt makes the whole run a
  no-op.
- **`train_data`** — rows of `{"inputs": {...}, "outputs": ...}`.

`PromptOptimizationResult` carries `initial_eval_score`, `final_eval_score`,
their per-scorer variants, `optimized_prompts` (new `PromptVersion` objects) and
`optimizer_name`. Compare the two scores — an unchanged score usually means the
baseline was already saturated, not that the optimizer failed.

## Optimizers

- `MetaPromptOptimizer(reflection_model=...)` — a reflection model rewrites the
  prompt. Cheap enough for a tutorial.
- `GepaPromptOptimizer(reflection_model=..., max_metric_calls=100)` — stronger,
  but the default call budget is expensive on a rate-limited provider.
