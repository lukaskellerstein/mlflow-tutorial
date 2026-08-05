# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Function Calls

### Setup and tracing

- `mlflow.set_tracking_uri()`
- `mlflow.set_experiment()`
- `mlflow.langchain.autolog()`

### Tracking the search

- `mlflow.start_run()` — parent run for the sweep
- `mlflow.start_run(nested=True)` — one child run per configuration
- `mlflow.log_params()`
- `mlflow.log_metrics()`
- `mlflow.set_tag()`

## Notes

- **There is no MLflow API in this lesson that runs the optimization.** That is
  the point: `mlflow.genai.optimize_prompts()` accepts `prompt_uris` and nothing
  else, so model choice, tool budget and delegation topology have no optimizer.
  MLflow's contribution is the nested-run search log.
- `mlflow.langchain.autolog()` requires the `langchain` meta-package to be
  declared even though the code imports only `langchain_openai` — MLflow's
  version check imports `langchain` itself.
