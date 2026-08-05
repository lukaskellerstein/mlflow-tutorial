# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Function Calls

### Setup and tracing

- `mlflow.set_tracking_uri()`
- `mlflow.set_experiment()`
- `mlflow.langchain.autolog()`

### Tracking the dev / held-out distinction

- `mlflow.start_run()` — parent run for the whole optimization
- `mlflow.start_run(nested=True)` — one child run per (variant, split) pair
- `mlflow.log_params()`
- `mlflow.log_metrics()` — including `dev_minus_heldout`, the overfitting signal
- `mlflow.set_tags()` — `split: dev` / `split: held_out` on every run

## Notes

- The `split` tag is the load-bearing part. A run whose split is unrecorded
  cannot be trusted later, because nobody can tell whether its number was tuned
  against or held out.
- `dev_minus_heldout` is logged as a metric rather than computed at read time so
  it can be charted across optimization iterations — a rising gap is the signal
  to stop tuning.
- `mlflow.langchain.autolog()` requires the `langchain` meta-package to be
  declared even though the code imports only `langchain_openai`.
