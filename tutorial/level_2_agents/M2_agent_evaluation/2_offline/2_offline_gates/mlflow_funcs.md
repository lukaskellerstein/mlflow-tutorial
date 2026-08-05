# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import GatewayEndpointModelConfig`
- `from mlflow.entities import GatewayModelLinkageType`
- `from mlflow.genai.scorers import ScorerSamplingConfig`
- `from mlflow.tracking._tracking_service.utils import _get_store`

## Function Calls

### Offline (Parts 1–2)

- `mlflow.log_artifact()`
- `mlflow.log_metric()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tags()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

### Online (Part 3)

- `mlflow.flush_trace_async_logging()`
- `mlflow.genai.get_scorer()`
- `mlflow.genai.make_judge()`

## Gateway store operations

Reached through the tracking store (`_get_store()`), which is a `RestStore` once
`set_tracking_uri` points at a server:

- `create_gateway_secret(secret_name=, secret_value=, provider=)`
- `create_gateway_model_definition(name=, secret_id=, provider=, model_name=)`
- `create_gateway_endpoint(name=, model_configs=[...])`
- `list_secret_infos()` / `list_gateway_model_definitions()` / `list_gateway_endpoints()`
- `delete_gateway_secret()` / `delete_gateway_model_definition()` / `delete_gateway_endpoint()`

> [!warning]
> Call `mlflow.set_tracking_uri()` **before** `_get_store()`. Without it the
> helper silently builds a *local* store and creates an `mlflow.db` in the
> working directory instead of talking to your server.

## Scorer lifecycle

- `Judge.register(name=)` — store the judge server-side
- `Scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=, filter_string=))`
- `Scorer.stop()` — a started scorer samples forever, at a model call per trace
- `Scorer.status` — `UNREGISTERED` / `STARTED` / `STOPPED`, derived from the
  sample rate rather than stored as a flag

`start()` returns a **new** scorer instance; the original still reports
`STOPPED`. Read the return value, or re-fetch with `get_scorer()`.

Only `gateway:/…` models can go online. `openai:/…` is a client-side URI and
`start()` rejects it — the scoring runs on the server, which needs its own
credentials.
