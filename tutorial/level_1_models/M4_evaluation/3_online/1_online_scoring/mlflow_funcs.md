# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.entities import GatewayEndpointModelConfig`
- `from mlflow.entities import GatewayModelLinkageType`
- `from mlflow.genai.scorers import ScorerSamplingConfig`
- `from mlflow.tracking._tracking_service.utils import _get_store`

## Function Calls

### Setup and tracing

- `mlflow.set_tracking_uri()`
- `mlflow.set_experiment()`
- `mlflow.openai.autolog()`

### Gateway (server-side credentials for the judge)

- `store.list_gateway_endpoints()`
- `store.list_secret_infos()`
- `store.create_gateway_secret()`
- `store.list_gateway_model_definitions()`
- `store.create_gateway_model_definition()`
- `store.create_gateway_endpoint()`

### Online scoring

- `mlflow.genai.make_judge()`
- `judge.register()`
- `scorer.start(sampling_config=...)`
- `scorer.update(sampling_config=...)`
- `scorer.stop()`
- `mlflow.genai.get_scorer()`

### Traces and assessments

- `mlflow.flush_trace_async_logging()`
- `mlflow.search_traces()`

## Notes

- `mlflow.openai.autolog()` is what produces the traces. Without it the scorer is
  active but has nothing to sample.
- `start()` requires a **gateway** model. The app's LMStudio base URL and key live
  in your process; the MLflow server has neither, and scoring runs there.
- `register()` only works for `INSTRUCTIONS`-kind scorers (`make_judge`).
  `@scorer`-decorated functions are `DECORATOR` kind and cannot be registered
  against a non-Databricks tracking URI.
- `stop()` sets the sample rate to 0 but keeps the scorer registered.
