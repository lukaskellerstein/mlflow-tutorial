"""
L2-8.1 -- LLM Model Serving Deep Dive

Serves LLM models through MLflow using a PyFunc wrapper: logging/loading
with signatures, multiple versions, Docker containerization, health checks.

NOTE: Logs and tests models locally. Does NOT start a live serving endpoint.
"""

import mlflow
import pandas as pd
from mlflow.models import infer_signature


class LLMChatModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapping an OpenAI-compatible LLM for chat completion."""

    def __init__(self, model_name: str = "google/gemma-4-e4b",
                 default_temperature: float = 0.7):
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.client = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def predict(self, context: mlflow.pyfunc.PythonModelContext,
                model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        responses, tokens_list = [], []
        for _, row in model_input.iterrows():
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": row["prompt"]}],
                temperature=float(row.get("temperature", self.default_temperature)),
                max_tokens=int(row.get("max_tokens", 256)),
            )
            responses.append(completion.choices[0].message.content)
            tokens_list.append(completion.usage.total_tokens if completion.usage else 0)
        return pd.DataFrame({"response": responses, "tokens_used": tokens_list})


# -- Part 1: Define and log the LLM PyFunc model -------------------------

def log_llm_model(temperature: float, run_name: str) -> str:
    """Log an LLMChatModel with the given default temperature."""
    sample_input = pd.DataFrame({
        "prompt": ["What is MLflow?", "Explain model serving."],
        "temperature": [temperature, temperature],
        "max_tokens": [128, 128],
    })
    sample_output = pd.DataFrame({
        "response": ["MLflow is an open-source platform...", "Model serving is..."],
        "tokens_used": [45, 38],
    })
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model_name": "google/gemma-4-e4b",
                           "default_temperature": temperature, "llm_provider": "LMStudio"})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=LLMChatModel(
                model_name="google/gemma-4-e4b",
                default_temperature=temperature,
            ),
            signature=signature,
            input_example=sample_input,
            pip_requirements=["openai>=1.0", "pandas>=2.0"],
        )
        print(f"  Model logged: {info.model_uri}")
        print(f"  Default temperature: {temperature}")
        return info.model_uri


# -- Part 2: Load and test locally ---------------------------------------

def test_model_locally(model_uri: str) -> None:
    """Load a logged model and run sample predictions."""
    print(f"  Loading model from: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    test_prompts = pd.DataFrame({
        "prompt": [
            "What is MLflow model serving in one sentence?",
            "Name three benefits of containerized ML deployments.",
        ],
        "temperature": [0.7, 0.5],
        "max_tokens": [100, 150],
    })
    print("  Running predictions...")
    results = loaded_model.predict(test_prompts)
    for i, row in results.iterrows():
        print(f"\n  Prompt {i + 1}: {test_prompts.iloc[i]['prompt']}")
        print(f"  Response: {row['response'][:120]}...")
        print(f"  Tokens used: {row['tokens_used']}")


# -- Part 3: Register multiple versions ----------------------------------

def register_model_versions(uri_v1: str, uri_v2: str) -> None:
    """Register both model versions and assign aliases."""
    reg_name = "llm-serving-demo"
    mv1 = mlflow.register_model(uri_v1, reg_name)
    mv2 = mlflow.register_model(uri_v2, reg_name)
    print(f"  Registered v{mv1.version} (temp=0.7) and v{mv2.version} (temp=0.3)")

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(reg_name, "champion", mv1.version)
    client.set_registered_model_alias(reg_name, "challenger", mv2.version)
    print(f"  Alias 'champion'   -> v{mv1.version} (creative, temp=0.7)")
    print(f"  Alias 'challenger' -> v{mv2.version} (deterministic, temp=0.3)")
    print(f"\n  Serve each version:")
    print(f"    mlflow models serve -m models:/{reg_name}@champion -p 5001")
    print(f"    mlflow models serve -m models:/{reg_name}@challenger -p 5002")


# -- Part 4: Docker containerization -------------------------------------

def show_docker_containerization() -> None:
    """Print Docker build/run commands for the LLM model."""
    commands = """  Build a Docker image:
    mlflow models build-docker \\
      --model-uri models:/llm-serving-demo@champion \\
      --name mlflow-llm-server

  Run the container (expose LMStudio host):
    podman run -p 5001:8080 \\
      --add-host=host.containers.internal:host-gateway \\
      mlflow-llm-server

  Send a prediction request:
    curl -X POST http://localhost:5001/invocations \\
      -H "Content-Type: application/json" \\
      -d '{{"dataframe_split": {{
            "columns": ["prompt", "temperature", "max_tokens"],
            "data": [["What is MLflow?", 0.7, 128]]
          }}}}'

  Input formats: dataframe_split | dataframe_records"""
    print(commands)


# -- Part 5: Health checks and monitoring ---------------------------------

def show_health_and_monitoring() -> None:
    """Print endpoint details and monitoring guidance."""
    endpoints = [
        ("GET /ping",        "Liveness check. Returns 200 when ready."),
        ("GET /health",      "Alias for /ping. Kubernetes readiness probe."),
        ("GET /version",     "Returns MLflow version for debugging."),
        ("POST /invocations", "Prediction endpoint. Accepts JSON or CSV."),
    ]
    for ep, desc in endpoints:
        print(f"  {ep:25s} {desc}")
    print("\n  Health check commands:")
    for path in ["/ping", "/health", "/version"]:
        print(f"    curl http://localhost:5001{path}")
    print("\n  Monitoring best practices:")
    for i, tip in enumerate([
        "Track request latency (p50, p95, p99) via reverse proxy.",
        "Log prediction counts and error rates to Prometheus.",
        "Monitor token usage per request to control LLM costs.",
        "Alert on latency spikes or error rate above threshold.",
        "Track model version staleness for timely updates.",
    ], 1):
        print(f"    {i}. {tip}")


# -- Main ----------------------------------------------------------------

def _section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    _section("Part 1: Define and log LLM PyFunc model (temp=0.7)")
    uri_v1 = log_llm_model(temperature=0.7, run_name="llm_serve_creative")
    print()

    _section("Part 2: Load and test locally with sample prompts")
    test_model_locally(uri_v1)
    print()

    _section("Part 3: Log second version (temp=0.3) and register both")
    uri_v2 = log_llm_model(temperature=0.3, run_name="llm_serve_deterministic")
    register_model_versions(uri_v1, uri_v2)
    print()

    _section("Part 4: Docker containerization")
    show_docker_containerization()
    print()

    _section("Part 5: Health check endpoints and monitoring")
    show_health_and_monitoring()
    print()

    _section("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("  Experiment: L2/M7_deployment/1_serving_deep_dive")
    print("  Registered model: llm-serving-demo (2 versions)")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M7_deployment/1_serving_deep_dive")
    main()
