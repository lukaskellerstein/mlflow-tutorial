"""L1-M6.1 — Model Serving

Combines serving basics with deep dive patterns:
- Wrap an LLM in a PythonModel with load_context for efficient initialization
- Log and register multiple model versions with different configurations
- Test models locally before serving
- Print CLI commands for serving, Docker deployment, and health checks
- Show monitoring best practices for LLM serving
"""

import json

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L1/M6_deployment_gateway/1_model_serving")

MODEL_NAME = "L1-llm-serving-demo"


class LLMChatModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapping an OpenAI-compatible LLM for chat completion.

    Uses load_context() to initialize the client once at model load time,
    not on every prediction call. Accepts a DataFrame with a 'question'
    column and optional 'temperature' and 'max_tokens' columns.
    """

    def __init__(self, model_name: str = "google/gemma-4-e4b",
                 default_temperature: float = 0.7):
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.client = None

    def load_context(self, context: PythonModelContext) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def predict(self, context: PythonModelContext,
                model_input: pd.DataFrame, params: dict | None = None) -> list:
        answers = []
        for _, row in model_input.iterrows():
            temp = float(row.get("temperature") or self.default_temperature)
            max_tok = int(row.get("max_tokens") or 256)
            if self.client is None:
                raise RuntimeError("load_context() has not run; client is not initialised")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Give clear, concise answers."},
                    {"role": "user", "content": row["question"]},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )
            answers.append(response.choices[0].message.content or "")
        return answers


# ── Part 1: Log and Register a Model ─────────────────────────────────────


def part1_log_model(temperature: float, run_name: str) -> str:
    """Create, log, and register the LLM model. Returns the model URI."""
    print("=" * 60)
    print(f"Part 1: Log LLM Model (temperature={temperature})")
    print("=" * 60)

    input_example = pd.DataFrame({
        "question": ["What is machine learning?"],
        "temperature": [temperature],
        "max_tokens": [256],
    })
    output_example = [
        "Machine learning is a branch of AI that enables computers "
        "to learn from data without being explicitly programmed."
    ]
    signature = infer_signature(input_example, output_example)
    print(f"  Model signature:\n{signature}\n")

    with mlflow.start_run(run_name=run_name):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=LLMChatModel(
                model_name="google/gemma-4-e4b",
                default_temperature=temperature,
            ),
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
            pip_requirements=["openai>=1.0", "mlflow>=2.0", "pandas>=2.0"],
        )
        mlflow.log_param("model_backend", "google/gemma-4-e4b")
        mlflow.log_param("default_temperature", temperature)
        print(f"  Model logged: {info.model_uri}")
        print(f"  Registered as: {MODEL_NAME}")
        model_uri = info.model_uri
    return model_uri


# ── Part 2: Test Locally ─────────────────────────────────────────────────


def part2_test_locally(model_uri: str) -> None:
    """Load the model and test it without serving."""
    print("\n" + "=" * 60)
    print("Part 2: Test the Model Locally (No Server)")
    print("=" * 60)

    print(f"  Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    test_questions = pd.DataFrame({
        "question": [
            "What is MLflow?",
            "Explain REST APIs in one sentence.",
            "What is Python used for?",
        ]
    })

    print(f"  Running {len(test_questions)} test questions...\n")
    predictions = model.predict(test_questions)

    for q, a in zip(test_questions["question"], predictions):
        print(f"  Q: {q}")
        print(f"  A: {a[:120]}...")
        print()

    print("  Local testing passed.")


# ── Part 3: Register Multiple Versions with Aliases ──────────────────────


def part3_register_versions(uri_v1: str, uri_v2: str) -> None:
    """Assign aliases to different model versions."""
    print("=" * 60)
    print("Part 3: Model Versions and Aliases")
    print("=" * 60)

    client = mlflow.MlflowClient()
    # Get the latest two versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    versions = sorted(versions, key=lambda v: int(v.version))

    if len(versions) >= 2:
        v1 = versions[-2].version
        v2 = versions[-1].version
        client.set_registered_model_alias(MODEL_NAME, "champion", v1)
        client.set_registered_model_alias(MODEL_NAME, "challenger", v2)
        print(f"  Alias 'champion'   -> v{v1} (creative, temp=0.7)")
        print(f"  Alias 'challenger' -> v{v2} (deterministic, temp=0.3)")
        print("\n  Serve each version:")
        print(f"    mlflow models serve -m models:/{MODEL_NAME}@champion -p 5001")
        print(f"    mlflow models serve -m models:/{MODEL_NAME}@challenger -p 5002")
    else:
        print(f"  Only {len(versions)} version(s) found.")


# ── Part 4: Serving Commands and Endpoints ────────────────────────────────


def part4_serving_commands() -> None:
    """Print CLI commands, endpoint details, and example curl requests."""
    print("\n" + "=" * 60)
    print("Part 4: Serving as REST API + Docker Deployment")
    print("=" * 60)

    print(f"""
  Start serving:
    mlflow models serve \\
      -m "models:/{MODEL_NAME}@champion" \\
      --port 5001 --no-conda

  Endpoints:
    POST /invocations  -- run predictions
    GET  /ping         -- health check (returns 200 OK)
    GET  /health       -- alias for /ping (Kubernetes probe)
    GET  /version      -- MLflow version info
""")

    sample_payload = {
        "dataframe_split": {
            "columns": ["question"],
            "data": [["What is machine learning?"]],
        }
    }
    print("  Example prediction request:")
    print("    curl -X POST http://127.0.0.1:5001/invocations \\")
    print('      -H "Content-Type: application/json" \\')
    print(f"      -d '{json.dumps(sample_payload)}'")

    batch_payload = {
        "dataframe_split": {
            "columns": ["question"],
            "data": [["What is Python?"], ["What is an API?"]],
        }
    }
    print("\n  Batch prediction (multiple questions):")
    print("    curl -X POST http://127.0.0.1:5001/invocations \\")
    print('      -H "Content-Type: application/json" \\')
    print(f"      -d '{json.dumps(batch_payload)}'")

    print(f"""
  Docker containerization:
    mlflow models build-docker \\
      --model-uri models:/{MODEL_NAME}@champion \\
      --name mlflow-llm-server

    podman run -p 5001:8080 \\
      --add-host=host.containers.internal:host-gateway \\
      mlflow-llm-server
""")


# ── Part 5: Monitoring Best Practices ─────────────────────────────────────


def part5_monitoring() -> None:
    """Print health check commands and monitoring guidance."""
    print("=" * 60)
    print("Part 5: Health Checks and Monitoring")
    print("=" * 60)

    for path in ["/ping", "/health", "/version"]:
        print(f"  curl http://localhost:5001{path}")

    print("\n  Monitoring best practices:")
    for i, tip in enumerate([
        "Track request latency (p50, p95, p99) via reverse proxy.",
        "Log prediction counts and error rates to Prometheus.",
        "Monitor token usage per request to control LLM costs.",
        "Alert on latency spikes or error rate above threshold.",
        "Track model version staleness for timely updates.",
    ], 1):
        print(f"    {i}. {tip}")

    print("""
  Deployment comparison:
    Serving  -- REST API, language-agnostic, production use
    Loading  -- in-process, Python only, scripts and notebooks
    Batch    -- CLI-based, file I/O, periodic jobs
    Docker   -- containerized, portable, cloud deployment
""")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    uri_v1 = part1_log_model(temperature=0.7, run_name="llm_serve_creative")
    part2_test_locally(uri_v1)
    uri_v2 = part1_log_model(temperature=0.3, run_name="llm_serve_deterministic")
    part3_register_versions(uri_v1, uri_v2)
    part4_serving_commands()
    part5_monitoring()

    print("=" * 60)
    print("Done! Try serving with the command from Part 4.")
    print("View runs in MLflow UI: http://127.0.0.1:5555")
    print(f"Registered model: {MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
