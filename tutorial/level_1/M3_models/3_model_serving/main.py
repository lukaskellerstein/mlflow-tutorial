"""
L1-M3.3 -- Model Serving Basics

Wrap an LLM call in a custom PythonModel, log it with MLflow,
test it locally, and show how to serve it as a REST API.
"""

import json

import mlflow
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M3_models/3_model_serving")

MODEL_NAME = "L1-llm-serving-demo"


class LLMModel(mlflow.pyfunc.PythonModel):
    """Wraps an LLM API call as a servable MLflow model.

    This PythonModel calls a local LMStudio server. When served via
    `mlflow models serve`, it exposes the LLM through a REST API so
    any language or service can call it over HTTP.
    """

    def predict(self, context, model_input, params=None):
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:1234/v1", api_key="lm-studio"
        )

        # model_input is a DataFrame with a "question" column
        questions = model_input["question"].tolist()
        answers = []
        for question in questions:
            response = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. "
                        "Give clear, concise answers.",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            answers.append(response.choices[0].message.content)
        return answers


def part1_log_model() -> str:
    """Create, log, and register the LLM model. Returns the run ID."""
    print("=" * 60)
    print("Part 1: Creating and Logging the LLM Model")
    print("=" * 60)

    # Define signature: input is a DataFrame with "question", output is strings
    input_example = pd.DataFrame(
        {"question": ["What is machine learning?"]}
    )
    output_example = [
        "Machine learning is a branch of AI that enables computers "
        "to learn from data without being explicitly programmed."
    ]
    signature = infer_signature(input_example, output_example)
    print(f"  Model signature:\n{signature}\n")

    with mlflow.start_run(run_name="log_llm_for_serving") as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=LLMModel(),
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
            pip_requirements=["openai>=1.0", "mlflow>=2.0"],
        )
        mlflow.log_param("model_backend", "google/gemma-4-e4b")
        mlflow.log_param("server", "http://localhost:1234/v1")
        run_id = run.info.run_id
        print(f"  Model logged and registered as: {MODEL_NAME}")
        print(f"  Run ID: {run_id}")

    return run_id


def part2_test_locally(run_id: str) -> None:
    """Load the model and test it without serving."""
    print("\n" + "=" * 60)
    print("Part 2: Testing the Model Locally (No Server)")
    print("=" * 60)

    model_uri = f"runs:/{run_id}/model"
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

    print("  Local testing passed. The model works in-process.")


def part3_serving_commands() -> None:
    """Print CLI commands and endpoint details for model serving."""
    print("\n" + "=" * 60)
    print("Part 3: Serving the Model as a REST API")
    print("=" * 60)

    print(f"""
  To serve this model, run:

    mlflow models serve \\
      -m "models:/{MODEL_NAME}/1" \\
      --port 5001 \\
      --no-conda

  This starts a local REST server on port 5001.
  The --no-conda flag uses your current Python environment.

  Endpoints:
    POST /invocations  -- run predictions
    GET  /ping         -- health check (returns 200 OK)
    GET  /version      -- MLflow version info
""")

    # Show example curl commands
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

    # Alternative: instances format
    alt_payload = {"instances": [{"question": "What is machine learning?"}]}
    print(f"\n  Alternative (instances format):")
    print(f"    -d '{json.dumps(alt_payload)}'")

    # Batch with multiple questions
    batch_payload = {
        "dataframe_split": {
            "columns": ["question"],
            "data": [
                ["What is Python?"],
                ["What is an API?"],
                ["Explain Docker briefly."],
            ],
        }
    }
    print(f"\n  Batch prediction (multiple questions):")
    print("    curl -X POST http://127.0.0.1:5001/invocations \\")
    print('      -H "Content-Type: application/json" \\')
    print(f"      -d '{json.dumps(batch_payload)}'")


def part4_deployment_options() -> None:
    """Show additional deployment options."""
    print("\n" + "=" * 60)
    print("Part 4: Additional Deployment Options")
    print("=" * 60)

    print(f"""
  Batch prediction (no server needed):
    mlflow models predict \\
      -m "models:/{MODEL_NAME}/1" \\
      -i questions.csv

  Container deployment (Podman/Docker):
    mlflow models build-docker \\
      -m "models:/{MODEL_NAME}/1" \\
      -n "llm-server"
    podman run -p 5001:8080 llm-server

  Load programmatically (Python only):
    import mlflow
    model = mlflow.pyfunc.load_model("models:/{MODEL_NAME}@champion")
    result = model.predict(pd.DataFrame({{"question": ["..."]}}))

  Comparison:
    Serving  -- REST API, language-agnostic, production use
    Loading  -- in-process, Python only, scripts and notebooks
    Batch    -- CLI-based, file I/O, periodic jobs
    Docker   -- containerized, portable, cloud deployment
""")


if __name__ == "__main__":
    run_id = part1_log_model()
    part2_test_locally(run_id)
    part3_serving_commands()
    part4_deployment_options()

    print("=" * 60)
    print("Done! Try serving with the command from Part 3.")
    print("View runs in MLflow UI: http://127.0.0.1:5000")
    print("=" * 60)
