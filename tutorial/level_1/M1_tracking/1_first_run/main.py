"""
L1-M1.1 — Your First MLflow Run

Connect to the MLflow tracking server, call a local LLM via LMStudio,
and log the configuration and results as an MLflow run.
"""

import time

import mlflow
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_tracking/1_first_run"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"


def call_llm(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> dict:
    """Call the LLM and return the response with timing info."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    return {
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "response_time_seconds": round(elapsed, 3),
    }


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    # ------------------------------------------------------------------
    # 1. What is MLflow?
    # ------------------------------------------------------------------
    print("=" * 60)
    print("MLflow — The 5 Pillars")
    print("=" * 60)
    pillars = {
        "1. Tracking": "Record parameters, metrics, and artifacts for every experiment run.",
        "2. Models": "Package ML/AI models in a standard format for any framework.",
        "3. Model Registry": "Centralized model store with versioning and lifecycle management.",
        "4. Evaluation": "Evaluate model quality with built-in and custom metrics.",
        "5. Deployment": "Serve models as REST APIs or batch-process predictions.",
    }
    for pillar, description in pillars.items():
        print(f"  {pillar}")
        print(f"    {description}")
    print()

    # ------------------------------------------------------------------
    # 2. Make an LLM call and track it
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Creating your first MLflow run")
    print("=" * 60)

    prompt = "Explain what MLflow is in 2 sentences."
    temperature = 0.7
    max_tokens = 256

    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Experiment   : {EXPERIMENT_NAME}")
    print(f"  Model        : {MODEL}")
    print(f"  Prompt       : {prompt}")
    print()

    with mlflow.start_run(run_name="first_llm_call") as run:
        mlflow.log_params({
            "model": MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
        })

        result = call_llm(client, prompt, temperature, max_tokens)

        mlflow.log_metrics({
            "response_time_seconds": result["response_time_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        })

        mlflow.set_tags({
            "level": "1",
            "module": "tracking",
            "lesson": "first_run",
        })

        print(f"  Run ID          : {run.info.run_id}")
        print(f"  Response time   : {result['response_time_seconds']}s")
        print(f"  Tokens (total)  : {result['total_tokens']}")
        print(f"  Finish reason   : {result['finish_reason']}")
        print()
        print("  LLM response:")
        print(f"    {result['content']}")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Open the MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  You should see the run with parameters, metrics, and tags.")


if __name__ == "__main__":
    main()
