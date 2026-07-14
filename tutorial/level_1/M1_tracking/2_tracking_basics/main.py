"""
L1-M1.2 — Tracking LLM Experiments

Demonstrates MLflow's core tracking capabilities with LLM calls:
- Logging parameters, metrics, artifacts, and tags
- Bulk logging with log_params() and log_metrics()
- Step-based metric logging across multiple prompts
- Saving LLM responses as artifacts
- Comparing runs with different configurations
"""

import os
import tempfile
import time

import mlflow
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_tracking/2_tracking_basics"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"


def call_llm(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    system_prompt: str | None = None,
) -> dict:
    """Call the LLM and return the response with timing and usage info."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    return {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "response_time_seconds": round(elapsed, 3),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    # ------------------------------------------------------------------
    # 1. Compare temperatures — separate runs for each config
    # ------------------------------------------------------------------
    section("Step 1: Comparing temperatures (0.3, 0.7, 1.0)")

    prompt = "Write a one-paragraph explanation of how neural networks learn."
    temperatures = [0.3, 0.7, 1.0]

    for temp in temperatures:
        with mlflow.start_run(run_name=f"temp_{temp}"):
            mlflow.log_params({
                "model": MODEL,
                "temperature": temp,
                "max_tokens": 256,
                "prompt": prompt,
            })

            result = call_llm(client, prompt, temperature=temp)

            mlflow.log_metrics({
                "response_time_seconds": result["response_time_seconds"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            })

            mlflow.set_tags({
                "level": "1",
                "module": "tracking",
                "lesson": "tracking_basics",
                "experiment_type": "temperature_comparison",
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                response_path = os.path.join(tmpdir, "response.txt")
                with open(response_path, "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Temperature: {temp}\n")
                    f.write(f"Model: {MODEL}\n\n")
                    f.write(f"Response:\n{result['content']}\n")
                mlflow.log_artifact(response_path)

            print(f"  temp={temp}  tokens={result['total_tokens']:>4d}"
                  f"  time={result['response_time_seconds']}s")

    # ------------------------------------------------------------------
    # 2. Step-based metrics — track across multiple prompts
    # ------------------------------------------------------------------
    section("Step 2: Step-based metric logging across multiple prompts")

    prompts = [
        "What is a transformer model?",
        "What is attention in machine learning?",
        "What is backpropagation?",
        "What is gradient descent?",
        "What is overfitting?",
    ]

    with mlflow.start_run(run_name="multi_prompt_steps"):
        mlflow.log_params({
            "model": MODEL,
            "temperature": 0.7,
            "max_tokens": 128,
            "num_prompts": len(prompts),
        })
        mlflow.set_tags({"experiment_type": "step_based_metrics"})

        cumulative_tokens = 0
        for step, p in enumerate(prompts):
            result = call_llm(client, p, temperature=0.7, max_tokens=128)
            cumulative_tokens += result["total_tokens"]

            mlflow.log_metric("response_time_seconds", result["response_time_seconds"], step=step)
            mlflow.log_metric("step_tokens", result["total_tokens"], step=step)
            mlflow.log_metric("cumulative_tokens", cumulative_tokens, step=step)

            print(f"  Step {step}  '{p[:40]}'  tokens={result['total_tokens']}"
                  f"  cumulative={cumulative_tokens}")

        mlflow.log_metric("total_tokens_all_prompts", cumulative_tokens)

    # ------------------------------------------------------------------
    # 3. Logging a summary artifact
    # ------------------------------------------------------------------
    section("Step 3: Logging a summary artifact")

    with mlflow.start_run(run_name="summary_artifact"):
        mlflow.log_params({"model": MODEL, "temperature": 0.7})
        mlflow.set_tags({"experiment_type": "artifact_demo"})

        summary_lines = ["# LLM Response Summary\n"]
        for p in prompts[:3]:
            result = call_llm(client, p, temperature=0.7, max_tokens=128)
            summary_lines.append(f"## {p}\n{result['content']}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.md")
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_lines))
            mlflow.log_artifact(summary_path)
            print(f"  Logged summary.md with {len(prompts[:3])} responses")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    section("Done!")
    print(f"  Open the MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Compare the temperature runs side-by-side in the UI.")
    print("  Check the step-based metrics chart for the multi_prompt run.")
    print("  Click on any run's artifacts to see the saved responses.")


if __name__ == "__main__":
    main()
