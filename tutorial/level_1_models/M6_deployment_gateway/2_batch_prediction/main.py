"""
L2-8.2 -- Batch LLM Inference Pipeline

Demonstrates batch LLM inference with MLflow: wrap an LLM in a PyFunc model,
run a batch of diverse prompts, collect responses with per-prompt timing and
token counts, log everything as artifacts and metrics, and show CLI commands
for offline batch scoring.
"""

import os
import tempfile
import time

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5555"
EXPERIMENT_NAME = "L2/M7_deployment/2_batch_prediction"
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "google/gemma-4-e4b"

# Rough cost estimate per 1K tokens (local model -- effectively free, but
# we track it to demonstrate the pattern for paid APIs).
COST_PER_1K_TOKENS = 0.0001


# ---------------------------------------------------------------------------
# PyFunc LLM wrapper
# ---------------------------------------------------------------------------
class LLMModel(mlflow.pyfunc.PythonModel):
    """Wraps an OpenAI-compatible LLM for batch scoring via mlflow.pyfunc."""

    def load_context(self, context: PythonModelContext) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        responses, latencies, tokens = [], [], []
        for prompt in model_input["prompt"]:
            start = time.perf_counter()
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            text = (completion.choices[0].message.content or "").strip()
            used = completion.usage.total_tokens if completion.usage else 0
            responses.append(text)
            latencies.append(round(elapsed_ms, 1))
            tokens.append(used)
        return pd.DataFrame({
            "response": responses,
            "latency_ms": latencies,
            "tokens_used": tokens,
        })


# ── Part 1: Create and log the LLM PyFunc model ──────────────────────
def part1_log_model() -> str:
    print("=" * 60)
    print("Part 1: Create and Log the LLM PyFunc Model")
    print("=" * 60)

    with mlflow.start_run(run_name="log_llm_model") as run:
        sample_input = pd.DataFrame({"prompt": ["Say hello."]})
        sample_output = pd.DataFrame({
            "response": ["Hello!"],
            "latency_ms": [120.0],
            "tokens_used": [15],
        })
        signature = infer_signature(sample_input, sample_output)

        mlflow.pyfunc.log_model(
            name="llm_model",
            python_model=LLMModel(),
            signature=signature,
            input_example=sample_input,
        )
        mlflow.log_params({"model": LLM_MODEL, "temperature": 0.7, "max_tokens": 256})

        model_uri = f"runs:/{run.info.run_id}/llm_model"
        print(f"  Model URI: {model_uri}")
        print(f"  Signature: {signature}")
        print()
    return model_uri


# ── Part 2: Batch inference ───────────────────────────────────────────
def part2_batch_inference(model_uri: str) -> tuple[pd.DataFrame, float]:
    print("=" * 60)
    print("Part 2: Batch LLM Inference")
    print("=" * 60)

    batch_prompts = pd.DataFrame({"prompt": [
        "Summarize the benefits of renewable energy in two sentences.",
        "Translate to French: 'The weather is beautiful today.'",
        "What is the capital of Japan?",
        "Classify this review as positive or negative: 'The food was terrible and the service was slow.'",
        "Write a haiku about programming.",
        "Explain quantum computing to a 10-year-old in three sentences.",
        "List three common Python debugging techniques.",
        "Rewrite this sentence more formally: 'Hey, can you fix the bug ASAP?'",
    ]})
    print(f"  Batch size: {len(batch_prompts)} prompts")

    model = mlflow.pyfunc.load_model(model_uri)
    print(f"  Model loaded from: {model_uri}")

    start = time.perf_counter()
    results = model.predict(batch_prompts)
    total_time = time.perf_counter() - start

    results_df = pd.concat([batch_prompts.reset_index(drop=True), results], axis=1)

    for i, (_, row) in enumerate(results_df.iterrows()):
        preview = str(row["response"])[:80].replace("\n", " ")
        print(f"  [{i+1}] {row['latency_ms']:.0f}ms | {row['tokens_used']} tok | {preview}...")

    print(f"\n  Total time: {total_time:.2f}s")
    print()
    return results_df, total_time


# ── Part 3: Log results to MLflow ─────────────────────────────────────
def part3_log_results(results_df: pd.DataFrame, total_time: float) -> None:
    print("=" * 60)
    print("Part 3: Log Batch Results to MLflow")
    print("=" * 60)

    with mlflow.start_run(run_name="batch_inference_results"):
        batch_size = len(results_df)
        avg_latency = float(results_df["latency_ms"].mean())
        total_tokens = int(results_df["tokens_used"].sum())
        cost_estimate = (total_tokens / 1000) * COST_PER_1K_TOKENS

        mlflow.log_metrics({
            "batch_size": batch_size,
            "total_latency_sec": round(total_time, 2),
            "avg_latency_per_prompt_ms": round(avg_latency, 1),
            "total_tokens": total_tokens,
            "cost_estimate_usd": round(cost_estimate, 6),
        })
        print(f"  batch_size              : {batch_size}")
        print(f"  total_latency_sec       : {total_time:.2f}")
        print(f"  avg_latency_per_prompt  : {avg_latency:.1f} ms")
        print(f"  total_tokens            : {total_tokens}")
        print(f"  cost_estimate_usd       : ${cost_estimate:.6f}")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "batch_results.csv")
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="results")
            print("  Logged artifact: results/batch_results.csv")
    print()


# ── Part 4: CLI batch prediction commands ─────────────────────────────
def part4_cli_commands(model_uri: str) -> None:
    print("=" * 60)
    print("Part 4: CLI Batch Prediction")
    print("=" * 60)

    print("  Use 'mlflow models predict' for offline batch scoring:\n")
    print(f'    mlflow models predict -m "{model_uri}" \\')
    print('      -i prompts.csv -o responses.csv --content-type csv')
    print()
    print("  Where prompts.csv has a single 'prompt' column:")
    print('    prompt')
    print('    "Summarize the benefits of renewable energy."')
    print('    "What is the capital of Japan?"')
    print()
    print("  Scheduling examples:")
    print("    # Cron (daily at 2 AM)")
    print('    0 2 * * * mlflow models predict -m "models:/llm-batch/champion" \\')
    print("      -i /data/daily_prompts.csv -o /output/responses.csv")
    print()
    print("    # Temporal / Airflow")
    print("    #   Wrap part2/part3 logic in a task/activity for scheduled")
    print("    #   batch inference with full observability.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print()
    print("=" * 60)
    print("L2-8.2 -- Batch LLM Inference Pipeline")
    print("=" * 60)
    print()

    model_uri = part1_log_model()
    results_df, total_time = part2_batch_inference(model_uri)
    part3_log_results(results_df, total_time)
    part4_cli_commands(model_uri)

    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  View runs at: {TRACKING_URI}")
    print(f"  Experiment  : {EXPERIMENT_NAME}")
    print()


if __name__ == "__main__":
    main()
