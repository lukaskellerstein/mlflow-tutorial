"""
L2-1.1 -- Nested Runs and Run Hierarchies

Demonstrates how to use nested runs for LLM configuration sweeps:
- A parent run groups the entire sweep
- Each child run (nested=True) logs one temperature x prompt variant combination
- 3 temperatures x 3 prompt styles = 9 nested runs
- Parent run records summary: best config by response quality
- search_runs() retrieves and ranks all child runs
"""

import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
TEMPERATURES = [0.3, 0.7, 1.0]

PROMPT_VARIANTS: dict[str, str] = {
    "concise": "You are a helpful assistant. Be concise and brief. Answer in 2-3 sentences maximum.",
    "detailed": "You are a helpful assistant. Be detailed and thorough. Provide comprehensive explanations with examples.",
    "creative": "You are a helpful assistant. Be creative and engaging. Use analogies and vivid language to explain concepts.",
}

TEST_QUESTION = "Explain what MLflow is and why it is useful."

MODEL_NAME = "google/gemma-4-e4b"


def call_llm(
    client: OpenAI,
    temperature: float,
    system_prompt: str,
    user_question: str,
) -> tuple[str, int, float]:
    """Call LLM and return response text, token count, and latency in seconds."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=temperature,
    )
    latency = time.time() - start

    text = response.choices[0].message.content or ""
    token_count = response.usage.total_tokens if response.usage else 0

    return text, token_count, latency


def run_child(
    client: OpenAI,
    temperature: float,
    variant_name: str,
    system_prompt: str,
) -> dict:
    """Run one config inside a nested child run."""
    run_label = f"temp_{temperature}_style_{variant_name}"

    with mlflow.start_run(run_name=run_label, nested=True) as child_run:
        # -- Log parameters ---------------------------------------------------
        mlflow.log_params({
            "temperature": temperature,
            "prompt_variant": variant_name,
            "model": MODEL_NAME,
            "question": TEST_QUESTION,
        })

        # -- Tags for easy filtering ------------------------------------------
        mlflow.set_tags({
            "prompt_variant": variant_name,
            "temperature": str(temperature),
        })

        # -- Call LLM ---------------------------------------------------------
        response_text, token_count, latency = call_llm(
            client, temperature, system_prompt, TEST_QUESTION
        )

        # -- Log metrics ------------------------------------------------------
        response_length = len(response_text)
        mlflow.log_metrics({
            "response_length": response_length,
            "token_count": token_count,
            "latency_seconds": round(latency, 3),
        })

        # -- Log response as a text artifact ----------------------------------
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / f"{run_label}.txt"
            artifact_path.write_text(
                f"System prompt: {system_prompt}\n"
                f"User question: {TEST_QUESTION}\n"
                f"Temperature:   {temperature}\n"
                f"Model:         {MODEL_NAME}\n"
                f"\n{'=' * 40}\nResponse:\n{'=' * 40}\n\n"
                f"{response_text}\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="responses")

        print(f"  {run_label:35s}  length={response_length:5d}  tokens={token_count:5d}  latency={latency:.2f}s")

        return {
            "run_id": child_run.info.run_id,
            "run_name": run_label,
            "temperature": temperature,
            "prompt_variant": variant_name,
            "response_length": response_length,
            "token_count": token_count,
            "latency_seconds": round(latency, 3),
        }


def run_config_sweep() -> None:
    """Execute the full nested-run LLM config sweep."""

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # -- Step 1: Print sweep info --------------------------------------------
    print("=" * 60)
    print("Step 1: LLM Configuration Sweep")
    print("=" * 60)
    num_configs = len(TEMPERATURES) * len(PROMPT_VARIANTS)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Temperatures:    {TEMPERATURES}")
    print(f"  Prompt variants: {list(PROMPT_VARIANTS.keys())}")
    print(f"  Total configs:   {num_configs}")
    print(f"  Question:        {TEST_QUESTION}")
    print()

    # -- Step 2: Parent run with nested children -----------------------------
    print("=" * 60)
    print("Step 2: Running sweep (nested runs)")
    print("=" * 60)
    results: list[dict] = []

    with mlflow.start_run(run_name="LLM Config Sweep") as parent_run:
        mlflow.set_tags({
            "sweep_type": "llm_config_sweep",
            "model": MODEL_NAME,
            "num_configs": str(num_configs),
        })

        for temperature in TEMPERATURES:
            for variant_name, system_prompt in PROMPT_VARIANTS.items():
                result = run_child(client, temperature, variant_name, system_prompt)
                results.append(result)

        # -- Step 3: Parent summary ------------------------------------------
        print()
        print("=" * 60)
        print("Step 3: Logging parent-run summary")
        print("=" * 60)

        best_by_length = max(results, key=lambda r: r["response_length"])
        shortest = min(results, key=lambda r: r["response_length"])
        fastest = min(results, key=lambda r: r["latency_seconds"])

        avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
        avg_tokens = sum(r["token_count"] for r in results) / len(results)

        mlflow.log_params({
            "best_config_by_length": best_by_length["run_name"],
            "shortest_config": shortest["run_name"],
            "fastest_config": fastest["run_name"],
        })
        mlflow.log_metrics({
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_token_count": round(avg_tokens, 1),
            "max_response_length": best_by_length["response_length"],
            "min_response_length": shortest["response_length"],
        })
        mlflow.set_tag("best_child_run_id", best_by_length["run_id"])

        print(f"  Most detailed:   {best_by_length['run_name']}  (length={best_by_length['response_length']})")
        print(f"  Most concise:    {shortest['run_name']}  (length={shortest['response_length']})")
        print(f"  Fastest:         {fastest['run_name']}  (latency={fastest['latency_seconds']:.2f}s)")
        print(f"  Avg latency:     {avg_latency:.2f}s")
        print(f"  Avg tokens:      {avg_tokens:.0f}")
        print(f"  Parent run ID:   {parent_run.info.run_id}")
        print()

    # -- Step 4: Query child runs with search_runs() -------------------------
    print("=" * 60)
    print("Step 4: Querying child runs with search_runs()")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name(
        "L2/M1_advanced_tracking/1_nested_runs"
    )
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        order_by=["metrics.response_length DESC"],
    )

    # Build a clean summary table
    summary_cols = [
        "run_id",
        "params.temperature",
        "tags.prompt_variant",
        "metrics.response_length",
        "metrics.token_count",
        "metrics.latency_seconds",
    ]
    available_cols = [c for c in summary_cols if c in child_runs.columns]
    summary = child_runs[available_cols].copy()
    summary.columns = [c.split(".")[-1] for c in available_cols]

    print()
    print(summary.to_string(index=False))
    print()

    print("=" * 60)
    print("Done! View the nested run hierarchy in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("  Expand the 'LLM Config Sweep' parent run to see children.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M1_advanced_tracking/1_nested_runs")
    run_config_sweep()
