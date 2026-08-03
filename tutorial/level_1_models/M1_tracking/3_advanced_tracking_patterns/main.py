"""
L1-M1.3 -- Advanced Tracking Patterns

Covers nested runs (parent-child hierarchies, config sweeps), async
logging (enable_async_logging, sync vs async timing), and artifact
organization patterns (organized subfolders, bulk upload).
"""

import json
import os
import tempfile
import time
from typing import cast

import mlflow
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5555"
EXPERIMENT_NAME = "L1/M1_tracking/3_advanced_tracking_patterns"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"


def call_llm(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
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
        "total_tokens": response.usage.total_tokens if response.usage else 0,
        "response_time_seconds": round(elapsed, 3),
        "response_length": len(choice.message.content or ""),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ── Part 1: Nested Runs ───────────────────────────────────────────────

TEMPERATURES = [0.3, 0.7, 1.0]
PROMPT_VARIANTS: dict[str, str] = {
    "concise": "You are a helpful assistant. Be concise and brief. Answer in 2-3 sentences maximum.",
    "detailed": "You are a helpful assistant. Be detailed and thorough.",
    "creative": "You are a helpful assistant. Be creative. Use analogies and vivid language.",
}
TEST_QUESTION = "Explain what MLflow is and why it is useful."


def part1_nested_runs(client: OpenAI) -> None:
    """Run an LLM config sweep using nested parent-child runs."""

    section("Part 1: Nested Runs -- LLM Configuration Sweep")
    num_configs = len(TEMPERATURES) * len(PROMPT_VARIANTS)
    print(f"  Model:           {MODEL}")
    print(f"  Temperatures:    {TEMPERATURES}")
    print(f"  Prompt variants: {list(PROMPT_VARIANTS.keys())}")
    print(f"  Total configs:   {num_configs}")
    print(f"  Question:        {TEST_QUESTION}")

    section("Step 1: Running sweep (nested runs)")
    results: list[dict] = []

    with mlflow.start_run(run_name="LLM Config Sweep") as parent_run:
        mlflow.set_tags(
            {
                "sweep_type": "llm_config_sweep",
                "model": MODEL,
                "num_configs": str(num_configs),
            }
        )

        for temperature in TEMPERATURES:
            for variant_name, system_prompt in PROMPT_VARIANTS.items():
                run_label = f"temp_{temperature}_style_{variant_name}"

                with mlflow.start_run(run_name=run_label, nested=True):
                    mlflow.log_params(
                        {
                            "temperature": temperature,
                            "prompt_variant": variant_name,
                            "model": MODEL,
                        }
                    )
                    mlflow.set_tags(
                        {
                            "prompt_variant": variant_name,
                            "temperature": str(temperature),
                        }
                    )

                    r = call_llm(client, TEST_QUESTION, temperature=temperature, system_prompt=system_prompt)

                    mlflow.log_metrics(
                        {
                            "response_length": r["response_length"],
                            "token_count": r["total_tokens"],
                            "latency_seconds": r["response_time_seconds"],
                        }
                    )

                    results.append(
                        {
                            "run_name": run_label,
                            "temperature": temperature,
                            "prompt_variant": variant_name,
                            "response_length": r["response_length"],
                            "token_count": r["total_tokens"],
                            "latency_seconds": r["response_time_seconds"],
                        }
                    )

                    print(
                        f"  {run_label:35s}  length={r['response_length']:5d}"
                        f"  tokens={r['total_tokens']:5d}  latency={r['response_time_seconds']:.2f}s"
                    )

        # Parent summary
        section("Step 2: Parent-run summary")
        best_by_length = max(results, key=lambda x: x["response_length"])
        shortest = min(results, key=lambda x: x["response_length"])
        fastest = min(results, key=lambda x: x["latency_seconds"])
        avg_latency = sum(x["latency_seconds"] for x in results) / len(results)

        mlflow.log_params(
            {
                "best_config_by_length": best_by_length["run_name"],
                "fastest_config": fastest["run_name"],
            }
        )
        mlflow.log_metrics(
            {
                "avg_latency_seconds": round(avg_latency, 3),
                "max_response_length": best_by_length["response_length"],
                "min_response_length": shortest["response_length"],
            }
        )

        print(f"  Most detailed: {best_by_length['run_name']}  (length={best_by_length['response_length']})")
        print(f"  Most concise:  {shortest['run_name']}  (length={shortest['response_length']})")
        print(f"  Fastest:       {fastest['run_name']}  (latency={fastest['latency_seconds']:.2f}s)")
        print(f"  Avg latency:   {avg_latency:.2f}s")
        print(f"  Parent run ID: {parent_run.info.run_id}")

    # Query children
    section("Step 3: Querying child runs with search_runs()")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME!r} not found")
    child_runs = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
            order_by=["metrics.response_length DESC"],
        ),
    )
    summary_cols = [
        "params.temperature",
        "tags.prompt_variant",
        "metrics.response_length",
        "metrics.token_count",
        "metrics.latency_seconds",
    ]
    available = [c for c in summary_cols if c in child_runs.columns]
    if available:
        display = child_runs[available].copy()
        display.columns = [c.split(".")[-1] for c in available]
        print(display.to_string(index=False))


# ── Part 2: Async Logging ─────────────────────────────────────────────

EVAL_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms.",
    "What is the difference between AI and ML?",
    "How does gradient descent work?",
    "What is overfitting and how do you prevent it?",
    "Explain the bias-variance tradeoff.",
    "What are transformers in deep learning?",
    "How does backpropagation work?",
]


def part2_async_logging(client: OpenAI) -> None:
    """Demonstrate async logging and sync vs async timing comparison."""

    section("Part 2: Async Logging")

    # Step 4: Async step-based logging
    section("Step 4: Async step-based logging")
    mlflow.config.enable_async_logging(True)
    print("  Async logging ENABLED")
    print(f"  Processing {len(EVAL_PROMPTS)} prompts through LLM...\n")

    with mlflow.start_run(run_name="async_batch_eval") as run:
        mlflow.log_params({"model": MODEL, "temperature": "0.7", "num_prompts": len(EVAL_PROMPTS)})

        for i, prompt in enumerate(EVAL_PROMPTS):
            r = call_llm(client, prompt, max_tokens=1024)

            mlflow.log_metric("response_length", r["response_length"], step=i)
            mlflow.log_metric("latency_ms", r["response_time_seconds"] * 1000, step=i)
            mlflow.log_metric("token_count", r["total_tokens"], step=i)

            print(
                f"    [{i:2d}] {prompt[:45]:<45s}  "
                f"latency={r['response_time_seconds']:.2f}s  "
                f"tokens={r['total_tokens']:3d}"
            )

    mlflow.config.enable_async_logging(False)
    print(f"\n  Run ID: {run.info.run_id}")
    print("  Step-based metrics logged asynchronously.")

    # Step 5: Sync vs async timing comparison
    section("Step 5: Sync vs Async timing comparison")

    print("  Pre-generating LLM responses for fair comparison...")
    pre_results = []
    for prompt in EVAL_PROMPTS[:6]:
        r = call_llm(client, prompt, max_tokens=1024)
        pre_results.append((r["response_length"], r["response_time_seconds"], r["total_tokens"]))
    print(f"  Collected {len(pre_results)} responses.\n")

    sync_elapsed = 0.0
    async_elapsed = 0.0

    # Synchronous
    mlflow.config.enable_async_logging(False)
    print("  Synchronous logging...")
    with mlflow.start_run(run_name="sync_timing_test"):
        t_start = time.perf_counter()
        for i, (resp_len, lat, tok) in enumerate(pre_results):
            mlflow.log_metric("response_length", resp_len, step=i)
            mlflow.log_metric("latency_s", lat, step=i)
            mlflow.log_metric("token_count", tok, step=i)
        sync_elapsed = time.perf_counter() - t_start
    print(f"    Time: {sync_elapsed:.4f}s")

    # Asynchronous
    mlflow.config.enable_async_logging(True)
    print("\n  Asynchronous logging...")
    with mlflow.start_run(run_name="async_timing_test"):
        t_start = time.perf_counter()
        for i, (resp_len, lat, tok) in enumerate(pre_results):
            mlflow.log_metric("response_length", resp_len, step=i)
            mlflow.log_metric("latency_s", lat, step=i)
            mlflow.log_metric("token_count", tok, step=i)
        async_elapsed = time.perf_counter() - t_start
    mlflow.config.enable_async_logging(False)
    print(f"    Time: {async_elapsed:.4f}s")

    print("\n  Results:")
    print(f"    Sync:  {sync_elapsed:.4f}s")
    print(f"    Async: {async_elapsed:.4f}s")
    if sync_elapsed > 0:
        speedup = sync_elapsed / max(async_elapsed, 0.0001)
        print(f"    Speedup: {speedup:.1f}x")
    print("  Async logging returns immediately, offloading I/O to a background thread.")


# ── Part 3: Artifact Organization Patterns ────────────────────────────


def part3_artifact_organization(client: OpenAI) -> None:
    """Demonstrate organized artifact subfolder structure and bulk uploads."""

    section("Part 3: Artifact Organization Patterns")
    print("  Best practice: use artifact_path to create a clean folder structure.")

    with mlflow.start_run(run_name="organized_artifacts") as run:
        # Step 6: Organized subfolder structure
        section("Step 6: Organized artifact subfolders")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Config subfolder hierarchy
            llm_config = {"model": MODEL, "temperature": 0.7, "max_tokens": 1024}
            config_path = os.path.join(tmpdir, "llm_config.json")
            with open(config_path, "w") as f:
                json.dump(llm_config, f, indent=2)
            mlflow.log_artifact(config_path, artifact_path="config/llm")
            print("  Logged -> config/llm/llm_config.json")

            eval_config = {
                "num_prompts": 8,
                "judge_model": "gemma-4-26b",
                "metrics": ["quality", "latency"],
            }
            eval_path = os.path.join(tmpdir, "eval_config.json")
            with open(eval_path, "w") as f:
                json.dump(eval_config, f, indent=2)
            mlflow.log_artifact(eval_path, artifact_path="config/evaluation")
            print("  Logged -> config/evaluation/eval_config.json")

        # Run LLM and log responses in organized folders
        prompts = ["What is MLflow?", "What is experiment tracking?", "What are AI agents?"]
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, prompt in enumerate(prompts):
                r = call_llm(client, prompt, max_tokens=1024)
                response_path = os.path.join(tmpdir, f"prompt_{i}.txt")
                with open(response_path, "w") as f:
                    f.write(f"Prompt: {prompt}\n\n{r['content']}")
                mlflow.log_artifact(response_path, artifact_path="responses")
                print(f"  Logged -> responses/prompt_{i}.txt")

        # Step 7: Bulk directory upload
        section("Step 7: Bulk directory upload")

        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = os.path.join(tmpdir, "reports")
            os.makedirs(reports_dir)
            for split in ["train", "validation", "test"]:
                report = {
                    "split": split,
                    "n_samples": {"train": 800, "validation": 100, "test": 100}[split],
                    "avg_tokens": round(50 + 20 * len(split), 1),
                }
                report_path = os.path.join(reports_dir, f"{split}_report.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

            mlflow.log_artifacts(reports_dir, artifact_path="reports")
            print("  Logged directory -> reports/")
            print("    - reports/train_report.json")
            print("    - reports/validation_report.json")
            print("    - reports/test_report.json")

        print(f"\n  Run ID: {run.info.run_id}")
        print("\n  Recommended artifact folder structure:")
        print("    config/llm/         -- LLM configuration")
        print("    config/evaluation/  -- evaluation settings")
        print("    responses/          -- individual LLM responses")
        print("    reports/            -- evaluation reports")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    part1_nested_runs(client)
    part2_async_logging(client)
    part3_artifact_organization(client)

    section("Done!")
    print(f"  Open the MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  You will see:")
    print("    - 'LLM Config Sweep' parent run with 9 nested children")
    print("    - 'async_batch_eval' with step-based metric charts")
    print("    - 'sync_timing_test' / 'async_timing_test' for comparison")
    print("    - 'organized_artifacts' with clean folder structure")


if __name__ == "__main__":
    main()
