"""
L1-M1.2 -- Search, Query, and MlflowClient

Covers the fluent search API (search_runs with filters/ordering,
search_experiments) and the full MlflowClient API (create/get/update/
delete/restore experiments and runs, ViewType, comparison reports).
"""

import time
from typing import cast

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5555"
EXPERIMENT_NAME = "L1/M1_tracking/2_search_query_mlflowclient"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"

COLS = ["run_id", "params.prompt_topic", "params.temperature", "metrics.total_tokens"]


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
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
        "response_time_seconds": round(elapsed, 3),
        "response_length": len(choice.message.content or ""),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ── Part A: Fluent Search API ─────────────────────────────────────────


def create_sample_runs(client: OpenAI) -> str:
    """Create LLM runs with different configs to query later."""
    section("Step 1: Creating sample runs with different configurations")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    configs = [
        {
            "topic": "transformers",
            "prompt": "Explain transformer architecture in 2 sentences.",
            "temperature": 0.3,
            "system_prompt": "You are a concise ML tutor.",
        },
        {
            "topic": "transformers",
            "prompt": "Explain transformer architecture in 2 sentences.",
            "temperature": 0.7,
            "system_prompt": "You are a concise ML tutor.",
        },
        {
            "topic": "transformers",
            "prompt": "Explain transformer architecture in 2 sentences.",
            "temperature": 1.0,
            "system_prompt": "You are a creative writer.",
        },
        {
            "topic": "rag",
            "prompt": "What is retrieval-augmented generation?",
            "temperature": 0.3,
            "system_prompt": "You are a concise ML tutor.",
        },
        {
            "topic": "rag",
            "prompt": "What is retrieval-augmented generation?",
            "temperature": 0.7,
            "system_prompt": None,
        },
        {
            "topic": "agents",
            "prompt": "What are AI agents and why do they matter?",
            "temperature": 0.7,
            "system_prompt": "You are a concise ML tutor.",
        },
    ]

    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['topic']}_t{cfg['temperature']}"):
            mlflow.log_params(
                {
                    "model": MODEL,
                    "prompt_topic": cfg["topic"],
                    "temperature": cfg["temperature"],
                    "max_tokens": 1024,
                    "has_system_prompt": cfg["system_prompt"] is not None,
                }
            )

            result = call_llm(
                client,
                cfg["prompt"],
                temperature=cfg["temperature"],
                system_prompt=cfg["system_prompt"],
            )

            mlflow.log_metrics(
                {
                    "response_time_seconds": result["response_time_seconds"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                }
            )

            mlflow.set_tag("lesson", "L1-M1.2")
            print(
                f"  {cfg['topic']:15s}  temp={cfg['temperature']}"
                f"  tokens={result['total_tokens']:>4d}"
                f"  time={result['response_time_seconds']}s"
            )

    return experiment.experiment_id


def demo_search_runs(experiment_id: str) -> None:
    """Show various search_runs() queries."""

    section("Step 2: search_runs -- all runs (no filter)")
    all_runs = cast(pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment_id]))
    print(f"  Total runs found: {len(all_runs)}")
    available = [c for c in COLS if c in all_runs.columns]
    print(all_runs[available].to_string(index=False))

    section("Step 3: search_runs -- filter by temperature")
    low_temp = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="params.temperature = '0.3'",
        ),
    )
    print(f"  Runs with temperature 0.3: {len(low_temp)}")
    if not low_temp.empty:
        available = [c for c in COLS if c in low_temp.columns]
        print(low_temp[available].to_string(index=False))

    section("Step 4: search_runs -- order by total tokens DESC")
    ordered = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.total_tokens DESC"],
        ),
    )
    print("  Runs ranked by total tokens (most first):")
    available = [c for c in COLS if c in ordered.columns]
    print(ordered[available].to_string(index=False))

    section("Step 5: Combined filter -- topic AND metric threshold")
    combined = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="params.prompt_topic = 'transformers' AND metrics.total_tokens > 100",
        ),
    )
    print(f"  Matching runs: {len(combined)}")
    if not combined.empty:
        available = [c for c in COLS if c in combined.columns]
        print(combined[available].to_string(index=False))


def demo_search_experiments() -> None:
    """List experiments on the tracking server."""
    section("Step 6: search_experiments -- list all experiments")
    experiments = mlflow.search_experiments()
    print(f"  Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"    [{exp.experiment_id}] {exp.name}")


def demo_dataframe_export(experiment_id: str) -> None:
    """Export search results to a pandas DataFrame and summarize."""
    section("Step 7: DataFrame export -- summary statistics")
    df = cast(pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment_id]))

    if "params.prompt_topic" in df.columns and "metrics.total_tokens" in df.columns:
        summary = (
            cast(
                pd.DataFrame,
                df.groupby("params.prompt_topic")["metrics.total_tokens"].agg(["count", "mean", "max"]),
            )
            .rename(columns={"count": "runs", "mean": "avg_tokens", "max": "max_tokens"})
            .sort_values("max_tokens", ascending=False)
        )
        print("  Token usage summary by topic:")
        print(summary.to_string())

        fastest = df.loc[df["metrics.response_time_seconds"].idxmin()]
        print(f"\n  Fastest run: {fastest['run_id']}")
        print(f"    topic       = {fastest['params.prompt_topic']}")
        print(f"    temperature = {fastest['params.temperature']}")
        print(f"    time        = {fastest['metrics.response_time_seconds']}s")


# ── Part B: MlflowClient API ──────────────────────────────────────────

TEST_QUESTION = "What are the benefits of experiment tracking in machine learning?"

LLM_CONFIGS = [
    {"name": "conservative", "temperature": 0.2, "system_prompt": "Be precise and factual."},
    {"name": "balanced", "temperature": 0.7, "system_prompt": "Be helpful and clear."},
    {"name": "creative", "temperature": 1.0, "system_prompt": "Be creative and engaging."},
]


def demo_mlflowclient(llm_client: OpenAI) -> None:
    """Demonstrate full MlflowClient CRUD operations."""
    ml_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # -- Create experiment and runs --
    section("Step 8: MlflowClient -- create experiment and runs")

    client_exp_name = f"{EXPERIMENT_NAME}/client_demo"
    experiment = ml_client.get_experiment_by_name(client_exp_name)
    if experiment and experiment.lifecycle_stage == "active":
        experiment_id = experiment.experiment_id
        print(f"  Using existing experiment: {client_exp_name}")
    else:
        experiment_id = ml_client.create_experiment(
            client_exp_name,
            tags={"project": "mlflow-tutorial", "level": "1"},
        )
        print(f"  Created experiment: {client_exp_name} (id={experiment_id})")

    run_ids: list[str] = []
    print("\n  Calling LLM with 3 configurations via MlflowClient...")
    for config in LLM_CONFIGS:
        run = ml_client.create_run(experiment_id, run_name=config["name"])
        run_id = run.info.run_id

        ml_client.log_param(run_id, "model", MODEL)
        ml_client.log_param(run_id, "temperature", config["temperature"])
        ml_client.log_param(run_id, "system_prompt", config["system_prompt"])

        result = call_llm(
            llm_client,
            TEST_QUESTION,
            temperature=config["temperature"],
            system_prompt=config["system_prompt"],
        )

        ml_client.log_metric(run_id, "response_length", result["response_length"])
        ml_client.log_metric(run_id, "latency_seconds", result["response_time_seconds"])
        ml_client.log_metric(run_id, "total_tokens", result["total_tokens"])

        ml_client.update_run(run_id, status="FINISHED")
        run_ids.append(run_id)

        print(f"  {config['name']}: latency={result['response_time_seconds']}s, tokens={result['total_tokens']}")

    # -- Query operations --
    section("Step 9: MlflowClient -- query operations")
    runs = ml_client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.latency_seconds ASC"],
    )
    print(f"  search_runs found {len(runs)} run(s) (ordered by latency ASC):")
    for r in runs:
        latency = r.data.metrics.get("latency_seconds", 0)
        print(f"    {r.info.run_name:<20s} latency={latency:.2f}s  (id={r.info.run_id[:8]}...)")

    detail_run = ml_client.get_run(run_ids[0])
    print(f"\n  get_run({run_ids[0][:8]}...) details:")
    print(f"    name:    {detail_run.info.run_name}")
    print(f"    status:  {detail_run.info.status}")
    print(f"    params:  {len(detail_run.data.params)} logged")
    print(f"    metrics: {detail_run.data.metrics}")

    # -- Update, delete, restore --
    section("Step 10: MlflowClient -- update, delete, restore")

    for rid in run_ids:
        ml_client.set_tag(rid, "tutorial_lesson", "L1-M1.2")
    print("  Added tags to all runs.")

    old_name = ml_client.get_run(run_ids[0]).info.run_name
    ml_client.update_run(run_ids[0], name="conservative_renamed")
    print(f"  Renamed run: '{old_name}' -> 'conservative_renamed'")
    ml_client.update_run(run_ids[0], name=old_name)
    print(f"  Renamed back to: '{old_name}'")

    ml_client.delete_run(run_ids[2])
    stage = ml_client.get_run(run_ids[2]).info.lifecycle_stage
    active_count = len(ml_client.search_runs([experiment_id], run_view_type=ViewType.ACTIVE_ONLY))
    all_count = len(ml_client.search_runs([experiment_id], run_view_type=ViewType.ALL))
    print(f"\n  Deleted run {run_ids[2][:8]}... (stage={stage})")
    print(f"    Active: {active_count}, All (incl. deleted): {all_count}")

    ml_client.restore_run(run_ids[2])
    stage = ml_client.get_run(run_ids[2]).info.lifecycle_stage
    print(f"    After restore: stage={stage}")

    # -- Comparison report --
    section("Step 11: Comparison report via MlflowClient")
    runs = ml_client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.latency_seconds ASC"],
    )

    print(f"\n  {'Config':<16s} {'Temp':>6s} {'Latency':>10s} {'Tokens':>8s} {'Resp Len':>10s}")
    print(f"  {'-' * 16} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 10}")
    for r in runs:
        print(
            f"  {r.info.run_name or 'unnamed':<16s} "
            f"{r.data.params.get('temperature', 'N/A'):>6s} "
            f"{r.data.metrics.get('latency_seconds', 0):>10.2f} "
            f"{int(r.data.metrics.get('total_tokens', 0)):>8d} "
            f"{int(r.data.metrics.get('response_length', 0)):>10d}"
        )

    # -- Fluent vs Client summary --
    section("Fluent API vs MlflowClient")
    print("  Fluent API: simple, manages 'active run' automatically.")
    print("    Best for: notebooks, single experiments, quick prototyping.")
    print("  MlflowClient: full CRUD, explicit run_id, no global state.")
    print("    Best for: automation, dashboards, CI/CD, batch operations.")
    print("    Only MlflowClient can: delete/restore runs, rename, create_run().")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    # Part A: Fluent search API
    experiment_id = create_sample_runs(client)
    demo_search_runs(experiment_id)
    demo_search_experiments()
    demo_dataframe_export(experiment_id)

    # Part B: MlflowClient
    demo_mlflowclient(client)

    section("Done!")
    print(f"  Open MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Look for experiments: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
