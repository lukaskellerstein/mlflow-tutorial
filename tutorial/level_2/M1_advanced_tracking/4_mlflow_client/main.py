"""
L2-1.4 — MlflowClient: Programmatic Access

MlflowClient low-level API for LLM experiment management: creating
experiments/runs, logging params/metrics/tags, querying with filters,
delete/restore, and building a comparison report across LLM configs.
"""

import time

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType
from openai import OpenAI

EXPERIMENT_NAME = "L2/M1_advanced_tracking/4_mlflow_client"

LLM_CONFIGS = [
    {"name": "conservative", "temperature": 0.2, "system_prompt": "Be precise and factual."},
    {"name": "balanced", "temperature": 0.7, "system_prompt": "Be helpful and clear."},
    {"name": "creative", "temperature": 1.0, "system_prompt": "Be creative and engaging."},
]

TEST_QUESTION = "What are the benefits of experiment tracking in machine learning?"


def call_llm_and_log(
    ml_client: MlflowClient,
    llm_client: OpenAI,
    experiment_id: str,
    config: dict,
) -> str:
    """Call LLM with a config and log everything via MlflowClient."""
    run = ml_client.create_run(experiment_id, run_name=config["name"])
    run_id = run.info.run_id

    # Log parameters
    ml_client.log_param(run_id, "model", "google/gemma-4-e4b")
    ml_client.log_param(run_id, "temperature", config["temperature"])
    ml_client.log_param(run_id, "system_prompt", config["system_prompt"])
    ml_client.log_param(run_id, "question", TEST_QUESTION)

    # Call LLM and measure latency
    start = time.time()
    response = llm_client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": TEST_QUESTION},
        ],
        temperature=config["temperature"],
    )
    latency = time.time() - start

    answer = response.choices[0].message.content or ""
    usage = response.usage

    # Log metrics
    ml_client.log_metric(run_id, "response_length", len(answer))
    ml_client.log_metric(run_id, "latency_seconds", round(latency, 3))
    if usage:
        ml_client.log_metric(run_id, "prompt_tokens", usage.prompt_tokens)
        ml_client.log_metric(run_id, "completion_tokens", usage.completion_tokens)
        ml_client.log_metric(run_id, "total_tokens", usage.total_tokens)

    # Mark the run as finished
    ml_client.update_run(run_id, status="FINISHED")

    print(f"  {config['name']}: latency={latency:.2f}s, "
          f"tokens={usage.total_tokens if usage else 'N/A'}, "
          f"response_length={len(answer)}")
    return run_id


def main() -> None:
    ml_client = MlflowClient()
    llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Part 1: Create experiment and runs
    print("=" * 60)
    print("Part 1: Create experiment and runs via MlflowClient")
    print("=" * 60)
    experiment = ml_client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment and experiment.lifecycle_stage == "active":
        experiment_id = experiment.experiment_id
        print(f"  Using existing experiment: {EXPERIMENT_NAME} (id={experiment_id})")
    else:
        experiment_id = ml_client.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "mlflow-tutorial", "level": "2"},
        )
        print(f"  Created experiment: {EXPERIMENT_NAME} (id={experiment_id})")

    run_ids: list[str] = []
    print("\n  Calling LLM with 3 configurations...")
    for config in LLM_CONFIGS:
        rid = call_llm_and_log(ml_client, llm_client, experiment_id, config)
        run_ids.append(rid)

    # Part 2: Query operations
    print("\n" + "=" * 60)
    print("Part 2: Query operations")
    print("=" * 60)
    experiments = ml_client.search_experiments(
        filter_string=f"name = '{EXPERIMENT_NAME}'",
    )
    exp_by_name = ml_client.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"\n  search_experiments found {len(experiments)} experiment(s)")
    print(f"  get_experiment_by_name: id={exp_by_name.experiment_id}")

    # 2b. search_runs — all runs in this experiment, ordered by latency
    runs = ml_client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.latency_seconds ASC"],
    )
    print(f"\n  search_runs found {len(runs)} run(s) (ordered by latency ASC):")
    for r in runs:
        latency = r.data.metrics.get("latency_seconds", 0)
        print(f"    {r.info.run_name:<20s} latency={latency:.2f}s  "
              f"(id={r.info.run_id[:8]}...)")

    # 2c. search_runs with a filter
    short_runs = ml_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.response_length < 500",
    )
    print(f"\n  Runs with response_length < 500: {len(short_runs)}")

    # 2d. get_run — detailed info for a single run
    detail_run = ml_client.get_run(run_ids[0])
    print(f"\n  get_run({run_ids[0][:8]}...) details:")
    print(f"    name:    {detail_run.info.run_name}")
    print(f"    status:  {detail_run.info.status}")
    print(f"    params:  {len(detail_run.data.params)} logged")
    print(f"    metrics: {detail_run.data.metrics}")

    # Part 3: Update and manage operations
    print("\n" + "=" * 60)
    print("Part 3: Update and manage operations")
    print("=" * 60)
    for rid in run_ids:
        ml_client.set_tag(rid, "tutorial_lesson", "L2-M1.4")
        ml_client.set_tag(rid, "llm_provider", "lm-studio")
    print("\n  Added tags to all runs.")

    # 3b. update_run — rename a run (then rename back)
    old_name = ml_client.get_run(run_ids[0]).info.run_name
    ml_client.update_run(run_ids[0], name="conservative_renamed")
    print(f"  Renamed run: '{old_name}' -> 'conservative_renamed'")
    ml_client.update_run(run_ids[0], name=old_name)

    # 3c. delete_run and restore_run
    ml_client.delete_run(run_ids[2])
    stage = ml_client.get_run(run_ids[2]).info.lifecycle_stage
    active_count = len(ml_client.search_runs(
        [experiment_id], run_view_type=ViewType.ACTIVE_ONLY))
    all_count = len(ml_client.search_runs(
        [experiment_id], run_view_type=ViewType.ALL))
    print(f"\n  Deleted run {run_ids[2][:8]}... (stage={stage})")
    print(f"    Active: {active_count}, All (incl. deleted): {all_count}")

    ml_client.restore_run(run_ids[2])
    stage = ml_client.get_run(run_ids[2]).info.lifecycle_stage
    print(f"    After restore: {stage}")

    # Part 4: Comparison report
    print("\n" + "=" * 60)
    print("Part 4: LLM configuration comparison report")
    print("=" * 60)

    runs = ml_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.model = 'google/gemma-4-e4b'",
        order_by=["metrics.latency_seconds ASC"],
    )

    # Build a pandas DataFrame from the run data
    rows = [{
        "Config": r.info.run_name or "unnamed",
        "Temp": r.data.params.get("temperature", "N/A"),
        "Latency": r.data.metrics.get("latency_seconds", 0.0),
        "Tokens": int(r.data.metrics.get("total_tokens", 0)),
        "Resp Len": int(r.data.metrics.get("response_length", 0)),
    } for r in runs]
    df = pd.DataFrame(rows)

    print(f"\n  {'Config':<16s} {'Temp':>6s} {'Latency':>10s} "
          f"{'Tokens':>8s} {'Resp Len':>10s}")
    print(f"  {'-' * 16} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 10}")
    for _, row in df.iterrows():
        print(f"  {row['Config']:<16s} {str(row['Temp']):>6s} "
              f"{row['Latency']:>10.2f} "
              f"{row['Tokens']:>8d} {row['Resp Len']:>10d}")

    # Identify the fastest configuration
    if runs:
        fastest = runs[0]
        print(f"\n  Fastest config: {fastest.info.run_name} "
              f"(latency={fastest.data.metrics.get('latency_seconds', 0):.2f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("Fluent API vs MlflowClient")
    print("=" * 60)
    print("  Fluent API: simple, manages 'active run' automatically.")
    print("    Best for: notebooks, single experiments, quick prototyping.")
    print("  MlflowClient: full CRUD, explicit run_id, no global state.")
    print("    Best for: automation, dashboards, CI/CD, batch operations.")
    print("\n" + "=" * 60)
    print("Done! View results: http://127.0.0.1:5000/#/experiments")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    main()
