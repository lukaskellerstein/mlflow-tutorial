"""
L1-M1.3 — Search and Query API

Demonstrates MLflow's search and query capabilities:
- mlflow.search_runs() with various filters
- mlflow.search_experiments() to list experiments
- MlflowClient for programmatic access
- Exporting results to pandas DataFrames
"""

import time

import mlflow
from mlflow.tracking import MlflowClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_tracking/3_search_query_api"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"

COLS = ["run_id", "params.prompt_topic", "params.temperature", "metrics.total_tokens"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def call_llm(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
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
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "response_time_seconds": round(elapsed, 3),
    }


def create_sample_runs(client: OpenAI) -> str:
    """Create LLM runs with different configs to query later."""
    section("Step 1: Creating sample runs with different configurations")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    configs = [
        {"topic": "transformers", "prompt": "Explain transformer architecture in 2 sentences.",
         "temperature": 0.3, "system_prompt": "You are a concise ML tutor."},
        {"topic": "transformers", "prompt": "Explain transformer architecture in 2 sentences.",
         "temperature": 0.7, "system_prompt": "You are a concise ML tutor."},
        {"topic": "transformers", "prompt": "Explain transformer architecture in 2 sentences.",
         "temperature": 1.0, "system_prompt": "You are a creative writer."},
        {"topic": "rag", "prompt": "What is retrieval-augmented generation?",
         "temperature": 0.3, "system_prompt": "You are a concise ML tutor."},
        {"topic": "rag", "prompt": "What is retrieval-augmented generation?",
         "temperature": 0.7, "system_prompt": None},
        {"topic": "agents", "prompt": "What are AI agents and why do they matter?",
         "temperature": 0.7, "system_prompt": "You are a concise ML tutor."},
    ]

    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['topic']}_t{cfg['temperature']}"):
            mlflow.log_params({
                "model": MODEL,
                "prompt_topic": cfg["topic"],
                "temperature": cfg["temperature"],
                "max_tokens": 128,
                "has_system_prompt": cfg["system_prompt"] is not None,
            })

            result = call_llm(
                client, cfg["prompt"],
                temperature=cfg["temperature"],
                system_prompt=cfg["system_prompt"],
            )

            mlflow.log_metrics({
                "response_time_seconds": result["response_time_seconds"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            })

            mlflow.set_tag("lesson", "L1-M1.3")
            print(f"  {cfg['topic']:15s}  temp={cfg['temperature']}"
                  f"  tokens={result['total_tokens']:>4d}"
                  f"  time={result['response_time_seconds']}s")

    return experiment.experiment_id


def demo_search_runs(experiment_id: str) -> None:
    """Show various search_runs() queries."""

    section("Step 2: search_runs — all runs (no filter)")
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    print(f"  Total runs found: {len(all_runs)}")
    print(all_runs[COLS].to_string(index=False))

    section("Step 3: search_runs — filter by temperature")
    low_temp = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.temperature = '0.3'",
    )
    print(f"  Runs with temperature 0.3: {len(low_temp)}")
    if not low_temp.empty:
        print(low_temp[COLS].to_string(index=False))

    section("Step 4: search_runs — filter by topic")
    topic_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.prompt_topic = 'transformers'",
    )
    print(f"  Transformer runs: {len(topic_runs)}")
    if not topic_runs.empty:
        print(topic_runs[COLS].to_string(index=False))

    section("Step 5: search_runs — order by total tokens DESC")
    ordered = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.total_tokens DESC"],
    )
    print("  Runs ranked by total tokens (most first):")
    print(ordered[COLS].to_string(index=False))

    section("Step 6: Combined filter — transformers AND high token usage")
    combined = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.prompt_topic = 'transformers' AND metrics.total_tokens > 100",
    )
    print(f"  Matching runs: {len(combined)}")
    if not combined.empty:
        print(combined[COLS].to_string(index=False))


def demo_search_experiments() -> None:
    """List experiments on the tracking server."""
    section("Step 7: search_experiments — list all experiments")
    experiments = mlflow.search_experiments()
    print(f"  Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"    [{exp.experiment_id}] {exp.name}")


def demo_mlflow_client(experiment_id: str) -> None:
    """Use MlflowClient for programmatic access."""
    section("Step 8: MlflowClient — programmatic access")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment(experiment_id)
    print(f"  Experiment name  : {experiment.name}")
    print(f"  Experiment ID    : {experiment.experiment_id}")
    print(f"  Artifact location: {experiment.artifact_location}\n")

    best_runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.response_time_seconds ASC"],
        max_results=1,
    )
    if best_runs:
        best = best_runs[0]
        print(f"  Fastest run ID   : {best.info.run_id}")
        print(f"  Run name         : {best.info.run_name}")
        print(f"  Topic            : {best.data.params.get('prompt_topic')}")
        print(f"  Temperature      : {best.data.params.get('temperature')}")
        print(f"  Response time    : {best.data.metrics.get('response_time_seconds')}s")
        print(f"  Total tokens     : {best.data.metrics.get('total_tokens')}")


def demo_dataframe_export(experiment_id: str) -> None:
    """Export search results to a pandas DataFrame and summarize."""
    section("Step 9: DataFrame export — summary statistics")
    df = mlflow.search_runs(experiment_ids=[experiment_id])

    summary = (
        df.groupby("params.prompt_topic")["metrics.total_tokens"]
        .agg(["count", "mean", "max"])
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


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    experiment_id = create_sample_runs(client)
    demo_search_runs(experiment_id)
    demo_search_experiments()
    demo_mlflow_client(experiment_id)
    demo_dataframe_export(experiment_id)
    section(f"Done! Open MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Look for experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
