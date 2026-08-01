"""L2-M4.2 -- GAIA General Assistant Benchmark.

Runs a ReAct agent against GAIA Level 1 tasks (multi-step
reasoning + tool use) and tracks per-task results in MLflow.
Compares two agent configurations side-by-side.
"""

import time

import datasets
import mlflow
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

SAMPLE_SIZE = 5


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return f"Cannot evaluate: {expression}"
    except Exception as e:
        return f"Error: {e}"


@tool
def knowledge_lookup(query: str) -> str:
    """Look up factual information to answer a knowledge question."""
    return (
        f"Knowledge lookup for: {query[:200]}. "
        "Based on available information, the relevant facts are: "
        "This requires multi-step reasoning to combine known facts "
        "and arrive at the final answer."
    )


@tool
def text_analyzer(text: str) -> str:
    """Analyze text content — extract key facts, count items, or summarize."""
    words = text.split()
    return (
        f"Text analysis: {len(words)} words. "
        f"Key content: {' '.join(words[:30])}... "
        "Analysis complete — use the extracted facts to form your answer."
    )


TOOLS = [calculator, knowledge_lookup, text_analyzer]


SYSTEM_PROMPT = (
    "You are a general-purpose AI assistant solving benchmark tasks. "
    "Use the available tools to reason step-by-step. "
    "Give a short, precise final answer."
)


def build_prompt(question: str) -> str:
    """Build user message from a GAIA question."""
    return f"## Question\n{question}"


def run_instance(
    agent, question: str, expected: str, level: int, task_id: str, config_name: str
) -> dict:
    """Run the agent on one GAIA instance and log to MLflow."""
    with mlflow.start_run(run_name=f"{config_name}_{task_id[:8]}", nested=True):
        mlflow.log_params({
            "task_id": task_id,
            "level": level,
            "config": config_name,
        })
        start = time.perf_counter()
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(question)}]})
            latency = time.perf_counter() - start
            answer = result["messages"][-1].content
            answer_short = answer.strip().split("\n")[-1].strip()
            exact_match = expected.lower().strip() in answer_short.lower()
            mlflow.log_metrics({
                "latency_s": round(latency, 2),
                "exact_match": int(exact_match),
                "response_length": len(answer),
                "num_messages": len(result["messages"]),
            })
            mlflow.set_tag("status", "success")
            print(f"  [{task_id[:8]}] L{level} match={exact_match} latency={latency:.1f}s")
            record = {
                "task_id": task_id, "level": level, "config": config_name,
                "latency_s": round(latency, 2), "exact_match": exact_match,
                "response_length": len(answer), "status": "success",
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            mlflow.log_metrics({"latency_s": round(latency, 2), "exact_match": 0})
            mlflow.set_tag("status", "error")
            mlflow.set_tag("error", str(exc)[:200])
            print(f"  [{task_id[:8]}] ERROR: {exc}")
            record = {
                "task_id": task_id, "level": level, "config": config_name,
                "latency_s": round(latency, 2), "exact_match": False,
                "response_length": 0, "status": f"error: {exc}",
            }
    return record


def run_config(
    name: str, temperature: float, instances: list[dict]
) -> list[dict]:
    """Run the agent across all GAIA instances for a given config."""
    print(f"\n{'=' * 60}")
    print(f"Config: {name}  (temperature={temperature})")
    print("=" * 60)

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-4-26b-a4b",
        temperature=temperature,
        max_tokens=1024,  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
    )
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{name}", nested=True):
        mlflow.log_params({
            "temperature": temperature,
            "model": "google/gemma-4-26b-a4b",
            "sample_size": len(instances),
        })
        for inst in instances:
            results.append(run_instance(
                agent,
                question=inst["Question"],
                expected=inst.get("Final answer", ""),
                level=inst.get("Level", 1),
                task_id=inst.get("task_id", "unknown"),
                config_name=name,
            ))

        df = pd.DataFrame(results)
        mlflow.log_metrics({
            "accuracy": round(float(df["exact_match"].mean()), 3),
            "avg_latency_s": round(float(df["latency_s"].mean()), 2),
            "success_rate": round((df["status"] == "success").mean(), 3),
        })
        csv_path = f"/tmp/gaia_{name}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print a comparison table across configurations."""
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    df = pd.DataFrame(all_results)
    summary = df.groupby("config").agg(
        accuracy=("exact_match", "mean"),
        avg_latency=("latency_s", "mean"),
        success_rate=("status", lambda s: (s == "success").mean()),
    )
    print(summary.to_string())

    if df["level"].nunique() > 1:
        print("\nAccuracy by Level:")
        by_level = df.groupby(["config", "level"])["exact_match"].mean().unstack()
        print(by_level.to_string())
    print()


def main() -> None:
    print("=" * 60)
    print("L2-M4.2 -- GAIA General Assistant Benchmark")
    print("=" * 60)

    mlflow.langchain.autolog()

    print("\nStep 1: Loading GAIA dataset ...")
    ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    sample = [ds[i] for i in range(min(SAMPLE_SIZE, len(ds)))]
    print(f"  Loaded {len(ds)} Level-1 instances, using {len(sample)} for demo")

    print("\nStep 2: Running benchmark ...")
    configs = [("focused", 0.3), ("creative", 0.7)]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="gaia_benchmark"):
        mlflow.log_params({"dataset": "GAIA", "level": 1, "sample_size": len(sample)})
        mlflow.set_tag("task", "gaia_benchmark")
        for name, temp in configs:
            all_results.extend(run_config(name, temp, sample))
        combined_csv = "/tmp/gaia_combined.csv"
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    print_summary(all_results)
    print("=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M4_agent_benchmarks/2_gaia")
    main()
