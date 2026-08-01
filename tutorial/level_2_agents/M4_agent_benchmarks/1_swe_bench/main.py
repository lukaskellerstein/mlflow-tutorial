"""L2-M4.1 -- SWE-Bench Evaluation Pipeline.

Builds a coding agent with LangChain v1.0+, runs it against
SWE-Bench Verified instances, and tracks results in MLflow.
Two agent configurations (temperature 0.3 vs 0.7) are compared.
"""

import time

import datasets
import mlflow
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

SAMPLE_SIZE = 5

# -- Tools for the coding agent ------------------------------------------------

@tool
def analyze_code(problem: str) -> str:
    """Analyze a coding problem and identify the root cause."""
    return (
        f"Analysis: The problem involves {problem[:200]}... "
        "Key areas: error handling, edge cases, type checking. "
        "Approach: trace the failing path and apply a minimal fix."
    )

@tool
def generate_patch(analysis: str, repo: str) -> str:
    """Generate a unified diff patch based on the analysis."""
    return (
        f"--- a/{repo}/fix.py\n+++ b/{repo}/fix.py\n"
        "@@ -1,5 +1,5 @@\n-# Original code with bug\n+# Fixed code\n"
        f" # Based on: {analysis[:80]}\n"
        " def fixed_function():\n"
        "-    pass  # buggy\n+    return True  # corrected\n"
    )

TOOLS = [analyze_code, generate_patch]

# -- Helpers -------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a software engineer. When given a coding problem, "
    "use analyze_code to analyze it first, then generate_patch to produce a diff fix."
)


def build_prompt(instance: dict) -> str:
    """Build the user message from a SWE-Bench instance."""
    hints = instance.get("hints_text", "") or ""
    return (
        f"Repository: {instance['repo']}\n\n"
        f"## Problem\n{instance['problem_statement']}\n\n"
        f"## Hints\n{hints}"
    )

def run_instance(agent, instance: dict, config_name: str) -> dict:
    """Run the agent on one SWE-Bench instance and log to MLflow."""
    iid, repo = instance["instance_id"], instance["repo"]
    with mlflow.start_run(run_name=f"{config_name}_{iid}", nested=True):
        mlflow.log_params({"instance_id": iid, "repo": repo, "config": config_name})
        start = time.perf_counter()
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(instance)}]})
            latency = time.perf_counter() - start
            last_msg = result["messages"][-1].content
            patch_ok = "+++" in last_msg or "---" in last_msg
            mlflow.log_metrics({"latency_s": round(latency, 2),
                                "patch_generated": int(patch_ok),
                                "response_length": len(last_msg)})
            mlflow.set_tag("status", "success")
            print(f"  [{iid}] latency={latency:.1f}s patch={patch_ok}")
            record = {"instance_id": iid, "repo": repo, "config": config_name,
                      "latency_s": round(latency, 2), "patch_generated": patch_ok,
                      "response_length": len(last_msg), "status": "success"}
        except Exception as exc:
            latency = time.perf_counter() - start
            mlflow.log_metrics({"latency_s": round(latency, 2), "patch_generated": 0})
            mlflow.set_tag("status", "error")
            mlflow.set_tag("error", str(exc)[:200])
            print(f"  [{iid}] ERROR: {exc}")
            record = {"instance_id": iid, "repo": repo, "config": config_name,
                      "latency_s": round(latency, 2), "patch_generated": False,
                      "response_length": 0, "status": f"error: {exc}"}
    return record

def run_config(name: str, temperature: float, instances: list[dict]) -> list[dict]:
    """Run the agent across all instances for a given config."""
    print(f"\n{'=' * 60}")
    print(f"Config: {name}  (temperature={temperature})")
    print("=" * 60)
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                     model="google/gemma-4-26b-a4b", temperature=temperature,
                     max_tokens=1024)  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    results: list[dict] = []
    with mlflow.start_run(run_name=f"config_{name}", nested=True):
        mlflow.log_params({"temperature": temperature,
                           "model": "google/gemma-4-26b-a4b",
                           "sample_size": len(instances)})
        for inst in instances:
            results.append(run_instance(agent, inst, name))
        df = pd.DataFrame(results)
        mlflow.log_metrics({"avg_latency_s": round(float(df["latency_s"].mean()), 2),
                            "patch_rate": round(float(df["patch_generated"].mean()), 2),
                            "total_response_chars": int(df["response_length"].sum())})
        csv_path = f"/tmp/swe_bench_{name}.csv"
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
        avg_latency=("latency_s", "mean"),
        patch_rate=("patch_generated", "mean"),
        avg_response_len=("response_length", "mean"),
        success_count=("status", lambda s: (s == "success").sum()),
    )
    print(summary.to_string())
    print()

# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("L2-M4.1 -- SWE-Bench Evaluation Pipeline")
    print("=" * 60)

    mlflow.langchain.autolog()

    print("\nStep 1: Loading SWE-Bench Verified dataset ...")
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    sample = [ds[i] for i in range(SAMPLE_SIZE)]
    print(f"  Loaded {len(ds)} instances, using {SAMPLE_SIZE} for demo")

    configs = [("precise", 0.3), ("creative", 0.7)]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="swe_bench_eval"):
        mlflow.log_param("dataset", "SWE-bench_Verified")
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.set_tag("task", "swe_bench_evaluation")
        for name, temp in configs:
            all_results.extend(run_config(name, temp, sample))
        combined_csv = "/tmp/swe_bench_combined.csv"
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    print_summary(all_results)
    print("=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M4_agent_benchmarks/1_swe_bench")
    main()
