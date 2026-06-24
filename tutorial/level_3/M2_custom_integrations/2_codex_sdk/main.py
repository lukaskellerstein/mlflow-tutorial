"""
L3-2.2 — Codex SDK + MLflow Integration

Demonstrates how to build MLflow tracing and evaluation for a code generation
agent modelled after the Codex SDK pattern. Since Codex SDK is TypeScript-based
and requires an OpenAI API key, this lesson builds a simulated Codex-style
agent using a local LLM and focuses on the MLflow INTEGRATION PATTERN:

- Custom tracing for a multi-step code generation pipeline
- Code quality scoring with custom metrics
- Strategy comparison (direct vs plan-then-generate)
"""

import time

import mlflow
import pandas as pd

from codegen_agent import CodeGenAgent
from scorers import compute_quality_scores

# ------------------------------------------------------------------ #
# Code Generation Tasks
# ------------------------------------------------------------------ #

TASKS = [
    "Write a Python function to calculate the nth Fibonacci number",
    "Write a Python function to validate email addresses",
    "Write a Python class for a simple key-value store with get, set, and delete",
]


# ------------------------------------------------------------------ #
# Part 1: Run Tasks with MLflow Tracking
# ------------------------------------------------------------------ #


def run_tasks(agent: CodeGenAgent, *, use_plan: bool, strategy: str) -> list[dict]:
    """Run all tasks under a parent MLflow run and log metrics."""
    results = []

    with mlflow.start_run(run_name=f"strategy_{strategy}"):
        mlflow.log_params({
            "strategy": strategy,
            "use_plan": use_plan,
            "model": agent.model,
            "temperature": agent.temperature,
            "num_tasks": len(TASKS),
        })

        for idx, task in enumerate(TASKS, start=1):
            print(f"\n  Task {idx}: {task}")
            start = time.time()
            output = agent.run_pipeline(task, use_plan=use_plan)
            elapsed = time.time() - start

            scores = compute_quality_scores(output["final_code"], task)

            with mlflow.start_run(run_name=f"task_{idx}_{strategy}", nested=True):
                mlflow.log_params({"task": task, "task_index": idx})
                mlflow.log_metrics({
                    "latency_seconds": round(elapsed, 2),
                    "code_length_chars": len(output["final_code"]),
                    "code_lines": len(output["final_code"].strip().splitlines()),
                    **scores,
                })
                mlflow.log_text(output["final_code"], "final_code.py")
                if output["plan"]:
                    mlflow.log_text(output["plan"], "plan.txt")
                mlflow.log_text(output["review_feedback"], "review.txt")

            avg_quality = round(sum(scores.values()) / len(scores), 2)
            print(f"    Latency: {elapsed:.1f}s | Quality: {avg_quality}")
            snippet = output["final_code"].strip().splitlines()
            for line in snippet[:6]:
                print(f"    | {line}")
            if len(snippet) > 6:
                print(f"    | ... ({len(snippet) - 6} more lines)")

            results.append({
                "task_index": idx, "task": task, "strategy": strategy,
                "latency": round(elapsed, 2),
                "code_lines": len(snippet), **scores,
            })

        # Aggregate metrics on parent run
        avg_metrics = {
            "avg_latency": round(
                sum(r["latency"] for r in results) / len(results), 2),
            "avg_completeness": round(
                sum(r["code_completeness"] for r in results) / len(results), 2),
            "avg_error_handling": round(
                sum(r["has_error_handling"] for r in results) / len(results), 2),
            "avg_conventions": round(
                sum(r["follows_conventions"] for r in results) / len(results), 2),
        }
        mlflow.log_metrics(avg_metrics)

    return results


# ------------------------------------------------------------------ #
# Part 2: Compare Strategies
# ------------------------------------------------------------------ #


def compare_strategies(direct: list[dict], planned: list[dict]) -> None:
    """Print a comparison table of both strategies."""
    print("\n" + "=" * 70)
    print("Strategy Comparison: Direct vs Plan-Then-Generate")
    print("=" * 70)

    header = f"{'Metric':<25} {'Direct':>12} {'Planned':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    def avg(data: list[dict], key: str) -> float:
        return sum(r[key] for r in data) / len(data)

    for metric in ["latency", "code_lines", "code_completeness",
                    "has_error_handling", "follows_conventions"]:
        d_val = avg(direct, metric)
        p_val = avg(planned, metric)
        delta = p_val - d_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<23} {d_val:>12.2f} {p_val:>12.2f} {sign}{delta:>9.2f}")

    d_total = (avg(direct, "code_completeness") +
               avg(direct, "has_error_handling") +
               avg(direct, "follows_conventions")) / 3
    p_total = (avg(planned, "code_completeness") +
               avg(planned, "has_error_handling") +
               avg(planned, "follows_conventions")) / 3
    print(f"\n  {'Overall Quality':<23} {d_total:>12.2f} {p_total:>12.2f}")
    winner = "Plan-Then-Generate" if p_total >= d_total else "Direct"
    print(f"\n  Winner: {winner}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    print("=" * 60)
    print("L3-2.2 — Codex SDK + MLflow Integration")
    print("=" * 60)
    print("\nThis lesson demonstrates MLflow integration patterns for")
    print("code generation agents (modelled after the Codex SDK).\n")

    agent = CodeGenAgent(model="gemma4:e2b", temperature=0.3)

    # Strategy 1: Direct generation
    print("-" * 60)
    print("Strategy 1: Direct Generation (no plan)")
    print("-" * 60)
    direct_results = run_tasks(agent, use_plan=False, strategy="direct")

    # Strategy 2: Plan-then-generate
    print("\n" + "-" * 60)
    print("Strategy 2: Plan-Then-Generate")
    print("-" * 60)
    planned_results = run_tasks(agent, use_plan=True, strategy="planned")

    # Compare
    compare_strategies(direct_results, planned_results)

    # Summary table
    print("\n" + "=" * 60)
    print("All Results")
    print("=" * 60)
    df = pd.DataFrame(direct_results + planned_results)
    cols = ["strategy", "task_index", "latency", "code_lines",
            "code_completeness", "has_error_handling", "follows_conventions"]
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("Done! View runs and traces in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("  Experiment: L3/M2_custom_integrations/2_codex_sdk")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M2_custom_integrations/2_codex_sdk")
    main()
