"""
L3-M3.2 — MLflow Plugins and Extensibility

Demonstrates MLflow's plugin architecture through production-quality examples:
- Plugin system overview (entry points, registries, extension types)
- Custom RunContextProvider that auto-tags runs with environment metadata
- MetricAggregator that computes rolling statistics alongside raw metrics
- Custom Model Evaluator plugin for LLM output quality checks
- End-to-end demonstration with all plugins active
"""

import platform
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.registry import _run_context_provider_registry
from pydantic import SecretStr

mlflow.set_tracking_uri("http://127.0.0.1:5555")

EXPERIMENT_NAME = "L3/M3_extensibility/2_plugins"


# ── Part 1: Plugin Architecture Overview ───────────────────────────────── #


def part1_plugin_architecture() -> None:
    """Explain MLflow's plugin system and list all extension points."""
    print("=" * 60)
    print("Part 1: MLflow Plugin Architecture Overview")
    print("=" * 60)

    extension_points = [
        ("mlflow.tracking_store", "Custom tracking backends (scheme-based URI routing)"),
        ("mlflow.artifact_repository", "Custom artifact storage (S3, GCS, custom)"),
        ("mlflow.model_registry_store", "Custom model registry backends"),
        ("mlflow.run_context_provider", "Auto-tag runs with environment context"),
        ("mlflow.request_header_provider", "Custom HTTP headers on API requests"),
        ("mlflow.request_auth_provider", "Custom authentication providers"),
        ("mlflow.default_experiment_provider", "Override default experiment resolution"),
        ("mlflow.model_evaluator", "Custom model evaluation plugins"),
        ("mlflow.project_backend", "Custom execution backends for Projects"),
        ("mlflow.deployments", "Custom deployment targets"),
        ("mlflow.app", "Custom MLflow server applications"),
        ("mlflow.dataset_source", "Custom dataset source types"),
        ("mlflow.dataset_constructor", "Custom dataset constructors"),
    ]

    print("\n  MLflow discovers plugins via Python entry points.")
    print("  A plugin is any installed package that declares an entry point")
    print("  in one of these groups:\n")
    for group, description in extension_points:
        print(f"    {group:<40s} {description}")

    print("\n  Registration patterns:")
    print("    - Scheme-based: entry point name = URI scheme (tracking, artifact, registry)")
    print("    - List-based:   all providers are loaded and iterated (context, headers, auth)")
    print()


# ── Part 2: Custom Run Context Provider ────────────────────────────────── #


class EnvironmentContextProvider(RunContextProvider):
    """Automatically tags every MLflow run with environment metadata.

    In production, you would install this as a package with an entry point:
        [project.entry-points."mlflow.run_context_provider"]
        env_context = "my_plugin:EnvironmentContextProvider"

    For this demo, we register it directly into the provider registry.
    """

    def in_context(self) -> bool:
        """Always active -- environment metadata is always available."""
        return True

    def tags(self) -> dict[str, str]:
        """Return tags describing the current execution environment."""
        result: dict[str, str] = {
            "env.hostname": socket.gethostname(),
            "env.python_version": platform.python_version(),
            "env.platform": platform.system(),
            "env.architecture": platform.machine(),
            "env.timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Attempt to capture git branch
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            result["env.git_branch"] = branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return result


def part2_run_context_provider() -> None:
    """Register and demonstrate the custom RunContextProvider."""
    print("=" * 60)
    print("Part 2: Custom Run Context Provider")
    print("=" * 60)

    # Register our provider into MLflow's global registry
    _run_context_provider_registry.register(EnvironmentContextProvider)
    print("\n  Registered EnvironmentContextProvider into MLflow registry.")
    print("  All new runs will automatically receive environment tags.\n")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="context-provider-demo") as run:
        mlflow.log_param("demo_type", "context_provider")
        mlflow.log_metric("placeholder", 1.0)

    # Fetch the run and display the auto-injected tags
    client = MlflowClient()
    run_data = client.get_run(run.info.run_id)
    env_tags = {k: v for k, v in run_data.data.tags.items() if k.startswith("env.")}
    print("  Auto-injected environment tags:")
    for key, value in sorted(env_tags.items()):
        print(f"    {key:<25s} = {value}")
    print()


# ── Part 3: Custom Metric Logger (MetricAggregator) ───────────────────── #


class MetricAggregator:
    """Wraps MLflow metric logging with rolling aggregation.

    As individual metric values are logged, the aggregator computes
    and logs summary statistics (mean, min, max, p50, p95) alongside
    the raw values. This is useful for tracking metric distributions
    across batches or iterations.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def log(self, key: str, value: float, step: int | None = None) -> None:
        """Log a raw metric value and update aggregated summaries."""
        self._history.setdefault(key, []).append(value)
        mlflow.log_metric(key, value, step=step)

        values = self._history[key]
        if len(values) >= 2:
            sorted_vals = sorted(values)
            aggregates = {
                f"{key}/mean": statistics.mean(values),
                f"{key}/min": min(values),
                f"{key}/max": max(values),
                f"{key}/stdev": statistics.stdev(values),
                f"{key}/p50": sorted_vals[len(sorted_vals) // 2],
                f"{key}/p95": sorted_vals[int(len(sorted_vals) * 0.95)],
                f"{key}/count": float(len(values)),
            }
            mlflow.log_metrics(aggregates, step=step)

    def summary(self, key: str) -> dict[str, float]:
        """Return the current aggregate summary for a metric."""
        values = self._history.get(key, [])
        if not values:
            return {}
        sorted_vals = sorted(values)
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "p50": sorted_vals[len(sorted_vals) // 2],
            "p95": sorted_vals[int(len(sorted_vals) * 0.95)],
            "count": len(values),
        }


def part3_metric_aggregator() -> None:
    """Demonstrate the MetricAggregator with simulated batch metrics."""
    print("=" * 60)
    print("Part 3: Custom Metric Logger (MetricAggregator)")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    aggregator = MetricAggregator()

    with mlflow.start_run(run_name="metric-aggregator-demo"):
        # Simulate per-batch latency and quality metrics
        import random

        random.seed(42)
        latencies = [random.uniform(0.1, 2.0) for _ in range(10)]
        scores = [random.uniform(0.6, 1.0) for _ in range(10)]

        print("\n  Logging per-batch metrics with rolling aggregation:\n")
        for i, (lat, score) in enumerate(zip(latencies, scores)):
            aggregator.log("batch_latency_s", round(lat, 4), step=i)
            aggregator.log("batch_quality_score", round(score, 4), step=i)

        # Display final summaries
        for metric_name in ["batch_latency_s", "batch_quality_score"]:
            summary = aggregator.summary(metric_name)
            print(f"  {metric_name}:")
            for stat, val in summary.items():
                print(f"    {stat:<8s} = {val:.4f}")
            print()


# ── Part 4: Custom Model Evaluator Plugin ──────────────────────────────── #


class LLMOutputEvaluator:
    """Evaluates LLM outputs on multiple quality dimensions.

    In production, this would extend mlflow.models.evaluation.ModelEvaluator
    and be registered via the mlflow.model_evaluator entry point. Here we
    demonstrate the evaluation logic and log results to MLflow directly.

    Checks performed:
    - response_length: character count (flags very short or very long)
    - contains_code: whether the response includes code blocks
    - word_diversity: ratio of unique words to total words
    - sentence_count: number of sentences in the response
    """

    @staticmethod
    def evaluate_response(response: str) -> dict[str, float]:
        """Run all quality checks on a single LLM response."""
        metrics: dict[str, float] = {}

        # Response length
        metrics["eval/char_count"] = float(len(response))
        metrics["eval/word_count"] = float(len(response.split()))

        # Code detection
        has_code = "```" in response or "def " in response or "import " in response
        metrics["eval/contains_code"] = 1.0 if has_code else 0.0

        # Word diversity (unique words / total words)
        words = response.lower().split()
        if words:
            metrics["eval/word_diversity"] = len(set(words)) / len(words)
        else:
            metrics["eval/word_diversity"] = 0.0

        # Sentence count (simple heuristic)
        sentence_endings = sum(1 for c in response if c in ".!?")
        metrics["eval/sentence_count"] = float(max(sentence_endings, 1))

        # Length quality flag (too short < 20 chars, too long > 2000 chars)
        if len(response) < 20:
            metrics["eval/length_quality"] = 0.0  # too short
        elif len(response) > 2000:
            metrics["eval/length_quality"] = 0.5  # verbose
        else:
            metrics["eval/length_quality"] = 1.0  # good

        return metrics


def part4_model_evaluator(llm: ChatOpenAI) -> list[dict[str, Any]]:
    """Run the custom evaluator against LLM responses."""
    print("=" * 60)
    print("Part 4: Custom Model Evaluator Plugin")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    evaluator = LLMOutputEvaluator()

    prompts = [
        "Explain what a Python decorator is in one sentence.",
        "Write a Python function that computes the Fibonacci sequence.",
        "What is MLflow?",
    ]

    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="evaluator-plugin-demo"):
        mlflow.log_param("num_prompts", len(prompts))
        mlflow.log_param("model", "google/gemma-4-26b-a4b")

        print()
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i + 1}: {prompt}")
            response = llm.invoke([{"role": "user", "content": prompt}])
            text = response.content
            print(f"  Response: {text[:100]}{'...' if len(text) > 100 else ''}")

            # Evaluate
            metrics = evaluator.evaluate_response(str(text))
            step_metrics = {k: v for k, v in metrics.items()}
            mlflow.log_metrics(step_metrics, step=i)

            result_entry = {"prompt": prompt, "response": text, **metrics}
            results.append(result_entry)

            print(
                f"  Evaluation: chars={metrics['eval/char_count']:.0f}, "
                f"words={metrics['eval/word_count']:.0f}, "
                f"diversity={metrics['eval/word_diversity']:.2f}, "
                f"code={metrics['eval/contains_code']:.0f}, "
                f"length_quality={metrics['eval/length_quality']:.1f}"
            )
            print()

        # Log summary table
        df = pd.DataFrame(results)
        mlflow.log_table(df, artifact_file="evaluation_results.json")
        avg_diversity = float(df["eval/word_diversity"].mean())
        avg_length = float(df["eval/char_count"].mean())
        mlflow.log_metrics(
            {
                "eval/avg_word_diversity": avg_diversity,
                "eval/avg_char_count": avg_length,
            }
        )
        print(f"  Summary: avg diversity={avg_diversity:.2f}, avg length={avg_length:.0f} chars")

    print()
    return results


# ── Part 5: End-to-End Demonstration ──────────────────────────────────── #


def part5_combined_demo(llm: ChatOpenAI) -> None:
    """Run an LLM task with all custom plugins active."""
    print("=" * 60)
    print("Part 5: Combined Plugin Demonstration")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    aggregator = MetricAggregator()
    evaluator = LLMOutputEvaluator()

    prompt = "Explain three benefits of using MLflow for ML experiment tracking."

    with mlflow.start_run(run_name="combined-plugins-demo") as run:
        mlflow.log_param("task", "explanation")
        mlflow.log_param("model", "google/gemma-4-26b-a4b")

        # The EnvironmentContextProvider auto-injects env tags (Part 2)
        print("\n  [Context Provider] Environment tags auto-injected on run creation.")

        # Invoke LLM and measure latency
        print(f"\n  Prompt: {prompt}")
        start = time.perf_counter()
        response = llm.invoke([{"role": "user", "content": prompt}])
        latency = time.perf_counter() - start
        text = response.content
        print(f"  Response: {text[:120]}{'...' if len(text) > 120 else ''}")

        # MetricAggregator (Part 3) -- log latency
        aggregator.log("inference_latency_s", round(latency, 4), step=0)
        print(f"\n  [Metric Aggregator] Logged latency: {latency:.4f}s")

        # Custom Evaluator (Part 4) -- evaluate response quality
        metrics = evaluator.evaluate_response(str(text))
        mlflow.log_metrics(metrics)
        print("  [Evaluator Plugin] Quality metrics logged:")
        for k, v in metrics.items():
            print(f"    {k:<30s} = {v:.2f}")

        # Verify all plugin data is present
        client = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        env_tags = {k: v for k, v in run_data.data.tags.items() if k.startswith("env.")}
        print(f"\n  [Verification] Environment tags present: {len(env_tags)}")
        print(f"  [Verification] Metrics logged: {len(run_data.data.metrics)}")

    print()


# ── Main ───────────────────────────────────────────────────────────────── #


def main() -> None:
    print("=" * 60)
    print("  L3-M3.2 — MLflow Plugins and Extensibility")
    print("=" * 60)
    print()

    # Parts 1-3 do not require an LLM
    part1_plugin_architecture()
    part2_run_context_provider()
    part3_metric_aggregator()

    # Parts 4-5 use the LLM
    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.7,
    )
    part4_model_evaluator(llm)
    part5_combined_demo(llm)

    print("=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5555")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print("  Check runs for auto-injected env.* tags and eval/* metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
