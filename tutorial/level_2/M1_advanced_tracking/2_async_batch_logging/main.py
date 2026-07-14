"""
L2-1.2 — Async and Batch Logging

Demonstrates MLflow's async and batch logging for LLM evaluation:
- Async logging: non-blocking metric logging during batch LLM calls
- Step-based metrics: logging quality scores and latencies per prompt
- Batch logging: log_metrics() and log_params() for bulk operations
- Sync vs async timing comparison
"""

import time

import mlflow
from openai import OpenAI

PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms.",
    "What is the difference between AI and ML?",
    "How does gradient descent work?",
    "What is overfitting and how do you prevent it?",
    "Explain the bias-variance tradeoff.",
    "What are transformers in deep learning?",
    "How does backpropagation work?",
    "What is transfer learning?",
    "Explain reinforcement learning briefly.",
    "What is a convolutional neural network?",
    "How do attention mechanisms work?",
]


def call_llm(client: OpenAI, prompt: str) -> tuple[str, float, int]:
    """Call LLM and return response text, latency in ms, and token count."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.choices[0].message.content or ""
    token_count = response.usage.completion_tokens if response.usage else len(text.split())
    return text, latency_ms, token_count


def part1_async_step_logging(client: OpenAI) -> None:
    """Process batch of prompts with async step-based logging."""
    print("=" * 60)
    print("Part 1: Async Logging with Step-Based Metrics")
    print("=" * 60)

    mlflow.config.enable_async_logging(True)
    print("  Async logging ENABLED")
    print(f"  Processing {len(PROMPTS)} prompts through LLM...\n")

    with mlflow.start_run(run_name="async_batch_eval") as run:
        mlflow.log_param("model", "google/gemma-4-e4b")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_tokens", 150)
        mlflow.log_param("num_prompts", len(PROMPTS))

        for i, prompt in enumerate(PROMPTS):
            text, latency_ms, token_count = call_llm(client, prompt)
            response_length = len(text)

            # Step-based logging -- these calls return immediately (async)
            mlflow.log_metric("response_length", response_length, step=i)
            mlflow.log_metric("latency_ms", round(latency_ms, 2), step=i)
            mlflow.log_metric("token_count", token_count, step=i)

            print(f"    [{i:2d}] {prompt[:45]:<45s}  "
                  f"latency={latency_ms:7.1f}ms  tokens={token_count:3d}  "
                  f"len={response_length:4d}")

    mlflow.config.enable_async_logging(False)

    print(f"\n  Run ID: {run.info.run_id}")
    print("  Step-based metrics logged asynchronously for each prompt.")
    print("  View per-prompt charts at: http://127.0.0.1:5000\n")


def part2_batch_logging(client: OpenAI) -> None:
    """Demonstrate batch logging with log_metrics() and log_params()."""
    print("=" * 60)
    print("Part 2: Batch Logging with log_metrics() and log_params()")
    print("=" * 60)

    # Run a small subset to gather aggregate stats
    latencies = []
    token_counts = []
    response_lengths = []

    print("  Gathering aggregate stats from LLM responses...\n")
    for prompt in PROMPTS[:6]:
        text, latency_ms, token_count = call_llm(client, prompt)
        latencies.append(latency_ms)
        token_counts.append(token_count)
        response_lengths.append(len(text))

    with mlflow.start_run(run_name="batch_logging_demo") as run:
        # Log all LLM config params at once
        params = {
            "model": "google/gemma-4-e4b",
            "temperature": "0.7",
            "max_tokens": "150",
            "provider": "lm-studio",
            "base_url": "http://localhost:1234/v1",
            "num_prompts_evaluated": str(len(PROMPTS[:6])),
        }
        mlflow.log_params(params)
        print(f"  Logged {len(params)} params in a single log_params() call")

        # Log aggregate metrics at once
        metrics = {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_token_count": round(sum(token_counts) / len(token_counts), 2),
            "total_tokens": sum(token_counts),
            "avg_response_length": round(sum(response_lengths) / len(response_lengths), 2),
            "min_response_length": min(response_lengths),
            "max_response_length": max(response_lengths),
        }
        mlflow.log_metrics(metrics)
        print(f"  Logged {len(metrics)} metrics in a single log_metrics() call")

        # Print the aggregate results
        print(f"\n  Aggregate Results:")
        print(f"    Avg latency:         {metrics['avg_latency_ms']:.2f} ms")
        print(f"    Avg token count:     {metrics['avg_token_count']:.0f}")
        print(f"    Avg response length: {metrics['avg_response_length']:.0f} chars")

    print(f"\n  Run ID: {run.info.run_id}")
    print("  Batch logging avoids multiple round-trips to the server.\n")


def part3_sync_vs_async_timing(client: OpenAI) -> None:
    """Compare sync vs async logging performance."""
    print("=" * 60)
    print("Part 3: Sync vs Async Timing Comparison")
    print("=" * 60)

    # Pre-generate results so LLM latency does not affect the comparison
    print("\n  Pre-generating LLM responses for fair comparison...")
    results = []
    for prompt in PROMPTS[:6]:
        text, latency_ms, token_count = call_llm(client, prompt)
        results.append((len(text), latency_ms, token_count))
    print(f"  Collected {len(results)} responses. Now comparing logging speed.\n")

    # --- Synchronous logging ---
    mlflow.config.enable_async_logging(False)
    print("  Synchronous logging...")

    with mlflow.start_run(run_name="sync_timing_test"):
        t_start = time.perf_counter()
        for i, (resp_len, lat, tok) in enumerate(results):
            mlflow.log_metric("response_length", resp_len, step=i)
            mlflow.log_metric("latency_ms", round(lat, 2), step=i)
            mlflow.log_metric("token_count", tok, step=i)
        sync_elapsed = time.perf_counter() - t_start

    print(f"    Time: {sync_elapsed:.4f}s")

    # --- Asynchronous logging ---
    mlflow.config.enable_async_logging(True)
    print("\n  Asynchronous logging...")

    with mlflow.start_run(run_name="async_timing_test"):
        t_start = time.perf_counter()
        for i, (resp_len, lat, tok) in enumerate(results):
            mlflow.log_metric("response_length", resp_len, step=i)
            mlflow.log_metric("latency_ms", round(lat, 2), step=i)
            mlflow.log_metric("token_count", tok, step=i)
        async_elapsed = time.perf_counter() - t_start

    mlflow.config.enable_async_logging(False)

    print(f"    Time: {async_elapsed:.4f}s")

    # --- Results ---
    print(f"\n  Results:")
    print(f"    Sync:  {sync_elapsed:.4f}s")
    print(f"    Async: {async_elapsed:.4f}s")
    if sync_elapsed > 0:
        speedup = sync_elapsed / max(async_elapsed, 0.0001)
        print(f"    Speedup: {speedup:.1f}x")
    print()
    print("  Async logging returns immediately, offloading I/O to a")
    print("  background thread. The speedup is most noticeable when the")
    print("  tracking server has higher latency (remote servers, network).\n")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M1_advanced_tracking/2_async_batch_logging")

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    part1_async_step_logging(client)
    part2_batch_logging(client)
    part3_sync_vs_async_timing(client)

    print("=" * 60)
    print("Done! View all runs in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)
