"""
L3-3.1 — Production Tracing Strategies

Demonstrates production-ready tracing patterns:
  1. Trace sampling strategies (all, percentage, errors-only, slow-requests)
  2. Structured trace metadata (request_id, user_id, environment, version)
  3. Error tracing with full context capture
  4. Performance monitoring (latency percentiles, token usage, summary)
"""

import random
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# ---------------------------------------------------------------------------
# 1. Trace Sampling Strategies
# ---------------------------------------------------------------------------
class TraceSampler:
    """Production trace sampling — controls which requests get traced."""

    @staticmethod
    def sample_all() -> bool:
        """100% sampling — use in dev/test environments."""
        return True

    @staticmethod
    def sample_percentage(rate: float) -> bool:
        """Random sampling at a given rate (0.0-1.0) — use in staging."""
        return random.random() < rate

    @staticmethod
    def sample_errors_only(has_error: bool) -> bool:
        """Only trace requests that result in errors — use in production."""
        return has_error

    @staticmethod
    def sample_slow_requests(duration_ms: float, threshold_ms: float) -> bool:
        """Only trace requests slower than a threshold — use in production."""
        return duration_ms >= threshold_ms


# ---------------------------------------------------------------------------
# 2. Traced LLM Caller
# ---------------------------------------------------------------------------
@dataclass
class TraceRecord:
    """Captured data from a single traced LLM call."""

    request_id: str
    prompt: str
    response: str
    duration_ms: float
    error: str | None = None
    error_type: str | None = None
    stack_trace: str | None = None
    token_estimate: int = 0
    tags: dict[str, str] = field(default_factory=dict)
    sampled: bool = False


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def traced_llm_call(
    llm: ChatOpenAI,
    prompt: str,
    metadata: dict[str, str],
) -> TraceRecord:
    """Execute an LLM call with full production tracing."""
    request_id = metadata.get("request_id", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        with mlflow.start_span(name="llm_call") as span:
            span.set_inputs({"prompt": prompt})
            span.set_attributes(
                {
                    "request_id": request_id,
                    "user_id": metadata.get("user_id", "unknown"),
                    "environment": metadata.get("environment", "dev"),
                    "app_version": metadata.get("app_version", "0.0.0"),
                }
            )
            result = llm.invoke([{"role": "user", "content": prompt}])
            response_text = str(result.content)
            duration_ms = (time.perf_counter() - start) * 1000
            token_est = _estimate_tokens(prompt) + _estimate_tokens(response_text)
            span.set_outputs({"response": response_text[:200]})
            span.set_attributes(
                {
                    "duration_ms": round(duration_ms, 1),
                    "token_estimate": token_est,
                }
            )
            # Tag the trace for filtering
            mlflow.set_trace_tag(span.trace_id, "request_id", request_id)
            mlflow.set_trace_tag(span.trace_id, "environment", metadata.get("environment", "dev"))
            for k, v in metadata.items():
                mlflow.set_trace_tag(span.trace_id, k, v)
            return TraceRecord(
                request_id=request_id,
                prompt=prompt,
                response=response_text,
                duration_ms=round(duration_ms, 1),
                token_estimate=token_est,
                tags=metadata,
                sampled=True,
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return TraceRecord(
            request_id=request_id,
            prompt=prompt,
            response="",
            duration_ms=round(duration_ms, 1),
            error=str(exc),
            error_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
            tags=metadata,
            sampled=True,
        )


def untraced_llm_call(llm: ChatOpenAI, prompt: str) -> TraceRecord:
    """Execute an LLM call without tracing (when sampler says skip)."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        result = llm.invoke([{"role": "user", "content": prompt}])
        duration_ms = (time.perf_counter() - start) * 1000
        return TraceRecord(
            request_id=request_id,
            prompt=prompt,
            response=str(result.content),
            duration_ms=round(duration_ms, 1),
            token_estimate=_estimate_tokens(prompt) + _estimate_tokens(str(result.content)),
            sampled=False,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return TraceRecord(
            request_id=request_id,
            prompt=prompt,
            response="",
            duration_ms=round(duration_ms, 1),
            error=str(exc),
            error_type=type(exc).__name__,
            sampled=False,
        )


# ---------------------------------------------------------------------------
# 3. Performance Analytics
# ---------------------------------------------------------------------------
def compute_latency_percentiles(records: list[TraceRecord]) -> dict[str, float]:
    """Compute p50, p95, p99 latency from trace records."""
    latencies = [r.duration_ms for r in records]
    if not latencies:
        return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "p99_ms": round(float(np.percentile(latencies, 99)), 1),
    }


def compute_error_rate(records: list[TraceRecord]) -> float:
    """Compute error rate across all records."""
    if not records:
        return 0.0
    errors = sum(1 for r in records if r.error is not None)
    return round(errors / len(records), 4)


def build_performance_summary(records: list[TraceRecord]) -> dict[str, Any]:
    """Build a complete performance summary from trace records."""
    latency_pcts = compute_latency_percentiles(records)
    total_tokens = sum(r.token_estimate for r in records)
    sampled_count = sum(1 for r in records if r.sampled)
    return {
        "total_requests": len(records),
        "sampled_requests": sampled_count,
        "sampling_rate": round(sampled_count / max(len(records), 1), 2),
        "error_rate": compute_error_rate(records),
        "total_token_estimate": total_tokens,
        "avg_tokens_per_request": round(total_tokens / max(len(records), 1)),
        **latency_pcts,
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain recursion in one sentence.",
    "Name three prime numbers.",
    "What color is the sky?",
    "Define 'machine learning' briefly.",
]


def run_part1_sampling(llm: ChatOpenAI) -> list[TraceRecord]:
    """Part 1: Demonstrate each sampling strategy."""
    print("=" * 60)
    print("Part 1: Trace Sampling Strategies")
    print("=" * 60)
    sampler = TraceSampler()
    all_records: list[TraceRecord] = []

    strategies = [
        ("sample_all (dev)", lambda r: sampler.sample_all()),
        ("sample_percentage 50% (staging)", lambda r: sampler.sample_percentage(0.5)),
        ("sample_errors_only (prod)", lambda r: sampler.sample_errors_only(r.error is not None)),
        (
            "sample_slow_requests >500ms (prod)",
            lambda r: sampler.sample_slow_requests(r.duration_ms, 500),
        ),
    ]

    prompt = PROMPTS[0]
    for name, should_sample_fn in strategies:
        print(f"\n  Strategy: {name}")
        metadata = {"environment": name.split("(")[-1].strip(")"), "request_id": str(uuid.uuid4())}
        # For sample_all and sample_percentage, we check BEFORE calling
        if "sample_all" in name or "sample_percentage" in name:
            if should_sample_fn(None):
                rec = traced_llm_call(llm, prompt, metadata)
                print(f"    Traced: yes | Duration: {rec.duration_ms}ms")
            else:
                rec = untraced_llm_call(llm, prompt)
                print(f"    Traced: no  | Duration: {rec.duration_ms}ms (skipped by sampler)")
        else:
            # For errors_only and slow_requests, we call first then decide retroactively
            rec = untraced_llm_call(llm, prompt)
            # Check if we should have traced this after the fact
            would_trace = should_sample_fn(rec)
            print(f"    Would trace: {would_trace} | Duration: {rec.duration_ms}ms")
        all_records.append(rec)

    print(f"\n  Total calls: {len(all_records)}")
    print(f"  Traced: {sum(1 for r in all_records if r.sampled)}")
    return all_records


def run_part2_metadata(llm: ChatOpenAI) -> list[TraceRecord]:
    """Part 2: Structured trace metadata for production filtering."""
    print("\n" + "=" * 60)
    print("Part 2: Structured Trace Metadata")
    print("=" * 60)

    users = ["user_alice", "user_bob", "user_charlie"]
    environments = ["production", "staging"]
    records: list[TraceRecord] = []

    for i, prompt in enumerate(PROMPTS[:3]):
        metadata = {
            "request_id": str(uuid.uuid4()),
            "user_id": users[i % len(users)],
            "environment": environments[i % len(environments)],
            "app_version": "2.1.0",
            "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        }
        print(f"\n  Request {i + 1}: user={metadata['user_id']}, env={metadata['environment']}")
        rec = traced_llm_call(llm, prompt, metadata)
        records.append(rec)
        print(f"    Duration: {rec.duration_ms}ms | Tokens: ~{rec.token_estimate}")
        print(f"    Tags set: {list(metadata.keys())}")

    print(f"\n  All {len(records)} traces tagged with production metadata.")
    print("  Filter in MLflow UI by tags: request_id, user_id, environment, etc.")
    return records


def run_part3_error_tracing(llm: ChatOpenAI) -> list[TraceRecord]:
    """Part 3: Error tracing with full context."""
    print("\n" + "=" * 60)
    print("Part 3: Error Tracing")
    print("=" * 60)

    records: list[TraceRecord] = []
    metadata_base = {"environment": "production", "app_version": "2.1.0"}

    # Successful call
    print("\n  3a. Normal call (should succeed):")
    meta = {**metadata_base, "request_id": str(uuid.uuid4()), "user_id": "user_test"}
    rec = traced_llm_call(llm, "What is 2+2?", meta)
    records.append(rec)
    print(f"    Status: {'ERROR' if rec.error else 'OK'} | Duration: {rec.duration_ms}ms")

    # Simulated error: bad model name
    print("\n  3b. Simulated error (invalid model):")
    bad_llm = ChatOpenAI(
        model="nonexistent_model_xyz",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.0,
    )
    meta = {**metadata_base, "request_id": str(uuid.uuid4()), "user_id": "user_test"}
    rec = traced_llm_call(bad_llm, "This should fail", meta)
    records.append(rec)
    print(f"    Status: ERROR | Type: {rec.error_type}")
    print(f"    Error: {rec.error[:80]}..." if rec.error and len(rec.error) > 80 else f"    Error: {rec.error}")

    # Error aggregation
    error_records = [r for r in records if r.error]
    ok_records = [r for r in records if not r.error]
    print(f"\n  Error summary: {len(error_records)} errors / {len(records)} total")
    print(f"  Success rate: {len(ok_records) / max(len(records), 1):.0%}")
    if error_records:
        print("  Error types:")
        error_types: dict[str, int] = {}
        for r in error_records:
            t = r.error_type or "Unknown"
            error_types[t] = error_types.get(t, 0) + 1
        for etype, count in error_types.items():
            print(f"    - {etype}: {count}")

    return records


def run_part4_performance(all_records: list[TraceRecord]) -> None:
    """Part 4: Performance monitoring from collected trace records."""
    print("\n" + "=" * 60)
    print("Part 4: Performance Monitoring")
    print("=" * 60)

    # Filter to only successful records for latency analysis
    ok_records = [r for r in all_records if r.error is None]
    summary = build_performance_summary(ok_records)

    print(f"\n  Requests analyzed: {summary['total_requests']}")
    print(f"  Sampled (traced): {summary['sampled_requests']} ({summary['sampling_rate']:.0%})")
    print(f"  Error rate: {summary['error_rate']:.1%}")
    print("\n  Latency percentiles:")
    print(f"    p50: {summary['p50_ms']:.1f} ms")
    print(f"    p95: {summary['p95_ms']:.1f} ms")
    print(f"    p99: {summary['p99_ms']:.1f} ms")
    print("\n  Token usage:")
    print(f"    Total tokens (est): {summary['total_token_estimate']}")
    print(f"    Avg tokens/request: {summary['avg_tokens_per_request']}")

    # Log performance summary to MLflow
    with mlflow.start_run(run_name="performance_summary"):
        mlflow.log_params(
            {
                "total_requests": summary["total_requests"],
                "sampled_requests": summary["sampled_requests"],
            }
        )
        mlflow.log_metrics(
            {
                "sampling_rate": summary["sampling_rate"],
                "error_rate": summary["error_rate"],
                "p50_latency_ms": summary["p50_ms"],
                "p95_latency_ms": summary["p95_ms"],
                "p99_latency_ms": summary["p99_ms"],
                "total_token_estimate": summary["total_token_estimate"],
                "avg_tokens_per_request": summary["avg_tokens_per_request"],
            }
        )
        mlflow.set_tags(
            {
                "run_type": "performance_summary",
                "environment": "tutorial",
            }
        )

        # Save records as CSV artifact
        rows = [
            {
                "request_id": r.request_id,
                "duration_ms": r.duration_ms,
                "token_estimate": r.token_estimate,
                "error": r.error or "",
                "sampled": r.sampled,
            }
            for r in all_records
        ]
        df = pd.DataFrame(rows)
        csv_path = "/tmp/production_trace_records.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="performance")
        print("\n  Performance summary logged to MLflow run.")

    # Strategy comparison table
    print("\n" + "=" * 60)
    print("Production Tracing Strategy Comparison")
    print("=" * 60)
    print(f"  {'Strategy':<30} {'Use Case':<15} {'Overhead':<10} {'Coverage'}")
    print(f"  {'-' * 30} {'-' * 15} {'-' * 10} {'-' * 15}")
    strategies = [
        ("sample_all (100%)", "Dev / Test", "High", "Complete"),
        ("sample_percentage (1-10%)", "Staging", "Low", "Statistical"),
        ("sample_errors_only", "Production", "Minimal", "Errors only"),
        ("sample_slow_requests", "Production", "Minimal", "Tail latency"),
    ]
    for name, use_case, overhead, coverage in strategies:
        print(f"  {name:<30} {use_case:<15} {overhead:<10} {coverage}")
    print()


def main() -> None:
    print("=" * 60)
    print("L3-3.1 — Production Tracing Strategies")
    print("=" * 60)

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.0,
    )

    records_p1 = run_part1_sampling(llm)
    records_p2 = run_part2_metadata(llm)
    records_p3 = run_part3_error_tracing(llm)

    all_records = records_p1 + records_p2 + records_p3
    run_part4_performance(all_records)

    print("=" * 60)
    print("Done! Check MLflow UI at http://127.0.0.1:5555")
    print("Filter traces by tags: environment, user_id, request_id")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L3/M1_production/1_production_tracing")
    main()
