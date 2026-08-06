"""
L3-M2.2 — Tracing Temporal.io Workflows

Integrates MLflow tracing with Temporal.io durable workflows. A three-activity
text-analysis pipeline (summarize, sentiment, keywords) runs inside Temporal
while MLflow captures a unified trace for each workflow execution, recording
per-activity inputs, outputs, and timing.

Architecture note: the workflow class lives in workflow_def.py to avoid
importing MLflow inside Temporal's workflow sandbox (which restricts network
I/O and non-deterministic code). Activities are defined here alongside
the MLflow tracing logic.
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import mlflow
from mlflow.entities import Trace
from openai import OpenAI
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker
from workflow_def import (
    AnalysisRequest,
    AnalysisResult,
    PipelineResult,
    TextAnalysisPipeline,
)

# -- Configuration --
# The LiteLLM gateway from infra/, not a provider directly. The aliases below are
# defined in infra/litellm/config.yaml, which also owns the fallback order and
# each model's context window. Swapping model or provider is a change there,
# never here.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin

TRACKING_URI = "http://127.0.0.1:5555"
EXPERIMENT = "L3/M2_advanced_tracing/2_temporal_tracing"
TASK_QUEUE = "mlflow-tutorial-text-analysis"
MODEL = "gemma-chat"
TEMPORAL_ADDRESS = "localhost:7233"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

llm_client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)


# ── Part 1: Temporal activities with MLflow tracing ───────


TASK_PROMPTS = {
    "summarize": "Summarize the following text in 2-3 concise sentences.",
    "sentiment": (
        "Analyze the sentiment of the following text. "
        "State whether it is positive, negative, or neutral and explain briefly."
    ),
    "keywords": ("Extract 5-7 key topics or keywords from the following text as a comma-separated list."),
}


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the LLM via OpenAI-compatible API and return the response text."""
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


@activity.defn
async def analyze_text(request: AnalysisRequest) -> AnalysisResult:
    """Temporal activity: run a single LLM analysis task with MLflow tracing."""
    start = time.time()
    system_prompt = TASK_PROMPTS.get(request.task, "Analyze the following text.")

    # Wrap the LLM call in an MLflow span to capture inputs/outputs
    with mlflow.start_span(name=f"activity_{request.task}") as span:
        span.set_inputs({"task": request.task, "text_length": len(request.text)})
        span.set_attribute("temporal.activity", request.task)

        result_text = _call_llm(system_prompt, request.text)

        duration = time.time() - start
        span.set_outputs(
            {
                "result_preview": result_text[:200],
                "duration_s": round(duration, 2),
            }
        )

    return AnalysisResult(
        task=request.task,
        result=result_text,
        duration_s=round(duration, 2),
    )


# ── Part 3: Run the workflow with MLflow tracing ──────────


@mlflow.trace(name="temporal_workflow_execution", span_type="WORKFLOW")
async def run_workflow(client: Client, text: str, workflow_id: str) -> PipelineResult:
    """Execute the Temporal workflow and trace the entire run in MLflow."""
    with mlflow.start_span(name="temporal_dispatch") as span:
        span.set_inputs({"workflow_id": workflow_id, "text_length": len(text)})

        result: PipelineResult = await client.execute_workflow(
            TextAnalysisPipeline.run,
            text,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        span.set_outputs(
            {
                "summary_preview": result.summary[:120],
                "sentiment_preview": result.sentiment[:120],
                "keywords_preview": result.keywords[:120],
            }
        )

    return result


# ── Part 4: Analyze traces ────────────────────────────────


def analyze_traces() -> None:
    """Query MLflow traces and compare per-activity timing."""
    print("\n" + "=" * 60)
    print("Part 4: Analyzing MLflow traces")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  Experiment not found — skipping analysis.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(
            experiment_ids=[experiment.experiment_id],
            return_type="list",
            max_results=8,  # limit to recent traces
        ),
    )
    print(f"\n  Showing up to {len(traces)} most recent trace(s)\n")

    for i, trace in enumerate(traces):
        print("-" * 50)
        print(f"  Trace {i + 1}  |  ID: {trace.info.trace_id}")
        print("-" * 50)

        spans = trace.data.spans
        if not spans:
            print("    (no spans)")
            continue

        print("    Spans:")
        for span in spans:
            start_ns = span.start_time_ns or 0
            end_ns = span.end_time_ns or start_ns
            dur_ms = (end_ns - start_ns) / 1e6
            indent = "      " if span.parent_id else "    "
            print(f"{indent}-> {span.name:35s} {dur_ms:>8.1f} ms  [{span.span_type}]")

        # Per-activity breakdown
        activity_spans = [s for s in spans if s.name.startswith("activity_")]
        if activity_spans:
            durations = {s.name: ((s.end_time_ns or 0) - (s.start_time_ns or 0)) / 1e6 for s in activity_spans}
            slowest = max(durations, key=durations.get)  # type: ignore[arg-type]
            fastest = min(durations, key=durations.get)  # type: ignore[arg-type]
            print(f"\n    Slowest activity: {slowest} ({durations[slowest]:.1f} ms)")
            print(f"    Fastest activity: {fastest} ({durations[fastest]:.1f} ms)")

        # Total workflow time from root span
        root_spans = [s for s in spans if s.parent_id is None]
        if root_spans:
            root = root_spans[0]
            total_ms = ((root.end_time_ns or 0) - (root.start_time_ns or 0)) / 1e6
            print(f"    Total workflow time: {total_ms:.1f} ms")
        print()


# ── Main ──────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Artificial intelligence is reshaping industries worldwide. "
    "From healthcare diagnostics to autonomous vehicles, machine learning "
    "models are being deployed at unprecedented scale. However, concerns "
    "about bias, transparency, and accountability remain pressing challenges "
    "that the research community must address.",
    "Temporal.io provides durable execution for distributed applications. "
    "Workflows survive process restarts and infrastructure failures, making "
    "them ideal for long-running business processes. Combined with MLflow "
    "tracing, teams gain full visibility into both execution durability "
    "and AI model performance.",
]


async def async_main() -> None:
    # -- Part 1 & 2: Connect to Temporal --
    print("=" * 60)
    print("Part 1-2: Connecting to Temporal and registering worker")
    print("=" * 60)

    try:
        client = await Client.connect(TEMPORAL_ADDRESS)
        print(f"  Connected to Temporal at {TEMPORAL_ADDRESS}")
    except Exception as exc:
        print(f"\n  Could not connect to Temporal at {TEMPORAL_ADDRESS}: {exc}")
        print("  Make sure Temporal is running (podman compose up -d from infra/).")
        print("\n  Falling back to local-only demonstration...\n")
        await _fallback_demo()
        return

    # -- Part 3: Run workflows inside a worker --
    print("\n" + "=" * 60)
    print("Part 3: Running the Temporal workflow with MLflow tracing")
    print("=" * 60)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TextAnalysisPipeline],
        activities=[analyze_text],
        activity_executor=ThreadPoolExecutor(4),
    ):
        for idx, text in enumerate(SAMPLE_TEXTS, 1):
            # Unique ID avoids "workflow already started" conflicts
            wf_id = f"mlflow-temporal-demo-{idx}-{uuid.uuid4().hex[:8]}"
            print(f"\n  --- Workflow: {wf_id} ---")
            print(f"  Input text: {text[:80]}...")

            result = await run_workflow(client, text, wf_id)

            print(f"  Summary:   {result.summary[:100]}...")
            print(f"  Sentiment: {result.sentiment[:100]}...")
            print(f"  Keywords:  {result.keywords[:100]}...")

    # Let async trace logging flush
    await asyncio.sleep(2)

    # -- Part 4: Analyze traces --
    analyze_traces()

    print("=" * 60)
    print("Done! Inspect results at:")
    print(f"  MLflow UI:    http://127.0.0.1:5555  (experiment: {EXPERIMENT})")
    print("  Temporal UI:  http://localhost:8080   (search for workflow IDs)")
    print("=" * 60)


async def _fallback_demo() -> None:
    """Run the analysis pipeline locally when Temporal is unavailable."""
    print("=" * 60)
    print("Fallback: Running activities directly with MLflow tracing")
    print("=" * 60)

    text = SAMPLE_TEXTS[0]

    with mlflow.start_run(run_name="fallback_local_demo"):
        mlflow.set_tag("mode", "fallback_no_temporal")

        for task in ["summarize", "sentiment", "keywords"]:
            print(f"\n  Running {task}...")
            system_prompt = TASK_PROMPTS.get(task, "Analyze this text.")

            with mlflow.start_span(name=f"activity_{task}") as span:
                span.set_inputs({"task": task, "text_length": len(text)})
                result_text = _call_llm(system_prompt, text)
                span.set_outputs({"result_preview": result_text[:200]})

            print(f"  Result: {result_text[:100]}...")

    await asyncio.sleep(2)
    analyze_traces()

    print("\n" + "=" * 60)
    print("Fallback complete. Start Temporal to see full workflow tracing.")
    print("=" * 60)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
