"""
L2-5.3 — Multi-Agent Systems

Demonstrates a multi-agent system with a supervisor/coordinator pattern
using LangGraph, with full MLflow tracing:
- Three specialized agents: Researcher, Writer, Reviewer
- A supervisor node that routes work between agents
- Workflow: researcher -> writer -> reviewer -> (revision loop if needed)
- Inter-agent handoff tracing and per-agent metrics
- Overall pipeline metrics logged to MLflow
"""

import operator
import time
from typing import Annotated, Literal, cast

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import Trace
from pydantic import SecretStr
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State shared across all nodes
# ---------------------------------------------------------------------------
class PipelineState(TypedDict):
    topic: str
    research_notes: str
    draft: str
    review_feedback: str
    review_passed: bool
    revision_count: int
    agent_durations: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------
def researcher_node(state: PipelineState) -> dict:
    """Researcher agent: generates key facts and points about a topic."""
    print("\n  [Researcher] Researching topic...")
    start = time.time()

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.7,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Given a topic, produce 3-5 concise "
                "bullet points covering the most important facts. Be factual and brief.",
            ),
            ("human", "Research this topic: {topic}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    notes = chain.invoke({"topic": state["topic"]})

    duration = time.time() - start
    print(f"  [Researcher] Done in {duration:.1f}s")

    return {
        "research_notes": notes,
        "agent_durations": [{"agent": "researcher", "duration_s": round(duration, 2)}],
        "messages": [f"Researcher produced notes on: {state['topic']}"],
    }


def writer_node(state: PipelineState) -> dict:
    """Writer agent: takes research notes and writes a structured summary."""
    revision = state.get("revision_count", 0)
    label = "Writing" if revision == 0 else f"Revising (attempt {revision + 1})"
    print(f"\n  [Writer] {label}...")
    start = time.time()

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.7,
    )

    if revision == 0:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a technical writer. Given research notes, write a clear, "
                    "structured summary in 2-3 short paragraphs. Use plain language.",
                ),
                ("human", "Write a summary based on these notes:\n{notes}"),
            ]
        )
        draft = (prompt | llm | StrOutputParser()).invoke({"notes": state["research_notes"]})
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a technical writer. Revise the draft based on the reviewer's "
                    "feedback. Keep the summary to 2-3 short paragraphs.",
                ),
                (
                    "human",
                    "Original draft:\n{draft}\n\nReviewer feedback:\n{feedback}\n\n"
                    "Please revise the draft.",
                ),
            ]
        )
        draft = (prompt | llm | StrOutputParser()).invoke(
            {
                "draft": state["draft"],
                "feedback": state["review_feedback"],
            }
        )

    duration = time.time() - start
    print(f"  [Writer] Done in {duration:.1f}s")

    return {
        "draft": draft,
        "agent_durations": [{"agent": "writer", "duration_s": round(duration, 2)}],
        "messages": [f"Writer produced draft (revision {revision})"],
    }


def reviewer_node(state: PipelineState) -> dict:
    """Reviewer agent: reviews the summary and provides pass/fail + feedback."""
    print("\n  [Reviewer] Reviewing draft...")
    start = time.time()

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an editor. Review the draft for clarity and accuracy. "
                "Respond with exactly one of these formats:\n"
                "PASS: <one sentence of praise>\n"
                "FAIL: <one sentence describing what to fix>\n"
                "Be concise. Only fail if there is a clear problem.",
            ),
            ("human", "Review this draft:\n{draft}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"draft": state["draft"]})

    duration = time.time() - start
    passed = review.strip().upper().startswith("PASS")
    print(f"  [Reviewer] Verdict: {'PASS' if passed else 'FAIL'} ({duration:.1f}s)")

    return {
        "review_feedback": review,
        "review_passed": passed,
        "revision_count": state.get("revision_count", 0) + 1,
        "agent_durations": [{"agent": "reviewer", "duration_s": round(duration, 2)}],
        "messages": [f"Reviewer verdict: {'PASS' if passed else 'FAIL'}"],
    }


def supervisor_router(state: PipelineState) -> Literal["writer", "__end__"]:
    """Route after review: revise if failed (max 1 revision), else finish."""
    if state.get("review_passed", False):
        return "__end__"
    if state.get("revision_count", 0) >= 2:
        print("  [Supervisor] Max revisions reached, finishing.")
        return "__end__"
    print("  [Supervisor] Sending back to Writer for revision.")
    return "writer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]:
    """Build the multi-agent LangGraph with supervisor routing."""
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Define edges: linear pipeline with a review loop
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_conditional_edges("reviewer", supervisor_router)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Run pipeline and analyze traces
# ---------------------------------------------------------------------------
def run_pipeline(topic: str, graph) -> dict:
    """Run the multi-agent pipeline for a single topic."""
    print(f"\n{'=' * 60}")
    print(f"Topic: {topic}")
    print("=" * 60)

    initial_state = {
        "topic": topic,
        "research_notes": "",
        "draft": "",
        "review_feedback": "",
        "review_passed": False,
        "revision_count": 0,
        "agent_durations": [],
        "messages": [],
    }

    start = time.time()
    result = graph.invoke(initial_state)
    total_duration = time.time() - start

    print(f"\n  Pipeline completed in {total_duration:.1f}s")
    print(f"  Revisions: {result['revision_count']}")
    print(f"  Review passed: {result['review_passed']}")

    return {**result, "total_duration_s": round(total_duration, 2)}


def analyze_traces() -> None:
    """Search and analyze traces from the multi-agent pipeline."""
    print("\n" + "=" * 60)
    print("Analyzing Traces")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name("L2/M5_agent_observability/3_multiagent_systems")
    if experiment is None:
        print("  No experiment found.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(
            locations=[experiment.experiment_id],
            max_results=5,
            return_type="list",
            flush=True,
        ),
    )

    print(f"  Found {len(traces)} trace(s)\n")

    for i, trace in enumerate(traces):
        info = trace.info
        duration_ms = info.execution_duration
        print(f"  --- Trace {i + 1} ---")
        print(f"  Trace ID:  {info.trace_id}")
        print(f"  State:     {info.state}")
        if duration_ms:
            print(f"  Duration:  {duration_ms} ms ({duration_ms / 1000:.1f}s)")

        spans = trace.data.spans
        print(f"  Spans ({len(spans)}):")

        # Show span hierarchy
        for span in spans:
            is_root = span.parent_id is None
            prefix = "  (root)" if is_root else "        "
            span_dur = ""
            if span.end_time_ns and span.start_time_ns:
                span_dur_ms = (span.end_time_ns - span.start_time_ns) / 1e6
                span_dur = f"  {span_dur_ms:.0f}ms"
            print(f"    {prefix} {span.name}  [{span.span_type}]{span_dur}")

        print()


def log_pipeline_metrics(results: list[dict]) -> None:
    """Log aggregated multi-agent metrics to an MLflow run."""
    print("\n" + "=" * 60)
    print("Logging Multi-Agent Metrics")
    print("=" * 60)

    with mlflow.start_run(run_name="multiagent_pipeline_metrics"):
        mlflow.set_tags(
            {
                "pipeline_type": "researcher_writer_reviewer",
                "agent_count": "3",
                "pattern": "supervisor",
            }
        )

        for idx, result in enumerate(results):
            topic_key = f"topic_{idx + 1}"

            # Overall metrics
            mlflow.log_metric(f"{topic_key}_total_duration_s", result["total_duration_s"])
            mlflow.log_metric(f"{topic_key}_revision_count", result["revision_count"])
            mlflow.log_metric(f"{topic_key}_review_passed", int(result["review_passed"]))

            # Per-agent durations
            for entry in result["agent_durations"]:
                agent = entry["agent"]
                dur = entry["duration_s"]
                # Use step to differentiate multiple calls to same agent
                mlflow.log_metric(f"{topic_key}_{agent}_duration_s", dur)

            # Handoff count = number of agent transitions
            handoff_count = len(result["agent_durations"]) - 1
            mlflow.log_metric(f"{topic_key}_handoff_count", max(handoff_count, 0))

            print(
                f"  [{topic_key}] duration={result['total_duration_s']}s, "
                f"revisions={result['revision_count']}, "
                f"handoffs={handoff_count}, "
                f"passed={result['review_passed']}"
            )

        # Averages across topics
        avg_duration = sum(r["total_duration_s"] for r in results) / len(results)
        avg_revisions = sum(r["revision_count"] for r in results) / len(results)
        pass_rate = sum(int(r["review_passed"]) for r in results) / len(results)
        mlflow.log_metrics(
            {
                "avg_total_duration_s": round(avg_duration, 2),
                "avg_revision_count": round(avg_revisions, 2),
                "review_pass_rate": round(pass_rate, 2),
            }
        )

        print(
            f"\n  Averages: duration={avg_duration:.1f}s, "
            f"revisions={avg_revisions:.1f}, pass_rate={pass_rate:.0%}"
        )

    print("  Metrics logged to MLflow run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M1_agent_frameworks/3_multiagent_systems")

    # Enable auto-tracing for LangChain/LangGraph
    print("=" * 60)
    print("Enabling LangChain auto-tracing: mlflow.langchain.autolog()")
    print("=" * 60)
    mlflow.langchain.autolog()

    # Build the multi-agent graph
    graph = build_graph()

    # Run the pipeline on two topics
    topics = [
        "The impact of large language models on software development",
        "Renewable energy storage technologies in 2025",
    ]

    results = []
    for topic in topics:
        result = run_pipeline(topic, graph)
        results.append(result)

    # Analyze traces
    analyze_traces()

    # Log aggregated metrics
    log_pipeline_metrics(results)

    print("\n" + "=" * 60)
    print("Done! View your multi-agent traces in the MLflow UI:")
    print("  http://127.0.0.1:5555 -> Traces tab")
    print("  Look for the researcher -> writer -> reviewer span hierarchy.")
    print("=" * 60)
