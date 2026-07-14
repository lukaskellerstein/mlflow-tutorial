"""
L2-5.2 — LangGraph Agent Observability

Build a research assistant agent using LangGraph StateGraph and observe
its execution through MLflow auto-tracing:
  - State transitions and node execution order
  - Conditional edge decisions (quality-check retry loop)
  - Per-node durations and execution timeline
  - Agent-level metrics logged to an MLflow run
"""

import time
from typing import Annotated, Literal

import mlflow
import mlflow.langchain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L2/M5_agent_observability/2_langgraph_agents")
mlflow.langchain.autolog(log_traces=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.7,
)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str
    current_step: str
    quality_pass: bool
    retry_count: int


# ---------------------------------------------------------------------------
# Knowledge base (simple, no external APIs)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: dict[str, str] = {
    "mlflow": (
        "MLflow is an open-source platform for the ML lifecycle. It provides "
        "experiment tracking, model registry, model deployment, and evaluation. "
        "MLflow supports LLM observability through auto-tracing and manual spans."
    ),
    "langgraph": (
        "LangGraph is a library for building stateful, multi-actor applications "
        "with LLMs using graph-based workflows. It supports conditional edges, "
        "cycles, and state persistence. Agents are modeled as nodes in a graph."
    ),
    "tracing": (
        "Distributed tracing captures the full execution path of a request. "
        "MLflow tracing records spans for each operation — LLM calls, tool "
        "invocations, and state transitions — enabling latency analysis and "
        "debugging of complex agent workflows."
    ),
    "agents": (
        "AI agents are systems that use LLMs to reason and take actions. "
        "Common patterns include ReAct (reason + act), plan-and-execute, and "
        "multi-agent collaboration. Observability is critical for debugging "
        "agent loops, tool selection, and quality of reasoning."
    ),
}


def search_knowledge(query: str) -> str:
    """Search the local knowledge base for relevant information."""
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in query_lower or any(w in query_lower for w in topic.split()):
            results.append(content)
    if not results:
        # Return a generic response when no match is found
        results.append(
            "No specific knowledge found. The topic may require broader research. "
            "Available topics: " + ", ".join(KNOWLEDGE_BASE.keys())
        )
    return " ".join(results)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def analyze_query(state: ResearchState) -> dict:
    """Analyze the user query and plan the research approach."""
    user_msg = state["messages"][-1]
    query = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)

    response = llm.invoke([
        HumanMessage(content=(
            f"Analyze this research query and list 2-3 key topics to investigate. "
            f"Be brief (2-3 bullet points).\n\nQuery: {query}"
        ))
    ])
    return {
        "messages": [AIMessage(content=f"[Analysis] {response.content}")],
        "current_step": "analyze_query",
        "research_notes": "",
        "retry_count": 0,
    }


def search_knowledge_node(state: ResearchState) -> dict:
    """Search the knowledge base for information on the query."""
    user_msg = state["messages"][0]
    query = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)
    findings = search_knowledge(query)

    return {
        "messages": [AIMessage(content=f"[Research] Found: {findings[:200]}...")],
        "current_step": "search_knowledge",
        "research_notes": findings,
    }


def synthesize_answer(state: ResearchState) -> dict:
    """Synthesize research notes into a coherent answer."""
    user_msg = state["messages"][0]
    query = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)
    notes = state.get("research_notes", "")

    response = llm.invoke([
        HumanMessage(content=(
            f"Based on these research notes, write a concise answer (2-3 sentences) "
            f"to the question.\n\nQuestion: {query}\n\nNotes: {notes}"
        ))
    ])
    return {
        "messages": [AIMessage(content=f"[Synthesis] {response.content}")],
        "current_step": "synthesize_answer",
    }


def quality_check(state: ResearchState) -> dict:
    """Check if the synthesized answer is adequate."""
    synthesis_msgs = [
        m for m in state["messages"]
        if isinstance(m, AIMessage) and str(m.content).startswith("[Synthesis]")
    ]
    answer = synthesis_msgs[-1].content if synthesis_msgs else ""

    response = llm.invoke([
        HumanMessage(content=(
            f"Rate this answer as PASS or FAIL. Reply with exactly one word: "
            f"PASS or FAIL.\n\nAnswer: {answer}"
        ))
    ])

    passed = "PASS" in response.content.upper()
    retry_count = state.get("retry_count", 0)

    return {
        "messages": [AIMessage(content=f"[QualityCheck] {'PASS' if passed else 'FAIL'}")],
        "current_step": "quality_check",
        "quality_pass": passed,
        "retry_count": retry_count + (0 if passed else 1),
    }


# ---------------------------------------------------------------------------
# Conditional edge: retry or finish
# ---------------------------------------------------------------------------
def should_retry(state: ResearchState) -> Literal["search_knowledge", "__end__"]:
    """Route back to search if quality check failed and retries remain."""
    if state.get("quality_pass", False):
        return "__end__"
    if state.get("retry_count", 0) >= 2:
        return "__end__"  # Max retries reached
    return "search_knowledge"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Construct the research assistant graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search_knowledge", search_knowledge_node)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("quality_check", quality_check)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "search_knowledge")
    graph.add_edge("search_knowledge", "synthesize_answer")
    graph.add_edge("synthesize_answer", "quality_check")
    graph.add_conditional_edges("quality_check", should_retry)

    return graph.compile()


# ---------------------------------------------------------------------------
# Trace analysis helpers
# ---------------------------------------------------------------------------
def analyze_trace(trace) -> dict:
    """Extract execution details from an MLflow trace."""
    spans = trace.data.spans
    node_entries: list[dict] = []

    for span in spans:
        duration_ms = None
        if span.end_time_ns and span.start_time_ns:
            duration_ms = round(
                (span.end_time_ns - span.start_time_ns) / 1_000_000, 1
            )
        node_entries.append({"name": span.name, "duration_ms": duration_ms})

    return {
        "total_spans": len(spans),
        "node_entries": node_entries,
        "node_order": [e["name"] for e in node_entries],
        "total_duration_ms": trace.info.execution_duration,
    }


def print_trace_analysis(label: str, trace, stats: dict) -> None:
    """Print formatted trace analysis."""
    print(f"\n{'─' * 60}")
    print(f"  Trace Analysis: {label}")
    print(f"{'─' * 60}")
    print(f"  Trace ID:       {trace.info.trace_id}")
    print(f"  Status:         {trace.info.state}")
    print(f"  Total duration: {stats['total_duration_ms']} ms")
    print(f"  Total spans:    {stats['total_spans']}")

    print(f"\n  Span execution order:")
    for i, entry in enumerate(stats["node_entries"], 1):
        dur = entry["duration_ms"] if entry["duration_ms"] is not None else "?"
        print(f"    {i}. {entry['name']:30s} ({dur} ms)")

    # Detect retry loops
    node_names = stats["node_order"]
    search_count = sum(1 for n in node_names if "search_knowledge" in n.lower())
    loops = max(0, search_count - 1)
    print(f"\n  Retry loops detected: {loops}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L2-5.2 — LangGraph Agent Observability")
    print("=" * 60)

    app = build_graph()

    queries = [
        "What is MLflow tracing and how does it help with agent observability?",
        "Explain how LangGraph agents work and their common patterns.",
    ]

    all_stats = []

    with mlflow.start_run(run_name="langgraph_agent_observability") as run:
        mlflow.set_tags({
            "agent_type": "langgraph_research_assistant",
            "model": "google/gemma-4-26b-a4b",
            "graph_nodes": "analyze_query,search_knowledge,synthesize_answer,quality_check",
        })

        for idx, query in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"  Query {idx}: {query}")
            print(f"{'=' * 60}")

            start = time.time()
            result = app.invoke({
                "messages": [HumanMessage(content=query)],
                "research_notes": "",
                "current_step": "",
                "quality_pass": False,
                "retry_count": 0,
            })
            elapsed_s = time.time() - start

            # Print the final messages
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"  {msg.content[:120]}")

            # Retrieve and analyze the trace (flush=True waits for async writes)
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                try:
                    trace = mlflow.get_trace(trace_id, flush=True)
                    if trace and trace.data and trace.data.spans:
                        stats = analyze_trace(trace)
                        print_trace_analysis(f"Query {idx}", trace, stats)
                        all_stats.append(stats)
                    else:
                        print(f"\n  Trace {trace_id} logged (view in MLflow UI)")
                except Exception as e:
                    print(f"\n  Trace {trace_id} logged (retrieval note: {e})")
                    print(f"  View full trace in MLflow UI")

            print(f"\n  Wall-clock time: {elapsed_s:.1f}s")

        # ----- Log aggregate metrics to the MLflow run -----
        print(f"\n{'=' * 60}")
        print("  Aggregate Metrics")
        print(f"{'=' * 60}")

        mlflow.log_metric("num_queries", len(queries))

        if all_stats:
            total_nodes = sum(s["total_spans"] for s in all_stats)
            total_loops = sum(
                max(0, sum(
                    1 for n in s["node_order"] if "search_knowledge" in n.lower()
                ) - 1)
                for s in all_stats
            )
            durations = [
                s["total_duration_ms"] for s in all_stats if s["total_duration_ms"]
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0

            mlflow.log_metrics({
                "total_nodes_visited": total_nodes,
                "total_retry_loops": total_loops,
                "avg_trace_duration_ms": round(avg_duration, 1),
            })

            print(f"  Total nodes visited: {total_nodes}")
            print(f"  Total retry loops:   {total_loops}")
            print(f"  Avg trace duration:  {avg_duration:.1f} ms")
        else:
            print("  Trace details not available for programmatic analysis.")
            print("  View traces in the MLflow UI for full span breakdowns.")

        print(f"\n  Run ID: {run.info.run_id}")
        print(f"  View in MLflow UI: http://127.0.0.1:5000/#/experiments")

    print(f"\n{'=' * 60}")
    print("  Done! Check the MLflow UI to explore traces and spans.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
