"""L2-M1.2 — LangGraph Agents

Combines LangGraph tracing with agent observability:
- Part 1: Simple workflow with conditional routing (classify -> process)
- Part 2: Research agent with retry loop and quality checks
- Auto-tracing with mlflow.langchain.autolog()
- Programmatic trace analysis (span durations, node paths, retries)
- Aggregate agent metrics logged to MLflow
"""

import time
from typing import Annotated, Any, Literal

import mlflow
import mlflow.langchain
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from mlflow.entities import Trace
from pydantic import SecretStr
from typing_extensions import TypedDict

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L2/M1_agent_frameworks/2_langgraph_agents")
mlflow.langchain.autolog(log_traces=True)

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("lm-studio"),
    temperature=0.0,
)


# ── Part 1: Simple Workflow with Conditional Routing ──────────────


class SimpleState(TypedDict):
    input_text: str
    complexity: str
    final_response: str


SimpleGraph = CompiledStateGraph[SimpleState, None, SimpleState, SimpleState]


def classify_input(state: SimpleState) -> dict[str, Any]:
    """Classify input as simple or complex."""
    prompt = (
        "Classify the following request as SIMPLE or COMPLEX.\n"
        "SIMPLE = short factual question or greeting.\n"
        "COMPLEX = needs analysis, explanation, or creativity.\n"
        "Respond with exactly one word: SIMPLE or COMPLEX.\n\n"
        f"Request: {state['input_text']}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    complexity = "simple" if "SIMPLE" in str(response.content).upper() else "complex"
    return {"complexity": complexity}


def process_simple(state: SimpleState) -> dict[str, Any]:
    """Handle simple inputs with a brief response."""
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": (f"Give a brief, direct answer in 1-2 sentences.\n\nQuestion: {state['input_text']}"),
            }
        ]
    )
    return {"final_response": str(response.content)}


def process_complex(state: SimpleState) -> dict[str, Any]:
    """Handle complex inputs with a detailed response."""
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"Provide a thorough answer. Use a numbered list if appropriate.\n\nRequest: {state['input_text']}"
                ),
            }
        ]
    )
    return {"final_response": str(response.content)}


def route_by_complexity(state: SimpleState) -> Literal["process_simple", "process_complex"]:
    return "process_simple" if state["complexity"] == "simple" else "process_complex"


def build_simple_graph() -> SimpleGraph:
    """Build the simple conditional routing graph."""
    builder = StateGraph(SimpleState)
    builder.add_node("classify_input", classify_input)
    builder.add_node("process_simple", process_simple)
    builder.add_node("process_complex", process_complex)
    builder.add_edge(START, "classify_input")
    builder.add_conditional_edges("classify_input", route_by_complexity)
    builder.add_edge("process_simple", END)
    builder.add_edge("process_complex", END)
    return builder.compile()


# ── Part 2: Research Agent with Retry Loop ────────────────────────


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    research_notes: str
    current_step: str
    quality_pass: bool
    retry_count: int


ResearchGraph = CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]


KNOWLEDGE_BASE: dict[str, str] = {
    "mlflow": "MLflow is an open-source platform for the ML lifecycle with tracking, registry, and evaluation.",
    "langgraph": "LangGraph builds stateful agent workflows using graphs with conditional edges and cycles.",
    "tracing": "Distributed tracing captures execution paths -- LLM calls, tool invocations, and state transitions.",
    "agents": "AI agents use LLMs to reason and act. Patterns: ReAct, plan-and-execute, multi-agent collaboration.",
}


def message_text(message: AnyMessage) -> str:
    """Read a message's content as plain text."""
    return str(message.content)


def search_knowledge(query: str) -> str:
    """Search the local knowledge base."""
    results = [v for k, v in KNOWLEDGE_BASE.items() if k in query.lower()]
    return " ".join(results) if results else "No match. Topics: " + ", ".join(KNOWLEDGE_BASE)


def analyze_query(state: ResearchState) -> dict[str, Any]:
    last_content = message_text(state["messages"][-1])
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"Analyze this query and list 2-3 key topics to investigate. Be brief.\n\nQuery: {last_content}"
                ),
            }
        ]
    )
    return {
        "messages": [{"role": "assistant", "content": f"[Analysis] {response.content}"}],
        "current_step": "analyze_query",
        "research_notes": "",
        "retry_count": 0,
    }


def search_knowledge_node(state: ResearchState) -> dict[str, Any]:
    query = message_text(state["messages"][0])
    findings = search_knowledge(query)
    return {
        "messages": [{"role": "assistant", "content": f"[Research] {findings[:200]}"}],
        "current_step": "search_knowledge",
        "research_notes": findings,
    }


def synthesize_answer(state: ResearchState) -> dict[str, Any]:
    query = message_text(state["messages"][0])
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"Write a concise answer (2-3 sentences) to the question.\n\n"
                    f"Question: {query}\n\nNotes: {state.get('research_notes', '')}"
                ),
            }
        ]
    )
    return {
        "messages": [{"role": "assistant", "content": f"[Synthesis] {response.content}"}],
        "current_step": "synthesize_answer",
    }


def quality_check(state: ResearchState) -> dict[str, Any]:
    synthesis = [m for m in state["messages"] if message_text(m).startswith("[Synthesis]")]
    answer = message_text(synthesis[-1]) if synthesis else ""
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": f"Rate this answer as PASS or FAIL. Reply with one word.\n\nAnswer: {answer}",
            }
        ]
    )
    passed = "PASS" in str(response.content).upper()
    return {
        "messages": [{"role": "assistant", "content": f"[QualityCheck] {'PASS' if passed else 'FAIL'}"}],
        "current_step": "quality_check",
        "quality_pass": passed,
        "retry_count": state.get("retry_count", 0) + (0 if passed else 1),
    }


def should_retry(state: ResearchState) -> Literal["search_knowledge", "__end__"]:
    if state.get("quality_pass", False):
        return "__end__"
    if state.get("retry_count", 0) >= 2:
        return "__end__"
    return "search_knowledge"


def build_research_graph() -> ResearchGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search_knowledge", search_knowledge_node)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("quality_check", quality_check)
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search_knowledge")
    graph.add_edge("search_knowledge", "synthesize_answer")
    graph.add_edge("synthesize_answer", "quality_check")
    graph.add_conditional_edges("quality_check", should_retry)
    return graph.compile()


# ── Trace Analysis ────────────────────────────────────────────────


def analyze_trace(trace: Trace) -> dict[str, Any]:
    spans = trace.data.spans
    entries: list[dict[str, Any]] = []
    for span in spans:
        dur = (
            round((span.end_time_ns - span.start_time_ns) / 1e6, 1) if span.end_time_ns and span.start_time_ns else None
        )
        entries.append({"name": span.name, "duration_ms": dur})
    return {
        "total_spans": len(spans),
        "entries": entries,
        "names": [e["name"] for e in entries],
        "total_duration_ms": trace.info.execution_duration,
    }


def print_trace_analysis(label: str, trace: Trace, stats: dict[str, Any]) -> None:
    print(f"\n  {'=' * 50}")
    print(f"  Trace: {label}  |  ID: {trace.info.trace_id}")
    print(f"  Duration: {stats['total_duration_ms']} ms  |  Spans: {stats['total_spans']}")
    for i, e in enumerate(stats["entries"], 1):
        dur = e["duration_ms"] if e["duration_ms"] is not None else "?"
        print(f"    {i}. {e['name']:30s} ({dur} ms)")
    retries = max(0, sum(1 for n in stats["names"] if "search_knowledge" in n.lower()) - 1)
    if retries > 0:
        print(f"  Retry loops: {retries}")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    # ── Part 1: Simple Workflow ───────────────────────────────────
    print("=" * 60)
    print("Part 1: Simple Workflow with Conditional Routing")
    print("=" * 60)

    simple_graph = build_simple_graph()
    simple_inputs = ["Hello, how are you?", "Explain the difference between REST and GraphQL APIs."]

    for idx, text in enumerate(simple_inputs, 1):
        print(f'\n  --- Input {idx}: "{text}" ---')
        start = time.time()
        result = simple_graph.invoke({"input_text": text, "complexity": "", "final_response": ""})
        elapsed = time.time() - start
        print(f"  Classified as: {result['complexity']}")
        print(f"  Response: {result['final_response'][:120]}...")
        print(f"  Time: {elapsed:.2f}s")

    time.sleep(2)

    # ── Part 2: Research Agent ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2: Research Agent with Quality-Check Retry Loop")
    print("=" * 60)

    research_graph = build_research_graph()
    queries = [
        "What is MLflow tracing and how does it help with agent observability?",
        "Explain how LangGraph agents work and their common patterns.",
    ]

    all_stats: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="langgraph_agents_demo"):
        mlflow.set_tags(
            {
                "agent_type": "langgraph_research_assistant",
                "model": "google/gemma-4-26b-a4b",
                "graph_nodes": "analyze,search,synthesize,quality_check",
            }
        )

        for idx, query in enumerate(queries, 1):
            print(f"\n  --- Query {idx}: {query[:60]}... ---")
            start = time.time()
            result = research_graph.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "research_notes": "",
                    "current_step": "",
                    "quality_pass": False,
                    "retry_count": 0,
                }
            )
            elapsed = time.time() - start

            # Print final messages
            for msg in result["messages"]:
                content = message_text(msg)
                if content.startswith("[Synthesis]"):
                    print(f"  Answer: {content[12:120]}...")
                    break

            # Analyze the trace
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                try:
                    trace = mlflow.get_trace(trace_id, flush=True)
                    if trace and trace.data and trace.data.spans:
                        stats = analyze_trace(trace)
                        print_trace_analysis(f"Query {idx}", trace, stats)
                        all_stats.append(stats)
                except Exception as e:
                    print(f"  Trace logged (note: {e})")

            print(f"  Wall-clock time: {elapsed:.1f}s")

        # Log aggregate metrics
        print(f"\n{'=' * 60}")
        print("  Aggregate Metrics")
        print(f"{'=' * 60}")

        mlflow.log_metric("num_queries", len(queries))
        if all_stats:
            total_spans = sum(s["total_spans"] for s in all_stats)
            total_retries = sum(
                max(0, sum(1 for n in s["names"] if "search_knowledge" in n.lower()) - 1) for s in all_stats
            )
            durations = [s["total_duration_ms"] for s in all_stats if s["total_duration_ms"]]
            avg_dur = sum(durations) / len(durations) if durations else 0

            mlflow.log_metrics(
                {
                    "total_spans_visited": total_spans,
                    "total_retry_loops": total_retries,
                    "avg_trace_duration_ms": round(avg_dur, 1),
                }
            )
            print(f"  Total spans: {total_spans}")
            print(f"  Retry loops: {total_retries}")
            print(f"  Avg duration: {avg_dur:.1f} ms")

    print(f"\n{'=' * 60}")
    print("Done! Check the MLflow UI to explore traces and spans.")
    print("  Experiment: L2/M1_agent_frameworks/2_langgraph_agents")
    print("  Open any trace to see the full span tree with inputs/outputs.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
