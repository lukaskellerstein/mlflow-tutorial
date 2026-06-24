"""
L2-M4.1 — Tracing LangGraph State Machines

Builds a LangGraph workflow with conditional routing and demonstrates
how MLflow auto-tracing captures every node execution, state transition,
and conditional edge decision.
"""

import time
from typing import Literal

import mlflow
import mlflow.langchain
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L2/M4_advanced_tracing/1_langgraph_tracing")
mlflow.langchain.autolog()

llm = ChatOllama(model="gemma4:e2b", temperature=0.0)


# -- State --

class GraphState(TypedDict):
    messages: list
    input_text: str
    complexity: str
    processed_text: str
    final_response: str


# -- Nodes --

def classify_input(state: GraphState) -> dict:
    """Classify the input as simple or complex based on LLM judgment."""
    prompt = (
        "Classify the following user request as either SIMPLE or COMPLEX.\n"
        "SIMPLE = a short factual question or greeting.\n"
        "COMPLEX = a request that needs analysis, explanation, or creativity.\n"
        "Respond with exactly one word: SIMPLE or COMPLEX.\n\n"
        f"Request: {state['input_text']}"
    )
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    classification = response.content.strip().upper()
    complexity = "simple" if "SIMPLE" in classification else "complex"
    return {
        "complexity": complexity,
        "messages": [HumanMessage(content=state["input_text"]), response],
    }


def process_simple(state: GraphState) -> dict:
    """Handle simple inputs with a direct, concise response."""
    prompt = (
        "Give a brief, direct answer to the following simple question or greeting. "
        "Keep it under two sentences.\n\n"
        f"Question: {state['input_text']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"processed_text": response.content}


def process_complex(state: GraphState) -> dict:
    """Handle complex inputs with a detailed, structured response."""
    prompt = (
        "Provide a thorough answer to the following request. "
        "Use a short numbered list if appropriate.\n\n"
        f"Request: {state['input_text']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"processed_text": response.content}


def generate_response(state: GraphState) -> dict:
    """Produce the final formatted response."""
    tag = "Simple" if state["complexity"] == "simple" else "Detailed"
    prompt = (
        f"You are a helpful assistant. Rewrite the following [{tag}] answer "
        "into a friendly, polished response for the user.\n\n"
        f"Answer: {state['processed_text']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


# -- Routing --

def route_by_complexity(state: GraphState) -> Literal["process_simple", "process_complex"]:
    """Route to the appropriate processing node."""
    if state["complexity"] == "simple":
        return "process_simple"
    return "process_complex"


# -- Graph construction --

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("classify_input", classify_input)
    builder.add_node("process_simple", process_simple)
    builder.add_node("process_complex", process_complex)
    builder.add_node("generate_response", generate_response)

    # Edges
    builder.set_entry_point("classify_input")
    builder.add_conditional_edges("classify_input", route_by_complexity)
    builder.add_edge("process_simple", "generate_response")
    builder.add_edge("process_complex", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


# -- Trace analysis --

def analyze_traces(experiment_name: str) -> None:
    """Retrieve traces from MLflow and analyze span structure and timing."""
    print("\n" + "=" * 60)
    print("Step 3: Analyzing traces from MLflow")
    print("=" * 60)

    # Get the experiment id so we can scope the search
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("  Experiment not found — skipping trace analysis.")
        return

    # Search for traces — return as list of Trace objects
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        return_type="list",
    )
    print(f"\n  Found {len(traces)} trace(s) in the experiment.\n")

    for i, trace in enumerate(traces):
        print("-" * 50)
        print(f"  Trace {i + 1}  |  ID: {trace.info.trace_id}")
        print("-" * 50)

        spans = trace.data.spans
        if not spans:
            print("    (no spans)")
            continue

        # Build a list of (name, duration_ms, span_type, parent_id) tuples
        span_details = []
        for span in spans:
            start = span.start_time_ns or 0
            end = span.end_time_ns or start
            duration_ms = (end - start) / 1e6
            span_details.append({
                "name": span.name,
                "duration_ms": round(duration_ms, 1),
                "span_type": span.span_type or "UNKNOWN",
                "parent_id": span.parent_id,
                "span_id": span.span_id,
                "status": str(span.status),
            })

        # Print execution path (spans in order)
        print("    Execution path (spans):")
        for sd in span_details:
            indent = "      " if sd["parent_id"] else "    "
            print(f"{indent}-> {sd['name']:30s}  {sd['duration_ms']:>8.1f} ms  [{sd['span_type']}]")

        # Identify the nodes that were actually visited
        node_names = {"classify_input", "process_simple", "process_complex", "generate_response"}
        visited = [sd["name"] for sd in span_details if sd["name"] in node_names]
        print(f"\n    Visited nodes: {' -> '.join(visited) if visited else '(none detected)'}")

        # Determine which conditional branch was taken
        if "process_simple" in visited:
            print("    Conditional edge: classify_input --> process_simple  (SIMPLE path)")
        elif "process_complex" in visited:
            print("    Conditional edge: classify_input --> process_complex (COMPLEX path)")

        # Find the slowest node-level span (bottleneck analysis)
        node_spans = [sd for sd in span_details if sd["name"] in node_names]
        if node_spans:
            slowest = max(node_spans, key=lambda s: s["duration_ms"])
            print(f"\n    Bottleneck node: {slowest['name']} ({slowest['duration_ms']:.1f} ms)")

        # Total trace duration from root span
        root_spans = [sd for sd in span_details if sd["parent_id"] is None]
        if root_spans:
            total_ms = root_spans[0]["duration_ms"]
        else:
            total_ms = sum(sd["duration_ms"] for sd in node_spans) if node_spans else 0
        print(f"    Total trace duration: ~{total_ms:.1f} ms")

        print()


# -- Main --

def main() -> None:
    experiment_name = "L2/M4_advanced_tracing/1_langgraph_tracing"

    # -- Step 1: Build the graph --
    print("=" * 60)
    print("Step 1: Building the LangGraph workflow")
    print("=" * 60)
    graph = build_graph()
    print("  Graph compiled with nodes: classify_input, process_simple,")
    print("  process_complex, generate_response")
    print("  Conditional edge: classify_input -> route_by_complexity")
    print()

    # -- Step 2: Run the graph with different inputs --
    print("=" * 60)
    print("Step 2: Invoking the graph with different inputs")
    print("=" * 60)

    test_inputs = [
        "Hello, how are you?",
        "Explain the difference between REST and GraphQL APIs.",
        "What is 2 + 2?",
    ]

    for idx, text in enumerate(test_inputs, 1):
        print(f"\n  --- Input {idx}: \"{text}\" ---")
        initial_state: GraphState = {
            "messages": [],
            "input_text": text,
            "complexity": "",
            "processed_text": "",
            "final_response": "",
        }
        start = time.time()
        result = graph.invoke(initial_state)
        elapsed = time.time() - start

        print(f"  Classified as: {result['complexity']}")
        print(f"  Response preview: {result['final_response'][:120]}...")
        print(f"  Wall-clock time: {elapsed:.2f}s")

    # Give async trace logging a moment to flush
    time.sleep(2)

    # -- Step 3: Analyze traces --
    analyze_traces(experiment_name)

    print("=" * 60)
    print("Done! Open MLflow UI at http://127.0.0.1:5000")
    print(f"  Experiment: {experiment_name}")
    print("  Click on any trace to inspect the full span tree,")
    print("  inputs/outputs, and state transitions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
