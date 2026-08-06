"""
L2-M1.1 — LangChain + LangGraph Agents with MLflow

Two ways to build the same ReAct agent, traced side by side:

  Part 1  langchain.agents.create_agent  — the prebuilt agent (LangChain v1)
  Part 2  langgraph.StateGraph           — the same loop, hand-built
  Part 3  comparison and trace analysis

create_agent RETURNS a compiled StateGraph, so Part 2 is not an alternative to
Part 1 — it is what Part 1 builds for you. Seeing both traced in MLflow is the
point: the prebuilt agent hides the node boundaries, the hand-built graph makes
every state transition a span you can inspect.

No chains anywhere. LangChain v1 agents are graphs, not LCEL pipelines.
"""

import time
from typing import Annotated, Any, Literal, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mlflow.entities import Trace
from pydantic import SecretStr
from typing_extensions import TypedDict

# The LiteLLM gateway from infra/, not a provider directly. "gemma-agent" is an
# alias defined in infra/litellm/config.yaml: it starts on the free OpenRouter
# tier and the proxy falls back to the paid model when free rate-limits or 404s.
# Swapping model or provider is a change there, never here.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/1_langchain_langgraph"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools to answer questions. "
    "Always use a tool when the question involves calculation, string reversal, "
    "or word counting. Return the tool result directly as your final answer."
)

TASKS = [
    "What is 15 * 23?",
    "Reverse the word 'MLflow'",
    "Count the words in the sentence: 'MLflow is a great platform for tracking experiments'",
]


# ── Tools — shared by both agents ─────────────────────────────────


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Supports +, -, *, /, and parentheses.

    Args:
        expression: A math expression string, e.g. '15 * 23'.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return f"Error: expression contains invalid characters: {expression}"
    try:
        result = eval(expression)  # safe: only digits and operators allowed
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def string_reverser(text: str) -> str:
    """Reverse the characters in a given string.

    Args:
        text: The string to reverse.
    """
    return text[::-1]


@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text.

    Args:
        text: The text whose words should be counted.
    """
    return f"The text contains {len(text.split())} word(s)."


TOOLS = [calculator, string_reverser, word_counter]


def get_llm() -> ChatOpenAI:
    """Chat model pointed at the LiteLLM gateway."""
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )


# ── Part 1: the prebuilt agent ────────────────────────────────────


def build_prebuilt_agent():
    """create_agent gives you the ReAct loop already wired and compiled."""
    return create_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# ── Part 2: the same loop, hand-built ─────────────────────────────


class AgentState(TypedDict):
    """add_messages appends to the list instead of overwriting it."""

    messages: Annotated[list, add_messages]


def build_graph_agent():
    """Build the ReAct loop node by node, so each transition is its own span."""
    llm_with_tools = get_llm().bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        """The reasoning node: decide to answer, or to call a tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        """The conditional edge: loop back through tools, or stop."""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "__end__"

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    # ToolNode executes every tool call on the last message and appends the
    # ToolMessages — the part people most often hand-roll and get subtly wrong.
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "model")
    return builder.compile()


# ── Running and measuring ─────────────────────────────────────────


def run_variant(agent, variant: str) -> list[dict]:
    """Run every task through one agent, logging a nested run per task."""
    results: list[dict] = []

    with mlflow.start_run(run_name=variant):
        mlflow.set_tags(
            {
                "variant": variant,
                "model_alias": MODEL_ALIAS,
                "gateway": "litellm",
                "tool_names": ", ".join(t.name for t in TOOLS),
            }
        )

        for idx, task in enumerate(TASKS, start=1):
            print(f"\n  Task {idx}: {task}")

            start = time.time()
            state = agent.invoke({"messages": [{"role": "user", "content": task}]})
            elapsed = time.time() - start

            messages = state["messages"]
            answer = str(messages[-1].content)
            tool_calls = sum(1 for m in messages if m.type == "tool")

            print(f"    Answer     : {answer[:100]}")
            print(f"    Tool calls : {tool_calls}   Steps: {len(messages)}   {elapsed:.2f}s")

            with mlflow.start_run(run_name=f"{variant}_task_{idx}", nested=True):
                mlflow.log_params({"task": task, "task_index": idx, "variant": variant})
                mlflow.log_metrics(
                    {
                        "latency_seconds": round(elapsed, 3),
                        "tool_calls": tool_calls,
                        "total_steps": len(messages),
                    }
                )

            results.append(
                {
                    "task_index": idx,
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "total_steps": len(messages),
                    "latency": round(elapsed, 3),
                }
            )

        mlflow.log_metrics(
            {
                "avg_latency": round(sum(r["latency"] for r in results) / len(results), 3),
                "total_tool_calls": sum(r["tool_calls"] for r in results),
                "total_steps": sum(r["total_steps"] for r in results),
            }
        )

    return results


# ── Part 3: comparison and trace analysis ─────────────────────────


def compare(prebuilt: list[dict], graph: list[dict]) -> None:
    """Log both variants side by side so the UI can diff them."""
    table: dict[str, list[Any]] = {
        "variant": [],
        "task_index": [],
        "tool_calls": [],
        "total_steps": [],
        "latency_seconds": [],
    }
    for variant, results in (("create_agent", prebuilt), ("state_graph", graph)):
        for r in results:
            table["variant"].append(variant)
            table["task_index"].append(r["task_index"])
            table["tool_calls"].append(r["tool_calls"])
            table["total_steps"].append(r["total_steps"])
            table["latency_seconds"].append(r["latency"])

    print(f"\n  {'variant':<14} {'task':<6} {'tools':<7} {'steps':<7} {'latency'}")
    print("  " + "-" * 48)
    for i in range(len(table["variant"])):
        print(
            f"  {table['variant'][i]:<14} {table['task_index'][i]:<6} "
            f"{table['tool_calls'][i]:<7} {table['total_steps'][i]:<7} "
            f"{table['latency_seconds'][i]:.2f}s"
        )

    with mlflow.start_run(run_name="variant_comparison"):
        mlflow.set_tag("run_type", "comparison")
        mlflow.log_table(data=table, artifact_file="comparison.json")
        mlflow.log_metrics(
            {
                "create_agent_avg_latency": round(sum(r["latency"] for r in prebuilt) / len(prebuilt), 3),
                "state_graph_avg_latency": round(sum(r["latency"] for r in graph) / len(graph), 3),
                "create_agent_total_steps": sum(r["total_steps"] for r in prebuilt),
                "state_graph_total_steps": sum(r["total_steps"] for r in graph),
            }
        )
        # draw_mermaid() is pure text — no mermaid.ink round-trip, so this works
        # offline. The MLflow UI renders the .md artifact with the diagram.
        graph_agent = build_graph_agent()
        mermaid = graph_agent.get_graph().draw_mermaid()
        mlflow.log_text(f"# Hand-built agent graph\n\n```mermaid\n{mermaid}```\n", "graph.md")
        print(f"\n  Graph structure logged as graph.md:\n{mermaid}")


def analyse_traces() -> None:
    """Walk the spans autolog captured and show where the two shapes differ."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  No experiment found — skipping trace analysis.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(locations=[experiment.experiment_id], return_type="list", flush=True),
    )
    if not traces:
        print("  No traces found.")
        return

    print(f"  Found {len(traces)} trace(s). Span breakdown of the 2 most recent:\n")
    for trace in traces[:2]:
        spans = trace.data.spans
        print(f"  Trace {trace.info.trace_id[:16]}... — {len(spans)} spans, {trace.info.execution_time_ms}ms")
        for span in spans:
            duration_ms = ((span.end_time_ns or 0) - (span.start_time_ns or 0)) / 1e6
            print(f"      [{span.span_type}] {span.name} ({duration_ms:.0f}ms)")
        print()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("L2-M1.1 — LangChain + LangGraph Agents with MLflow")
    print("=" * 60)

    # One call instruments both variants: create_agent returns a StateGraph, so
    # LangChain autolog covers the hand-built graph too.
    mlflow.langchain.autolog(log_traces=True)

    print("\n" + "=" * 60)
    print("Part 1: create_agent — the prebuilt ReAct agent")
    print("=" * 60)
    prebuilt = run_variant(build_prebuilt_agent(), "create_agent")

    print("\n" + "=" * 60)
    print("Part 2: StateGraph — the same loop, hand-built")
    print("=" * 60)
    graph = run_variant(build_graph_agent(), "state_graph")

    print("\n" + "=" * 60)
    print("Part 3: Comparison")
    print("=" * 60)
    compare(prebuilt, graph)

    print("\n" + "=" * 60)
    print("Part 4: Trace analysis")
    print("=" * 60)
    analyse_traces()

    print("=" * 60)
    print("Done. View traces in the MLflow UI:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT)
    main()
