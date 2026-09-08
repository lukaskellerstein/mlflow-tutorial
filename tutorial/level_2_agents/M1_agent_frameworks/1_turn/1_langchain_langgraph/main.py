"""
L2-M1.1.1 — LangChain + LangGraph Agents with MLflow

One ReAct agent, built once and used everywhere:

  Part 1  langchain.agents.create_agent — build it, log a picture of the graph
          it compiled, run every task through it
  Part 2  trace analysis

create_agent RETURNS a compiled LangGraph StateGraph. That is why a single
mlflow.langchain.autolog() call captures the whole loop, and why the agent can
draw its own graph: the state machine is already there, you just did not have to
write the nodes and edges yourself.

No chains anywhere. LangChain v1 agents are graphs, not LCEL pipelines.
"""

import tempfile
import time
from pathlib import Path
from typing import cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Trace
from pydantic import SecretStr

# The MLflow AI Gateway -- the tracking server itself, not a provider directly.
# "gemma-agent" is an alias defined in infra/mlflow/gateway/seed_gateway.py: it resolves
# to a local Unsloth model, and there is no fallback -- so what the run says
# answered is what answered. Swapping model or provider is a change there, never here.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/1_turn/1_langchain_langgraph"

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


# ── Tools ─────────────────────────────────────────────────────────


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
    """Chat model pointed at the MLflow AI Gateway."""
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )


# ── Part 1: the agent ─────────────────────────────────────────────


def build_agent():
    """create_agent gives you the ReAct loop already wired and compiled."""
    return create_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def log_compiled_graph(agent) -> None:
    """Log the graph the agent compiled, as a PNG the MLflow UI displays inline.

    The agent IS a compiled StateGraph, so it can draw itself. draw_mermaid_png()
    renders through the mermaid.ink service -- the one call in this lesson that
    leaves the machine. The MLflow UI has no mermaid renderer, so the raw text
    would show up as a code block; a PNG shows up as the picture.

    If the service cannot be reached, log the mermaid source instead. A missing
    diagram must not take the whole run down, but it is never swallowed quietly.
    """
    graph = agent.get_graph()

    try:
        png = graph.draw_mermaid_png()
    except Exception as exc:
        print(f"\n  PNG render failed ({type(exc).__name__}: {exc})")
        print("  Logging the mermaid source as graph.md instead.")
        mlflow.log_text(
            f"# The graph create_agent compiled\n\n```mermaid\n{graph.draw_mermaid()}```\n",
            "graph.md",
        )
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        png_path = Path(tmp_dir) / "graph.png"
        png_path.write_bytes(png)
        mlflow.log_artifact(str(png_path))

    print(f"\n  Graph image logged as graph.png ({len(png)} bytes)")


def run_tasks(agent) -> None:
    """Log the agent's compiled graph, then run every task in a nested run."""
    latencies: list[float] = []
    total_tool_calls = 0
    total_steps = 0

    with mlflow.start_run(run_name="create_agent"):
        mlflow.set_tags(
            {
                "framework": "langchain_v1_create_agent",
                "model_alias": MODEL_ALIAS,
                "gateway": "mlflow-ai-gateway",
                "tool_names": ", ".join(t.name for t in TOOLS),
            }
        )

        # On this run, not on a task run: the graph describes the agent, not one task.
        log_compiled_graph(agent)

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

            with mlflow.start_run(run_name=f"task_{idx}", nested=True):
                mlflow.log_params({"task": task, "task_index": idx})
                mlflow.log_metrics(
                    {
                        "latency_seconds": round(elapsed, 3),
                        "tool_calls": tool_calls,
                        "total_steps": len(messages),
                    }
                )

            latencies.append(round(elapsed, 3))
            total_tool_calls += tool_calls
            total_steps += len(messages)

        mlflow.log_metrics(
            {
                "avg_latency": round(sum(latencies) / len(latencies), 3),
                "total_tool_calls": total_tool_calls,
                "total_steps": total_steps,
            }
        )


# ── Part 2: trace analysis ────────────────────────────────────────


def analyse_traces() -> None:
    """Walk the spans autolog captured and show the shape of one agent turn."""
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
    print("L2-M1.1.1 — LangChain + LangGraph Agents with MLflow")
    print("=" * 60)

    # One call instruments the whole agent: create_agent returns a StateGraph, so
    # LangChain autolog covers every node, model call and tool call underneath.
    mlflow.langchain.autolog(log_traces=True)

    # Built once, used by every part below.
    agent = build_agent()

    print("\n" + "=" * 60)
    print("Part 1: create_agent — the ReAct agent")
    print("=" * 60)
    run_tasks(agent)

    print("\n" + "=" * 60)
    print("Part 2: Trace analysis")
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
