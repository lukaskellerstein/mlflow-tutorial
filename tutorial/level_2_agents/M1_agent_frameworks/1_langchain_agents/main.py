"""
L2-5.1 — LangChain Agent Tracking with MLflow

Demonstrates how to build a ReAct agent using langchain's create_agent
(backed by LangGraph), instrument it with MLflow auto-tracing, and then
analyse the captured traces to understand the agent's decision chain.

Key concepts:
- Custom tool creation with @tool
- ReAct agent via langchain.agents.create_agent
- MLflow langchain autolog for automatic tracing
- Trace search and analysis after execution
"""

import time
from typing import cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Trace
from pydantic import SecretStr

# ── Part 1: Custom Tools ──────────────────────────────────────────


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
    words = text.split()
    return f"The text contains {len(words)} word(s)."


TOOLS = [calculator, string_reverser, word_counter]


# ── Part 2: Build a ReAct Agent ───────────────────────────────────


def build_agent():
    """Create a ReAct agent with ChatOpenAI and the custom tools."""
    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("lm-studio"),
        temperature=0.0,
    )
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt="You are a helpful assistant. Use the provided tools to answer questions. "
        "Always use a tool when the question involves calculation, string reversal, "
        "or word counting. Return the tool result directly as your final answer.",
    )
    return agent


# ── Part 3: Run the Agent on Several Tasks ────────────────────────

TASKS = [
    "What is 15 * 23?",
    "Reverse the word 'MLflow'",
    "Count the words in the sentence: 'MLflow is a great platform for tracking experiments'",
]


def run_agent_tasks(agent) -> list[dict]:
    """Invoke the agent on each task and collect run metadata."""
    results = []

    with mlflow.start_run(run_name="langchain_agent_tasks"):
        mlflow.set_tags(
            {
                "agent_type": "react",
                "model": "google/gemma-4-26b-a4b",
                "num_tools": str(len(TOOLS)),
                "tool_names": ", ".join(t.name for t in TOOLS),
            }
        )

        for idx, task in enumerate(TASKS, start=1):
            print(f"\n{'=' * 60}")
            print(f"Task {idx}: {task}")
            print("=" * 60)

            start = time.time()
            response = agent.invoke({"messages": [{"role": "user", "content": task}]})
            elapsed = time.time() - start

            # Extract the final answer from the last message
            final_message = response["messages"][-1]
            answer = final_message.content
            print(f"Answer : {answer}")
            print(f"Latency: {elapsed:.2f}s")

            # Count tool calls and messages (reasoning steps)
            messages = response["messages"]
            tool_call_count = sum(1 for m in messages if m.type == "tool")
            total_steps = len(messages)

            # Log per-task metrics inside a nested run
            with mlflow.start_run(run_name=f"task_{idx}", nested=True):
                mlflow.log_params(
                    {
                        "task": task,
                        "task_index": idx,
                    }
                )
                mlflow.log_metrics(
                    {
                        "latency_seconds": round(elapsed, 3),
                        "tool_calls": tool_call_count,
                        "total_steps": total_steps,
                        "success": 1,
                    }
                )

            results.append(
                {
                    "task_index": idx,
                    "task": task,
                    "answer": answer,
                    "tool_calls": tool_call_count,
                    "total_steps": total_steps,
                    "latency": round(elapsed, 3),
                }
            )

        # Log aggregate metrics on the parent run
        mlflow.log_metrics(
            {
                "total_tasks": len(TASKS),
                "avg_latency": round(sum(r["latency"] for r in results) / len(results), 3),
                "total_tool_calls": sum(r["tool_calls"] for r in results),
            }
        )

    return results


# ── Part 4: Analyse Traces ────────────────────────────────────────


def analyse_traces() -> None:
    """Search traces captured by autolog and print the agent decision chain."""
    print("\n" + "=" * 60)
    print("Part 4: Trace Analysis")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name("L2/M5_agent_observability/1_langchain_agents")
    if experiment is None:
        print("  No experiment found — skipping trace analysis.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(
            locations=[experiment.experiment_id],
            return_type="list",
        ),
    )

    if not traces:
        print("  No traces found.")
        return

    print(f"\n  Found {len(traces)} trace(s).\n")

    for i, trace in enumerate(traces, start=1):
        trace_id = trace.info.trace_id
        spans = trace.data.spans

        print(f"  --- Trace {i} (ID: {trace_id}) ---")
        print(f"  Number of spans: {len(spans)}")

        # Walk the spans to show the decision chain
        for span in spans:
            span_name = span.name
            span_type = span.span_type
            status = span.status

            indent = "    "
            print(f"{indent}[{span_type}] {span_name} — {status}")

            # Show tool inputs/outputs for tool spans
            if span_type and "TOOL" in str(span_type).upper():
                if span.inputs:
                    inputs_str = str(span.inputs)
                    if len(inputs_str) > 120:
                        inputs_str = inputs_str[:120] + "..."
                    print(f"{indent}  Inputs : {inputs_str}")
                if span.outputs:
                    outputs_str = str(span.outputs)
                    if len(outputs_str) > 120:
                        outputs_str = outputs_str[:120] + "..."
                    print(f"{indent}  Outputs: {outputs_str}")

        print()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("L2-5.1 — LangChain Agent Tracking with MLflow")
    print("=" * 60)

    # Step 1: Enable auto-tracing
    print("\nStep 1: Enabling MLflow LangChain autolog")
    mlflow.langchain.autolog(log_traces=True)

    # Step 2: Build the agent
    print("Step 2: Building ReAct agent with google/gemma-4-26b-a4b + 3 tools")
    agent = build_agent()

    # Step 3: Run tasks
    print("\nStep 3: Running agent on tasks")
    results = run_agent_tasks(agent)

    # Print summary table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Task':<6} {'Tool Calls':<12} {'Steps':<8} {'Latency':<10} {'Answer'}")
    print("-" * 80)
    for r in results:
        answer_short = r["answer"][:40] + "..." if len(r["answer"]) > 40 else r["answer"]
        print(
            f"{r['task_index']:<6} {r['tool_calls']:<12} {r['total_steps']:<8} {r['latency']:<10.3f} {answer_short}"
        )

    # Step 4: Analyse traces
    analyse_traces()

    print("=" * 60)
    print("Done! View traces in the MLflow UI:")
    print("  http://127.0.0.1:5555/#/experiments")
    print("  Look for experiment: L2/M5_agent_observability/1_langchain_agents")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M1_agent_frameworks/1_langchain_agents")
    main()
