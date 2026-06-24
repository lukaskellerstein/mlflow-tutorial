"""
L3-1.3 — Agent Architecture Comparison

Systematically compare three agent architectures on the same Q&A task
with shared tools and evaluation criteria:

  1. Simple Chain    — prompt -> LLM -> answer (no tools, no loop)
  2. ReAct Agent     — langchain.agents.create_agent with tools
  3. Multi-step Pipe — LangGraph StateGraph: classify -> process -> respond

All architectures use ChatOllama(model="gemma4:e2b") and are evaluated
on the same 5-question benchmark.  Results are logged as nested MLflow
runs and printed as a comparison table.
"""

import time
from typing import Annotated

import mlflow
import mlflow.langchain
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L3/M1_agent_evaluation/3_architecture_comparison")
mlflow.langchain.autolog(log_traces=True)

# ---------------------------------------------------------------------------
# Shared LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model="gemma4:e2b", temperature=0.0)

# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------
KNOWLEDGE: dict[str, str] = {
    "python": "Python is a high-level programming language known for readability. "
              "It supports multiple paradigms and has a vast ecosystem of libraries.",
    "mlflow": "MLflow is an open-source platform for the ML lifecycle. It provides "
              "tracking, model registry, evaluation, and deployment capabilities.",
    "docker": "Docker is a containerization platform that packages applications "
              "and dependencies into lightweight, portable containers.",
    "kubernetes": "Kubernetes is an open-source container orchestration system that "
                  "automates deployment, scaling, and management of containers.",
    "langchain": "LangChain is a framework for building LLM-powered applications. "
                 "It provides abstractions for chains, agents, memory, and tools.",
}


@tool
def lookup(topic: str) -> str:
    """Look up factual information about a technology topic.

    Args:
        topic: The technology topic to look up (e.g. 'python', 'mlflow').
    """
    key = topic.strip().lower()
    for k, v in KNOWLEDGE.items():
        if k in key or key in k:
            return v
    return f"No information found for '{topic}'. Known topics: {', '.join(KNOWLEDGE)}."


@tool
def word_count(text: str) -> str:
    """Count the number of words in the given text.

    Args:
        text: The text to count words in.
    """
    return f"Word count: {len(text.split())}"


TOOLS = [lookup, word_count]

# ---------------------------------------------------------------------------
# Evaluation dataset — 5 test cases
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question": "What is Python and what is it known for?",
        "expected_keyword": "readability",
        "needs_tool": True,
    },
    {
        "question": "Describe what MLflow does.",
        "expected_keyword": "tracking",
        "needs_tool": True,
    },
    {
        "question": "What is 2 + 2?",
        "expected_keyword": "4",
        "needs_tool": False,
    },
    {
        "question": "Explain what Docker containers are.",
        "expected_keyword": "container",
        "needs_tool": True,
    },
    {
        "question": "Say hello in French.",
        "expected_keyword": "bonjour",
        "needs_tool": False,
    },
]


# ===================================================================
# Architecture 1: Simple Chain (no tools, no loop)
# ===================================================================
def run_simple_chain(question: str) -> dict:
    """Prompt -> LLM -> answer.  No tool access."""
    start = time.time()
    response = llm.invoke([
        SystemMessage(content="Answer the question concisely in 1-2 sentences."),
        HumanMessage(content=question),
    ])
    elapsed = time.time() - start
    answer = response.content
    token_est = len(answer.split())  # rough proxy
    return {"answer": answer, "latency": elapsed, "tool_calls": 0, "tokens_est": token_est}


# ===================================================================
# Architecture 2: ReAct Agent (langchain create_agent)
# ===================================================================
_react_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="You are a helpful assistant. Use the provided tools when the "
                  "question is about a technology topic. Answer concisely.",
)


def run_react_agent(question: str) -> dict:
    """ReAct loop with tool access."""
    start = time.time()
    result = _react_agent.invoke({"messages": [HumanMessage(content=question)]})
    elapsed = time.time() - start
    messages = result["messages"]
    answer = messages[-1].content if messages else ""
    tc = sum(1 for m in messages if m.type == "tool")
    token_est = sum(len(m.content.split()) for m in messages if hasattr(m, "content") and m.content)
    return {"answer": answer, "latency": elapsed, "tool_calls": tc, "tokens_est": token_est}


# ===================================================================
# Architecture 3: Multi-step Pipeline (StateGraph)
# ===================================================================
class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    context: str
    answer: str


def classify_node(state: PipelineState) -> dict:
    """Classify the question as 'tech_lookup', 'general', or 'math'."""
    user_msg = state["messages"][-1]
    q = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)
    resp = llm.invoke([
        SystemMessage(
            content="Classify the following question into exactly one category. "
                    "Reply with ONLY the category name, nothing else.\n"
                    "Categories: tech_lookup, general, math"
        ),
        HumanMessage(content=q),
    ])
    cat = resp.content.strip().lower().replace("'", "").replace('"', '')
    # Normalize to one of the known categories
    if "tech" in cat or "lookup" in cat:
        cat = "tech_lookup"
    elif "math" in cat:
        cat = "math"
    else:
        cat = "general"
    return {"category": cat}


def process_node(state: PipelineState) -> dict:
    """Fetch context using tools if the category warrants it."""
    category = state.get("category", "general")
    user_msg = state["messages"][0]
    q = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)

    if category == "tech_lookup":
        ctx = lookup.invoke(q)
    else:
        ctx = ""
    return {"context": ctx}


def respond_node(state: PipelineState) -> dict:
    """Generate the final answer using LLM + gathered context."""
    user_msg = state["messages"][0]
    q = user_msg.content if isinstance(user_msg, HumanMessage) else str(user_msg)
    ctx = state.get("context", "")

    prompt_parts = [
        SystemMessage(content="Answer concisely in 1-2 sentences."),
    ]
    if ctx:
        prompt_parts.append(SystemMessage(content=f"Context: {ctx}"))
    prompt_parts.append(HumanMessage(content=q))

    resp = llm.invoke(prompt_parts)
    return {
        "answer": resp.content,
        "messages": [AIMessage(content=resp.content)],
    }


def _build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("classify", classify_node)
    g.add_node("process", process_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "process")
    g.add_edge("process", "respond")
    g.set_finish_point("respond")
    return g.compile()


_pipeline = _build_pipeline()


def run_pipeline(question: str) -> dict:
    """Classify -> process -> respond pipeline."""
    start = time.time()
    result = _pipeline.invoke({
        "messages": [HumanMessage(content=question)],
        "category": "",
        "context": "",
        "answer": "",
    })
    elapsed = time.time() - start
    answer = result.get("answer", "")
    # The pipeline uses the lookup tool directly in the process node
    tool_calls = 1 if result.get("context", "") else 0
    token_est = len(answer.split())
    return {"answer": answer, "latency": elapsed, "tool_calls": tool_calls, "tokens_est": token_est}


# ===================================================================
# Scoring helpers
# ===================================================================
def score_correctness(answer: str, expected_keyword: str) -> float:
    """1.0 if the expected keyword appears in the answer, else 0.0."""
    return 1.0 if expected_keyword.lower() in answer.lower() else 0.0


def score_tool_usage(tool_calls: int, needs_tool: bool) -> float:
    """1.0 if tool use matches expectation, 0.0 otherwise."""
    if needs_tool:
        return 1.0 if tool_calls > 0 else 0.0
    return 1.0 if tool_calls == 0 else 0.5  # penalize unnecessary tool use


# ===================================================================
# Main evaluation loop
# ===================================================================
ARCHITECTURES: list[tuple[str, callable]] = [
    ("simple_chain", run_simple_chain),
    ("react_agent", run_react_agent),
    ("multi_step_pipeline", run_pipeline),
]


def evaluate_architecture(
    name: str, run_fn: callable, dataset: list[dict]
) -> list[dict]:
    """Run an architecture on every test case and return per-case metrics."""
    rows = []
    for i, case in enumerate(dataset, 1):
        result = run_fn(case["question"])
        correctness = score_correctness(result["answer"], case["expected_keyword"])
        tool_score = score_tool_usage(result["tool_calls"], case["needs_tool"])
        rows.append({
            "architecture": name,
            "case": i,
            "question": case["question"],
            "answer": result["answer"][:120],
            "correctness": correctness,
            "tool_usage": tool_score,
            "latency_s": round(result["latency"], 3),
            "tokens_est": result["tokens_est"],
        })
    return rows


def main() -> None:
    print("=" * 70)
    print("  L3-1.3 — Agent Architecture Comparison")
    print("=" * 70)

    all_rows: list[dict] = []

    with mlflow.start_run(run_name="architecture_comparison") as parent:
        mlflow.set_tags({
            "comparison_type": "architecture",
            "num_architectures": str(len(ARCHITECTURES)),
            "num_test_cases": str(len(EVAL_DATASET)),
            "model": "gemma4:e2b",
        })

        for arch_name, arch_fn in ARCHITECTURES:
            print(f"\n{'─' * 70}")
            print(f"  Evaluating: {arch_name}")
            print(f"{'─' * 70}")

            with mlflow.start_run(run_name=arch_name, nested=True):
                rows = evaluate_architecture(arch_name, arch_fn, EVAL_DATASET)
                all_rows.extend(rows)

                # Compute aggregate metrics for this architecture
                avg_correct = sum(r["correctness"] for r in rows) / len(rows)
                avg_tool = sum(r["tool_usage"] for r in rows) / len(rows)
                avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
                total_tokens = sum(r["tokens_est"] for r in rows)
                avg_tokens = total_tokens / len(rows)

                mlflow.log_params({
                    "architecture": arch_name,
                    "model": "gemma4:e2b",
                    "num_cases": len(rows),
                })
                mlflow.log_metrics({
                    "avg_correctness": round(avg_correct, 3),
                    "avg_tool_usage": round(avg_tool, 3),
                    "avg_latency_s": round(avg_latency, 3),
                    "total_tokens_est": total_tokens,
                    "avg_tokens_est": round(avg_tokens, 1),
                })

                # Print per-case results
                for r in rows:
                    status = "PASS" if r["correctness"] == 1.0 else "FAIL"
                    print(f"  [{status}] Q{r['case']}: {r['question'][:50]}")
                    print(f"         Answer: {r['answer'][:80]}")
                    print(f"         Correctness={r['correctness']:.0f}  "
                          f"ToolUsage={r['tool_usage']:.1f}  "
                          f"Latency={r['latency_s']:.2f}s")

        # -----------------------------------------------------------
        # Build comparison table
        # -----------------------------------------------------------
        df = pd.DataFrame(all_rows)
        summary = df.groupby("architecture").agg(
            correctness=("correctness", "mean"),
            tool_usage=("tool_usage", "mean"),
            latency_s=("latency_s", "mean"),
            tokens_est=("tokens_est", "sum"),
        ).round(3)

        # Add composite score: quality (equal weight correctness + tool_usage)
        summary["quality"] = ((summary["correctness"] + summary["tool_usage"]) / 2).round(3)

        # Token efficiency = quality / tokens (higher is better)
        summary["token_efficiency"] = (
            summary["quality"] / summary["tokens_est"].clip(lower=1) * 100
        ).round(3)

        print(f"\n{'=' * 70}")
        print("  COMPARISON TABLE: Architecture x Metric")
        print(f"{'=' * 70}")
        print()
        header = (f"  {'Architecture':<22} {'Correct':>8} {'ToolUse':>8} "
                  f"{'Latency':>8} {'Tokens':>7} {'Quality':>8} {'Efficiency':>10}")
        print(header)
        print("  " + "-" * 74)
        for arch, row in summary.iterrows():
            print(f"  {arch:<22} {row['correctness']:>8.3f} {row['tool_usage']:>8.3f} "
                  f"{row['latency_s']:>7.2f}s {int(row['tokens_est']):>7} "
                  f"{row['quality']:>8.3f} {row['token_efficiency']:>10.3f}")

        # Log comparison artifact
        summary_path = "comparison_table.csv"
        summary.to_csv(summary_path)
        mlflow.log_artifact(summary_path)

        # Log parent-level summary
        best_quality = summary["quality"].idxmax()
        fastest = summary["latency_s"].idxmin()
        most_efficient = summary["token_efficiency"].idxmax()

        mlflow.log_params({
            "best_quality_arch": best_quality,
            "fastest_arch": fastest,
            "most_efficient_arch": most_efficient,
        })

        # -----------------------------------------------------------
        # Cost-quality tradeoff analysis
        # -----------------------------------------------------------
        print(f"\n{'=' * 70}")
        print("  COST-QUALITY TRADEOFF ANALYSIS")
        print(f"{'=' * 70}")
        print(f"\n  Best quality:       {best_quality} "
              f"(score={summary.loc[best_quality, 'quality']:.3f})")
        print(f"  Fastest:            {fastest} "
              f"(latency={summary.loc[fastest, 'latency_s']:.3f}s)")
        print(f"  Most efficient:     {most_efficient} "
              f"(efficiency={summary.loc[most_efficient, 'token_efficiency']:.3f})")

        # Pareto frontier: architectures not dominated on both quality and latency
        print(f"\n  Pareto Frontier (quality vs latency):")
        pareto = []
        for arch in summary.index:
            dominated = False
            for other in summary.index:
                if other == arch:
                    continue
                if (summary.loc[other, "quality"] >= summary.loc[arch, "quality"]
                        and summary.loc[other, "latency_s"] <= summary.loc[arch, "latency_s"]
                        and (summary.loc[other, "quality"] > summary.loc[arch, "quality"]
                             or summary.loc[other, "latency_s"] < summary.loc[arch, "latency_s"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(arch)
                print(f"    * {arch}: quality={summary.loc[arch, 'quality']:.3f}, "
                      f"latency={summary.loc[arch, 'latency_s']:.3f}s")

        mlflow.set_tag("pareto_frontier", ", ".join(pareto))

        print(f"\n  Parent Run ID: {parent.info.run_id}")
        print(f"  View in MLflow UI: http://127.0.0.1:5000")

    print(f"\n{'=' * 70}")
    print("  Done! Check nested runs in the MLflow UI for full details.")
    print(f"{'=' * 70}")

    # Clean up temp file
    import os
    if os.path.exists(summary_path):
        os.remove(summary_path)


if __name__ == "__main__":
    main()
