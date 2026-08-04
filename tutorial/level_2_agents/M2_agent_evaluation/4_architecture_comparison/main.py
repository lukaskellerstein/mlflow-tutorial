"""
L2-M2.4 — Agent Architecture Comparison

Systematically compare three agent architectures on the same Q&A task
with shared tools and evaluation criteria:

  1. Simple Chain    — prompt -> LLM -> answer (no tools, no loop)
  2. ReAct Agent     — langchain.agents.create_agent with tools
  3. Multi-step Pipe — LangGraph StateGraph: classify -> process -> respond

All three run on the same model (the `gemma-large` gateway alias) against the same
5-question benchmark, and -- the part that makes the comparison trustworthy --
they are all scored by the SAME registered correctness judge through
`mlflow.genai.evaluate()`, not by a local substring check. Results are logged as
nested MLflow runs and printed as a comparison table with a cost-quality Pareto
frontier.
"""

import os
import time
from collections.abc import Callable
from typing import Annotated, Any, cast

import mlflow
import mlflow.langchain
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.scorers import scorer
from pydantic import SecretStr
from typing_extensions import TypedDict

# ── MLflow setup ──────────────────────────────────────────────────
# The LiteLLM gateway from infra/, not a provider directly. "gemma-large" is an
# alias defined in infra/litellm/config.yaml -- swapping model or provider is a
# change there, never here. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-large"

# The correctness judge resolves its model through LiteLLM, which reads these.
# Assignments, not setdefault -- a real OPENAI_API_KEY in the environment would
# win and every judge call would be rejected by the gateway. See L2-M2.2.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L2/M2_agent_evaluation/3_architecture_comparison")
mlflow.langchain.autolog(log_traces=True)

# ── Shared LLM ────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL_ALIAS,
    base_url=GATEWAY_URL,
    api_key=SecretStr(GATEWAY_KEY),
    temperature=0.0,
)

# ── Shared tools ──────────────────────────────────────────────────
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

# ── Evaluation dataset — 5 test cases ─────────────────────────────
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


# ── Architecture 1: Simple Chain (no tools, no loop) ──────────────
def run_simple_chain(question: str) -> dict:
    """Prompt -> LLM -> answer.  No tool access."""
    start = time.time()
    response = llm.invoke(
        [
            {"role": "system", "content": "Answer the question concisely in 1-2 sentences."},
            {"role": "user", "content": question},
        ]
    )
    elapsed = time.time() - start
    answer = str(response.content)
    token_est = len(answer.split())  # rough proxy
    return {"answer": answer, "latency": elapsed, "tool_calls": 0, "tokens_est": token_est}


# ── Architecture 2: ReAct Agent (langchain create_agent) ──────────
_react_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="You are a helpful assistant. Use the provided tools when the "
    "question is about a technology topic. Answer concisely.",
)


def run_react_agent(question: str) -> dict:
    """ReAct loop with tool access."""
    start = time.time()
    result = _react_agent.invoke({"messages": [{"role": "user", "content": question}]})
    elapsed = time.time() - start
    messages = result["messages"]
    answer = messages[-1].content if messages else ""
    tc = sum(1 for m in messages if m.type == "tool")
    token_est = sum(len(m.content.split()) for m in messages if hasattr(m, "content") and m.content)
    return {"answer": answer, "latency": elapsed, "tool_calls": tc, "tokens_est": token_est}


# ── Architecture 3: Multi-step Pipeline (StateGraph) ──────────────
class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    context: str
    answer: str


def classify_node(state: PipelineState) -> dict:
    """Classify the question as 'tech_lookup', 'general', or 'math'."""
    user_msg = state["messages"][-1]
    q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
    resp = llm.invoke(
        [
            {
                "role": "system",
                "content": "Classify the following question into exactly one category. "
                "Reply with ONLY the category name, nothing else.\n"
                "Categories: tech_lookup, general, math",
            },
            {"role": "user", "content": q},
        ]
    )
    cat = str(resp.content).strip().lower().replace("'", "").replace('"', "")
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
    q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    if category == "tech_lookup":
        ctx = lookup.invoke(q)
    else:
        ctx = ""
    return {"context": ctx}


def respond_node(state: PipelineState) -> dict:
    """Generate the final answer using LLM + gathered context."""
    user_msg = state["messages"][0]
    q = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
    ctx = state.get("context", "")

    prompt_parts = [
        {"role": "system", "content": "Answer concisely in 1-2 sentences."},
    ]
    if ctx:
        prompt_parts.append({"role": "system", "content": f"Context: {ctx}"})
    prompt_parts.append({"role": "user", "content": q})

    resp = llm.invoke(prompt_parts)
    return {
        "answer": resp.content,
        "messages": [{"role": "assistant", "content": resp.content}],
    }


def _build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("classify", classify_node)
    g.add_node("process", process_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "classify")
    g.add_edge("classify", "process")
    g.add_edge("process", "respond")
    g.add_edge("respond", END)
    return g.compile()


_pipeline = _build_pipeline()


def run_pipeline(question: str) -> dict:
    """Classify -> process -> respond pipeline."""
    start = time.time()
    result = _pipeline.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "category": "",
            "context": "",
            "answer": "",
        }
    )
    elapsed = time.time() - start
    answer = result.get("answer", "")
    # The pipeline uses the lookup tool directly in the process node
    tool_calls = 1 if result.get("context", "") else 0
    token_est = len(answer.split())
    return {"answer": answer, "latency": elapsed, "tool_calls": tool_calls, "tokens_est": token_est}


# ── Scoring ───────────────────────────────────────────────────────
#
# A comparison is only as trustworthy as the yardstick, and a substring check is
# a bad yardstick: "the capital is Paris" and "I could not find that" both score
# 0.0 against the keyword "Paris" if the model phrases it differently, and any
# answer that happens to contain the word scores 1.0 however wrong the rest is.
#
# Correctness is therefore judged by a REGISTERED judge (L2-M2.2), not a local
# function. That matters for comparison specifically: a registered judge is a
# named, versioned server-side object, so "react_agent scored 0.83" stays
# meaningful outside the script that produced it, and L2-M2.6 can put the very
# same judge on production traffic.
JUDGE_NAME = "answer_correctness"

CORRECTNESS_INSTRUCTIONS = """\
You are grading an answer from an AI agent.

The question is in {{ inputs }}.
The agent's answer is in {{ outputs }}.
{{ expectations }} contains `expected_keyword` -- the fact the answer must convey.

Does the answer correctly convey that fact? Judge the meaning, not the wording:
an answer that expresses the fact in different words is correct, and an answer
that merely contains the word while saying something wrong is not.

Answer true or false."""


def get_or_register_judge() -> Any:
    """Fetch the shared correctness judge, registering it the first time.

    This is the reuse pattern: later lessons and later runs get the SAME judge by
    name rather than a fresh anonymous one, which is what makes scores comparable
    across runs. The lesson still works standalone -- if the judge is not on the
    server yet, it is created and registered here.
    """
    try:
        judge = mlflow.genai.get_scorer(name=JUDGE_NAME)
        print(f"  reusing registered judge '{JUDGE_NAME}'")
        return judge
    except Exception:
        judge = mlflow.genai.make_judge(
            name=JUDGE_NAME,
            instructions=CORRECTNESS_INSTRUCTIONS,
            model=f"openai:/{MODEL_ALIAS}",
            feedback_value_type=bool,
        )
        registered = judge.register(name=JUDGE_NAME)
        print(f"  registered judge '{JUDGE_NAME}' (first run)")
        return registered


@scorer
def tool_usage(outputs: dict, expectations: dict) -> Feedback:
    """Deterministic, so it stays a local @scorer rather than a registered judge.

    Rewarding "used a tool when one was needed" needs no model, and a @scorer
    cannot be registered against an open-source server anyway (see L2-M2.2).
    """
    calls = int(outputs.get("tool_calls", 0))
    needs = bool(expectations.get("needs_tool", False))
    if needs:
        value = 1.0 if calls > 0 else 0.0
        why = "used a tool as required" if calls else "needed a tool and used none"
    else:
        value = 1.0 if calls == 0 else 0.5
        why = "no tool needed, none used" if calls == 0 else "called a tool unnecessarily"
    return Feedback(value=value, rationale=why, source=AssessmentSource(source_type="CODE", source_id="tool_usage"))


def as_score(value: Any) -> float:
    """Normalise a scorer's value out of `result_df` to a float.

    Two traps, both of which silently produce 0.0 for a correct answer:

    1. `result_df` is a pandas frame, so a boolean judge verdict arrives as
       `np.True_` -- a numpy bool, which is NOT a Python `bool` and, under
       numpy 2.x, not an `int` either. Every isinstance check misses it and the
       value falls through to the default. `.item()` unwraps numpy scalars.
    2. Judges answer `true`/`false` or `"yes"`/`"no"` depending on the judge, so
       strings need comparing against a whitelist rather than truth-testing
       (`bool("no")` is True).
    """
    if value is None:
        return 0.0
    if hasattr(value, "item"):  # numpy scalar -> plain Python scalar
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "yes", "pass"} else 0.0
    return 0.0


# ── Main evaluation loop ──────────────────────────────────────────
ARCHITECTURES: list[tuple[str, Callable[[str], dict]]] = [
    ("simple_chain", run_simple_chain),
    ("react_agent", run_react_agent),
    ("multi_step_pipeline", run_pipeline),
]


def evaluate_architecture(name: str, run_fn: Callable[[str], dict], dataset: list[dict], judge: Any) -> list[dict]:
    """Run an architecture on every case, then score the batch with genai.evaluate.

    Latency and token counts come from running the architecture; correctness and
    tool usage come from the evaluation harness. Scoring the whole batch in one
    `mlflow.genai.evaluate()` call -- rather than scoring case by case in Python --
    is what puts the results in MLflow as a comparable evaluation run.
    """
    results = [run_fn(case["question"]) for case in dataset]
    data = pd.DataFrame(
        [
            {
                "inputs": {"question": case["question"]},
                "outputs": {"answer": result["answer"], "tool_calls": result["tool_calls"]},
                "expectations": {
                    "expected_keyword": case["expected_keyword"],
                    "needs_tool": case["needs_tool"],
                },
            }
            for case, result in zip(dataset, results)
        ]
    )
    evaluation = mlflow.genai.evaluate(data=data, scorers=[judge, tool_usage])
    scored = evaluation.result_df

    rows = []
    for i, (case, result) in enumerate(zip(dataset, results)):
        correctness, tool_score = 0.0, 0.0
        if scored is not None and i < len(scored):
            correctness = as_score(scored.iloc[i].get(f"{JUDGE_NAME}/value"))
            tool_score = as_score(scored.iloc[i].get("tool_usage/value"))
        rows.append(
            {
                "architecture": name,
                "case": i + 1,
                "question": case["question"],
                "answer": result["answer"][:120],
                "correctness": correctness,
                "tool_usage": tool_score,
                "latency_s": round(result["latency"], 3),
                "tokens_est": result["tokens_est"],
            }
        )
    return rows


def main() -> None:
    print("=" * 70)
    print("  L2-M2.4 — Agent Architecture Comparison")
    print("  Three architectures, one dataset, one registered judge")
    print("=" * 70)

    all_rows: list[dict] = []
    judge = get_or_register_judge()

    with mlflow.start_run(run_name="architecture_comparison") as parent:
        mlflow.set_tags(
            {
                "comparison_type": "architecture",
                "num_architectures": str(len(ARCHITECTURES)),
                "num_test_cases": str(len(EVAL_DATASET)),
                "model": MODEL_ALIAS,
            }
        )

        for arch_name, arch_fn in ARCHITECTURES:
            print(f"\n{'─' * 70}")
            print(f"  Evaluating: {arch_name}")
            print(f"{'─' * 70}")

            with mlflow.start_run(run_name=arch_name, nested=True):
                rows = evaluate_architecture(arch_name, arch_fn, EVAL_DATASET, judge)
                all_rows.extend(rows)

                # Compute aggregate metrics for this architecture
                avg_correct = sum(r["correctness"] for r in rows) / len(rows)
                avg_tool = sum(r["tool_usage"] for r in rows) / len(rows)
                avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
                total_tokens = sum(r["tokens_est"] for r in rows)
                avg_tokens = total_tokens / len(rows)

                mlflow.log_params(
                    {
                        "architecture": arch_name,
                        "model": MODEL_ALIAS,
                        "num_cases": len(rows),
                    }
                )
                mlflow.log_metrics(
                    {
                        "avg_correctness": round(avg_correct, 3),
                        "avg_tool_usage": round(avg_tool, 3),
                        "avg_latency_s": round(avg_latency, 3),
                        "total_tokens_est": total_tokens,
                        "avg_tokens_est": round(avg_tokens, 1),
                    }
                )

                # Print per-case results
                for r in rows:
                    status = "PASS" if r["correctness"] == 1.0 else "FAIL"
                    print(f"  [{status}] Q{r['case']}: {r['question'][:50]}")
                    print(f"         Answer: {r['answer'][:80]}")
                    print(
                        f"         Correctness={r['correctness']:.0f}  "
                        f"ToolUsage={r['tool_usage']:.1f}  "
                        f"Latency={r['latency_s']:.2f}s"
                    )

        # ── Build comparison table ────────────────────────────────
        df = pd.DataFrame(all_rows)
        summary = (
            df.groupby("architecture")
            .agg(
                correctness=("correctness", "mean"),
                tool_usage=("tool_usage", "mean"),
                latency_s=("latency_s", "mean"),
                tokens_est=("tokens_est", "sum"),
            )
            .round(3)
        )

        # Add composite score: quality (equal weight correctness + tool_usage)
        summary["quality"] = ((summary["correctness"] + summary["tool_usage"]) / 2).round(3)

        # Token efficiency = quality / tokens (higher is better)
        summary["token_efficiency"] = (
            summary["quality"] / cast(pd.Series, summary["tokens_est"]).clip(lower=1) * 100
        ).round(3)

        print(f"\n{'=' * 70}")
        print("  COMPARISON TABLE: Architecture x Metric")
        print(f"{'=' * 70}")
        print()
        header = (
            f"  {'Architecture':<22} {'Correct':>8} {'ToolUse':>8} "
            f"{'Latency':>8} {'Tokens':>7} {'Quality':>8} {'Efficiency':>10}"
        )
        print(header)
        print("  " + "-" * 74)
        for arch, row in summary.iterrows():
            print(
                f"  {arch:<22} {row['correctness']:>8.3f} {row['tool_usage']:>8.3f} "
                f"{row['latency_s']:>7.2f}s {int(row['tokens_est']):>7} "
                f"{row['quality']:>8.3f} {row['token_efficiency']:>10.3f}"
            )

        # Log comparison artifact
        summary_path = "comparison_table.csv"
        summary.to_csv(summary_path)
        mlflow.log_artifact(summary_path)

        # Log parent-level summary
        best_quality = cast(pd.Series, summary["quality"]).idxmax()
        fastest = cast(pd.Series, summary["latency_s"]).idxmin()
        most_efficient = cast(pd.Series, summary["token_efficiency"]).idxmax()

        mlflow.log_params(
            {
                "best_quality_arch": best_quality,
                "fastest_arch": fastest,
                "most_efficient_arch": most_efficient,
            }
        )

        # ── Cost-quality tradeoff analysis ────────────────────────
        print(f"\n{'=' * 70}")
        print("  COST-QUALITY TRADEOFF ANALYSIS")
        print(f"{'=' * 70}")
        print(f"\n  Best quality:       {best_quality} (score={summary.loc[best_quality, 'quality']:.3f})")
        print(f"  Fastest:            {fastest} (latency={summary.loc[fastest, 'latency_s']:.3f}s)")
        print(
            f"  Most efficient:     {most_efficient} (efficiency={summary.loc[most_efficient, 'token_efficiency']:.3f})"
        )

        # Pareto frontier: architectures not dominated on both quality and latency
        print("\n  Pareto Frontier (quality vs latency):")
        pareto = []
        for arch in summary.index:
            dominated = False
            for other in summary.index:
                if other == arch:
                    continue
                if (
                    summary.loc[other, "quality"] >= summary.loc[arch, "quality"]
                    and summary.loc[other, "latency_s"] <= summary.loc[arch, "latency_s"]
                    and (
                        summary.loc[other, "quality"] > summary.loc[arch, "quality"]
                        or summary.loc[other, "latency_s"] < summary.loc[arch, "latency_s"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(arch)
                print(
                    f"    * {arch}: quality={summary.loc[arch, 'quality']:.3f}, "
                    f"latency={summary.loc[arch, 'latency_s']:.3f}s"
                )

        mlflow.set_tag("pareto_frontier", ", ".join(pareto))

        print(f"\n  Parent Run ID: {parent.info.run_id}")
        print("  View in MLflow UI: http://127.0.0.1:5555")

    print(f"\n{'=' * 70}")
    print("  Done! Check nested runs in the MLflow UI for full details.")
    print(f"{'=' * 70}")

    # Clean up temp file
    import os

    if os.path.exists(summary_path):
        os.remove(summary_path)


if __name__ == "__main__":
    main()
