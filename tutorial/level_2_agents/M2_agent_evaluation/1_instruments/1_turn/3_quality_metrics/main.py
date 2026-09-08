"""L2-M2.1.1.3 -- Turn-Level Quality Metrics.

Everything here judges ONE turn: one request, one final answer. That is a
boundary in time, not blindness to the steps. A turn-level scorer can see
everything the agent did inside that turn -- it simply cannot see the next turn.

Two scopes live in this file, and the difference matters:

  A  outside the turn -- the question and the final answer only
     task_completion, reasoning_quality, response_quality

  B  inside the turn  -- every tool call, its arguments, the whole trajectory
     tool_selection (from a flattened outputs dict),
     ToolCallCorrectness and ToolCallEfficiency (straight from the trace)

Scope C -- the whole conversation -- is 2_conversation/3_judging_conversations.

One naming rule worth carrying out of here: a JUDGE is a scorer that decides
with an LLM. The practical test is whether it takes a `model=` parameter.
`tool_selection_scorer` computes and is not a judge; `ToolCallCorrectness`
decides and is.
"""

import json
import os
import re
from typing import Any

import mlflow
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency, scorer
from pydantic import SecretStr

# The MLflow AI Gateway -- the tracking server itself, not a provider directly. "gemma-agent" is an
# alias defined in infra/mlflow/gateway/seed_gateway.py -- swapping model or provider is a
# change there, never here.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
# The agent and the thing grading it are named separately on purpose: both
# resolve to the same model today, but a judge and an agent are different jobs.
JUDGE_ALIAS = "gemma-judge"

# The built-in scorers resolve their model through the litellm LIBRARY (which MLflow
# uses internally -- not the proxy this repo used to run), and it reads these.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
BUILTIN_MODEL = f"openai:/{JUDGE_ALIAS}"

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/1_turn/3_quality_metrics"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in",
    "on", "by", "it", "that", "this", "for", "with", "as", "at", "from",
}  # fmt: skip

# -- Tools ------------------------------------------------------------------ #


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def dictionary_lookup(word: str) -> str:
    """Look up the definition of a word."""
    defs = {
        "photosynthesis": "The process by which green plants convert sunlight into chemical energy.",
        "algorithm": "A step-by-step procedure for solving a problem or accomplishing a task.",
        "entropy": "A measure of disorder or randomness in a system.",
        "recursion": "A method where the solution depends on smaller instances of the same problem.",
        "catalyst": "A substance that increases the rate of a chemical reaction without being consumed.",
        "mitosis": "Cell division producing two daughter cells with the same chromosome number.",
    }
    return defs.get(word.lower(), f"No definition found for '{word}'.")


@tool
def text_formatter(text: str, style: str) -> str:
    """Format text: 'uppercase', 'lowercase', 'title', or 'reverse'."""
    styles = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "title": text.title(),
        "reverse": text[::-1],
    }
    return styles.get(style.lower(), f"Unknown style '{style}'.")


TOOLS = [calculator, dictionary_lookup, text_formatter]

# -- LangGraph Agent -------------------------------------------------------- #


def build_agent(temperature: float = 0.7) -> Any:
    """Build a ReAct-style agent with the three tools."""
    llm = ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=temperature,
    ).bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(agent: Any, query: str) -> dict[str, Any]:
    """Run agent, flatten the trajectory into a dict a scope-A scorer can read."""
    msgs = agent.invoke({"messages": [{"role": "user", "content": query}]})["messages"]
    tools_used = [tc["name"] for m in msgs if getattr(m, "type", "") == "ai" and m.tool_calls for tc in m.tool_calls]
    answer = next(
        (
            m.content
            for m in reversed(msgs)
            if getattr(m, "type", "") == "ai" and m.content and not getattr(m, "tool_calls", None)
        ),
        "",
    )
    return {"answer": answer, "tools_used": tools_used, "num_messages": len(msgs)}


@mlflow.trace(name="agent_turn")
def run_agent_traced(agent: Any, query: str) -> dict[str, Any]:
    """Same call, wrapped so we own the root span and get ONE trace per turn.

    The trace is what a scope-B built-in scorer reads. It carries every tool
    call and its arguments, which the flattened dict above throws away.
    """
    return run_agent(agent, query)


# -- Evaluation Dataset ----------------------------------------------------- #

EVAL_CASES = [
    {
        "query": "What is 125 * 8?",
        "expected_output": "1000",
        "expected_tools": ["calculator"],
        "category": "single_tool",
    },
    {
        "query": "Define the word 'entropy'.",
        "expected_output": "A measure of disorder or randomness in a system.",
        "expected_tools": ["dictionary_lookup"],
        "category": "single_tool",
    },
    {
        "query": "Format the text 'hello world' in uppercase.",
        "expected_output": "HELLO WORLD",
        "expected_tools": ["text_formatter"],
        "category": "single_tool",
    },
    {
        "query": "What is 15 + 27, and also define 'algorithm'?",
        "expected_output": "42. An algorithm is a step-by-step procedure for solving a problem.",
        "expected_tools": ["calculator", "dictionary_lookup"],
        "category": "multi_tool",
    },
    {
        "query": "What is the meaning of life?",
        "expected_output": "A philosophical question with no single definitive answer.",
        "expected_tools": [],
        "category": "no_tool",
    },
    {
        "query": "Calculate (10 + 5) * 3, then format the result in title case.",
        "expected_output": "45",
        "expected_tools": ["calculator", "text_formatter"],
        "category": "multi_tool",
    },
]

# -- Scope A: the question and the final answer ----------------------------- #


@scorer
def task_completion_scorer(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Binary + partial credit: keyword overlap with expected output."""
    answer = str(outputs.get("answer", "")) if isinstance(outputs, dict) else str(outputs)
    expected = str(expectations.get("expected_output", ""))
    if not answer.strip():
        return Feedback(
            value=0.0,
            rationale="Empty answer.",
            source=AssessmentSource(source_type="CODE", source_id="task_completion"),
        )

    exp_kw = set(re.findall(r"\w+", expected.lower())) - STOP_WORDS
    ans_kw = set(re.findall(r"\w+", answer.lower()))
    if not exp_kw:
        score = 1.0 if len(ans_kw) > 3 else 0.5
        overlap = 0.0
    else:
        overlap = len(exp_kw & ans_kw) / len(exp_kw)
        score = 1.0 if overlap >= 0.6 else (0.5 if overlap >= 0.3 else 0.0)
    return Feedback(
        value=score,
        rationale=f"Keyword overlap={overlap:.2f}, score={score}",
        source=AssessmentSource(source_type="CODE", source_id="task_completion"),
    )


JUDGE_PROMPT = """\
Score the RESPONSE to the QUESTION from 0.0 to 1.0 on reasoning quality.
1.0=clear logical reasoning, 0.7=good with minor gaps, 0.4=some relevance, 0.0=empty/incoherent.
QUESTION: {question}
RESPONSE: {response}
Return ONLY JSON: {{"score": <float>, "rationale": "<one sentence>"}}"""


@scorer
def reasoning_quality_scorer(inputs: dict, outputs: dict) -> Feedback:
    """An INLINE judge -- it takes a model, so it is a judge, not just a scorer."""
    answer = str(outputs.get("answer", "")) if isinstance(outputs, dict) else str(outputs)
    question = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
    raw = ChatOpenAI(
        model=JUDGE_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    ).invoke(JUDGE_PROMPT.format(question=question, response=answer))
    raw = str(raw.content).strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        try:
            scores = json.loads(m.group()) if m else {"score": 0.5}
        except json.JSONDecodeError:
            scores = {"score": 0.5, "rationale": "Parse error"}
    return Feedback(
        value=float(scores.get("score", 0.5)),
        rationale=str(scores.get("rationale", "")),
        source=AssessmentSource(source_type="LLM_JUDGE", source_id=JUDGE_ALIAS),
    )


@scorer
def response_quality_scorer(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Composite: length adequacy (0.3) + structure (0.3) + relevance (0.4).

    The weights are here, in code, rather than buried in a judge's prose. That
    is the whole point of a composite -- your definition of "good" is readable
    and tunable by whoever owns the metric.
    """
    answer = str(outputs.get("answer", "")) if isinstance(outputs, dict) else str(outputs)
    expected = str(expectations.get("expected_output", ""))
    wc = len(answer.split())
    length_s = 0.0 if wc == 0 else (0.3 if wc < 5 else (1.0 if wc <= 100 else 0.7))
    sents = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    struct_s = min(len(sents) / 2.0, 1.0)
    exp_kw = set(re.findall(r"\w+", expected.lower())) - STOP_WORDS
    ans_kw = set(re.findall(r"\w+", answer.lower()))
    rel_s = len(exp_kw & ans_kw) / max(len(exp_kw), 1)
    comp = round(0.3 * length_s + 0.3 * struct_s + 0.4 * rel_s, 3)
    return Feedback(
        value=comp,
        rationale=f"len={length_s:.2f}({wc}w), struct={struct_s:.2f}, rel={rel_s:.2f}",
        source=AssessmentSource(source_type="CODE", source_id="response_quality"),
    )


# -- Scope B: inside the turn ------------------------------------------------ #


@scorer
def tool_selection_scorer(outputs: dict, expectations: dict) -> Feedback:
    """Precision/recall/F1 of tool choices, from the FLATTENED outputs dict.

    This sees the steps only because run_agent() put `tools_used` in the dict.
    It never sees the arguments, and it cannot tell one call from three
    identical ones -- which is exactly what the trace-based scorers below fix.
    """
    used = set(outputs.get("tools_used", []) if isinstance(outputs, dict) else [])
    expected = set(expectations.get("expected_tools", []))
    src = AssessmentSource(source_type="CODE", source_id="tool_selection")

    if not expected and not used:
        return Feedback(value=1.0, rationale="No tools expected or used.", source=src)
    if not expected or not used:
        return Feedback(value=0.0, rationale=f"Used={list(used)}, expected={list(expected)}.", source=src)
    prec = len(used & expected) / len(used)
    rec = len(used & expected) / len(expected)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return Feedback(value=round(f1, 3), rationale=f"P={prec:.2f}, R={rec:.2f}, F1={f1:.3f}", source=src)


CUSTOM_SCORERS = [
    task_completion_scorer,
    tool_selection_scorer,
    reasoning_quality_scorer,
    response_quality_scorer,
]


# Both take `trace`, and both take `model=` -- so both are judges that read the
# whole trajectory of one turn. ToolCallCorrectness runs ground-truth-free by
# default: with no expectations it asks whether the calls were reasonable.
def build(scorer_cls: Any, name: str) -> Any:
    """Construct a built-in scorer bound to the gateway model.

    `model` is a genuine pydantic field on these classes -- the attribute really
    does hold the value afterwards -- but basedpyright resolves a narrower
    __init__ than pydantic synthesises and reports "No parameter named model".
    Building through this factory keeps the checker useful on the rest of the
    file without inline suppressions. `name` is set explicitly because it is the
    metric the scorer logs under.
    """
    return scorer_cls(name=name, model=BUILTIN_MODEL)


TRACE_SCORERS = [
    build(ToolCallCorrectness, "tool_call_correctness"),
    build(ToolCallEfficiency, "tool_call_efficiency"),
]

# -- Evaluation + Reporting -------------------------------------------------- #


def run_evaluation(agent: Any) -> Any:
    """Run agent on all test cases and evaluate with the four custom scorers."""
    print(f"\n  Running agent on {len(EVAL_CASES)} test cases...")
    rows = [
        {
            "inputs": {"query": c["query"]},
            "outputs": run_agent(agent, c["query"]),
            "expectations": {
                "expected_output": c["expected_output"],
                "expected_tools": c["expected_tools"],
            },
        }
        for c in EVAL_CASES
    ]
    print(f"  Evaluating with {len(CUSTOM_SCORERS)} scorers...")
    return mlflow.genai.evaluate(data=pd.DataFrame(rows), scorers=CUSTOM_SCORERS)


def print_report(results: Any, label: str) -> dict[str, float]:
    """Print quality report and return metrics dict."""
    print(f"\n{'=' * 70}\n  Quality Report: {label}\n{'=' * 70}")
    metrics = results.metrics
    print("\n  Aggregate Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.3f}")

    df = results.result_df
    if df is not None:
        cols = [f"{s.name}/value" for s in CUSTOM_SCORERS if f"{s.name}/value" in df.columns]
        hdr = f"  {'Case':<45s}" + "".join(f" {c.replace('_scorer/value', ''):>14s}" for c in cols)
        print(f"\n  Per-Case Results:\n{hdr}")
        print(f"  {'-' * 45}" + (" " + "-" * 14) * len(cols))
        for i, row in df.iterrows():
            q = EVAL_CASES[i]["query"][:43] if isinstance(i, int) and i < len(EVAL_CASES) else "?"
            vals = ""
            for c in cols:
                try:
                    vals += f" {float(row.get(c)):>14.3f}"
                except (ValueError, TypeError):
                    vals += f" {'N/A':>14s}"
            print(f"  {q:<45s}{vals}")
    return metrics


def compare_configs(ma: dict, mb: dict, la: str, lb: str) -> None:
    """Side-by-side comparison of two configurations."""
    print(f"\n{'=' * 70}\n  Comparison: {la} vs {lb}\n{'=' * 70}")
    means_a = {k: v for k, v in ma.items() if k.endswith("/mean")}
    means_b = {k: v for k, v in mb.items() if k.endswith("/mean")}
    common = sorted(set(means_a) & set(means_b))

    print(f"\n  {'Metric':<38s} {la:>10s} {lb:>10s} {'Delta':>10s} {'Winner':>10s}")
    print(f"  {'-' * 38} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    wins = {la: 0, lb: 0, "TIE": 0}
    for m in common:
        a, b, d = means_a[m], means_b[m], means_b[m] - means_a[m]
        w = "TIE" if abs(d) < 0.01 else (lb if d > 0 else la)
        wins[w] += 1
        short = m.replace("_scorer/mean", "").replace("/mean", "")
        print(f"  {short:<38s} {a:>10.3f} {b:>10.3f} {d:>+10.3f} {w:>10s}")
    print(f"\n  {la} wins {wins[la]}, {lb} wins {wins[lb]}, ties {wins['TIE']}")
    ta, tb = sum(means_a.get(m, 0) for m in common), sum(means_b.get(m, 0) for m in common)
    winner = la if ta > tb else (lb if tb > ta else "TIE")
    print(f"  Overall: {winner} ({ta:.3f} vs {tb:.3f})")


def score_from_traces(agent: Any) -> None:
    """Scope B: hand the built-in judges the TRACE, not a flattened dict."""
    print(f"\n{'=' * 70}\n  Scope B: scorers that read the trace\n{'=' * 70}")
    print(f"  re-running {len(EVAL_CASES)} cases under an explicit root span...")

    traces = []
    for case in EVAL_CASES:
        run_agent_traced(agent, str(case["query"]))
        if trace_id := mlflow.get_last_active_trace_id():
            traces.append(trace_id)

    # Traces export asynchronously. Fetching one straight after the call that
    # produced it races the exporter, so flush before reading them back.
    mlflow.flush_trace_async_logging()
    data = pd.DataFrame({"trace": [t for tid in traces if (t := mlflow.get_trace(tid))]})
    print(f"  captured {len(data)} traces, evaluating with {len(TRACE_SCORERS)} built-in judges...")

    with mlflow.start_run(run_name="trace_based_scorers"):
        mlflow.log_params({"scorers": ",".join(s.name for s in TRACE_SCORERS), "judge_model": BUILTIN_MODEL})
        results = mlflow.genai.evaluate(data=data, scorers=TRACE_SCORERS)
        print("\n  Aggregate Metrics:")
        for k, v in sorted(results.metrics.items()):
            print(f"    {k}: {v:.3f}")

    print("\n  These two never saw `tools_used`. They read the spans, so they can")
    print("  see the ARGUMENTS a tool was called with, and whether the same call")
    print("  was made twice. The flattened dict throws both away.")
    print("\n  Read the harness WARNING above if there is one. A small local model")
    print("  sometimes fails a judge's structured-output parser, and MLflow drops")
    print("  that case from the mean rather than scoring it zero. A mean over 5 of")
    print("  6 cases is not the same number as a mean over 6 -- check the count")
    print("  before comparing two runs.")


def main() -> None:
    print("=" * 70)
    print("  L2-M2.1.1.3 -- Turn-Level Quality Metrics")
    print("  Scope A and B, over ONE request and ONE answer")
    print("=" * 70)

    print(f"\n{'=' * 70}\n  Configuration A: temperature=0.3\n{'=' * 70}")
    agent_a = build_agent(temperature=0.3)
    with mlflow.start_run(run_name="config_a_temp_0.3"):
        mlflow.log_params({"temperature": 0.3, "model": MODEL_ALIAS, "num_test_cases": len(EVAL_CASES)})
        eval_a = run_evaluation(agent_a)
    ma = print_report(eval_a, "Config A (temp=0.3)")

    print(f"\n{'=' * 70}\n  Configuration B: temperature=0.9\n{'=' * 70}")
    agent_b = build_agent(temperature=0.9)
    with mlflow.start_run(run_name="config_b_temp_0.9"):
        mlflow.log_params({"temperature": 0.9, "model": MODEL_ALIAS, "num_test_cases": len(EVAL_CASES)})
        eval_b = run_evaluation(agent_b)
    mb = print_report(eval_b, "Config B (temp=0.9)")

    compare_configs(ma, mb, "temp=0.3", "temp=0.9")

    score_from_traces(agent_a)

    print(f"\n{'=' * 70}")
    print("  Every scorer above judged ONE turn. None of them can ask whether the")
    print("  task was ever finished, or whether turn 4 remembered turn 1.")
    print("  That is 2_conversation/3_judging_conversations.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
