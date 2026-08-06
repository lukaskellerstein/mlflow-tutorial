"""
L2-M2.1.3 — Agent Quality Metrics and Session Scorers

Comprehensive quality metrics for AI agents: task completion (binary + partial),
tool selection accuracy (precision/recall/F1), reasoning quality (LLM judge),
response quality (composite). Evaluates a LangGraph agent with three tools,
then statistically compares two agent configurations.

Part 2 adds the dimension none of those four can express. Every scorer above
judges ONE turn -- a question and its answer. Agents are multi-turn, and the
interesting failures are conversational: the agent that answers each question
adequately while never actually completing the task, or that forgets on turn 4
what the user said on turn 1. MLflow ships session-level scorers for exactly
that, and they take a whole session rather than an (input, output) pair.
"""

import json
import os
import re
import uuid
from typing import Any

import mlflow
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mlflow.entities import AssessmentSource, Feedback, Trace
from mlflow.genai.scorers import (
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    KnowledgeRetention,
    UserFrustration,
    scorer,
)
from pydantic import SecretStr

# The LiteLLM gateway from infra/, not a provider directly. "gemma-agent" is an
# alias defined in infra/litellm/config.yaml -- swapping model or provider is a
# change there, never here. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-agent"
# The agent and the thing grading it are named separately on purpose: both
# resolve to the same model today, but a judge and an agent are different
# jobs and will not always want the same one. Splitting them here means that
# change is a config edit, not a re-read of this lesson.
JUDGE_ALIAS = "gemma-judge"

# Session-level scorers judge a whole conversation, so they need their model
# resolved through LiteLLM the same way judges do in L2-M2.1.2.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
SESSION_MODEL = f"openai:/{JUDGE_ALIAS}"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
# Experiment name left exactly as it was, even though the lesson is now numbered
# 3 rather than 2. Renaming experiment strings is a separate decision from
# renumbering lessons -- a rename starts a fresh experiment and leaves whatever
# was logged under the old name stranded beside it.
mlflow.set_experiment("L2/M2_agent_evaluation/1_instruments/3_quality_metrics")

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "by",
    "it",
    "that",
    "this",
    "for",
    "with",
    "as",
    "at",
    "from",
}

# ── Tools ──────────────────────────────────────────────────────────────────── #


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

# ── LangGraph Agent ────────────────────────────────────────────────────────── #


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
    """Run agent, extract final answer and tool calls."""
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


# ── Evaluation Dataset ─────────────────────────────────────────────────────── #

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

# ── Custom Scorers ─────────────────────────────────────────────────────────── #


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


@scorer
def tool_selection_scorer(outputs: dict, expectations: dict) -> Feedback:
    """Precision/recall/F1 of tool choices vs expected tools."""
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


JUDGE_PROMPT = """\
Score the RESPONSE to the QUESTION from 0.0 to 1.0 on reasoning quality.
1.0=clear logical reasoning, 0.7=good with minor gaps, 0.4=some relevance, 0.0=empty/incoherent.
QUESTION: {question}
RESPONSE: {response}
Return ONLY JSON: {{"score": <float>, "rationale": "<one sentence>"}}"""


@scorer
def reasoning_quality_scorer(inputs: dict, outputs: dict) -> Feedback:
    """LLM judge for reasoning coherence."""
    answer = str(outputs.get("answer", "")) if isinstance(outputs, dict) else str(outputs)
    question = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
    raw = ChatOpenAI(
        model=MODEL_ALIAS,
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
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gemma-agent"),
    )


@scorer
def response_quality_scorer(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Composite: length adequacy (0.3) + structure (0.3) + relevance (0.4)."""
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


ALL_SCORERS = [
    task_completion_scorer,
    tool_selection_scorer,
    reasoning_quality_scorer,
    response_quality_scorer,
]

# ── Evaluation + Reporting ─────────────────────────────────────────────────── #


def run_evaluation(agent: Any) -> dict[str, Any]:
    """Run agent on all test cases and evaluate with mlflow.genai.evaluate()."""
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
    data = pd.DataFrame(rows)
    print(f"  Evaluating with {len(ALL_SCORERS)} scorers...")
    return {"results": mlflow.genai.evaluate(data=data, scorers=ALL_SCORERS), "data": data}


def print_report(results: Any, label: str) -> dict[str, float]:
    """Print quality report and return metrics dict."""
    print(f"\n{'=' * 70}\n  Quality Report: {label}\n{'=' * 70}")
    metrics = results.metrics
    print("\n  Aggregate Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.3f}")

    df = results.result_df
    if df is not None:
        scorer_names = [s.name for s in ALL_SCORERS]
        cols = [f"{s}/value" for s in scorer_names if f"{s}/value" in df.columns]
        hdr = f"  {'Case':<45s}" + "".join(
            f" {c.replace('_scorer/value', '').replace('/value', ''):>14s}" for c in cols
        )
        print(f"\n  Per-Case Results:\n{hdr}")
        print(f"  {'-' * 45}" + (" " + "-" * 14) * len(cols))
        for i, row in df.iterrows():
            q = EVAL_CASES[i]["query"][:43] if i < len(EVAL_CASES) else "?"
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


# ── Part 2: session-level scoring ──────────────────────────────────────────── #

# A scripted four-turn conversation. Turn 4 deliberately refers back to turn 1
# without restating it -- that is the only way to see whether the agent retained
# anything, and no single-turn scorer can ask the question.
CONVERSATION = [
    "I need the total cost of 12 items at 4.75 each.",
    "What does the word 'amortize' mean?",
    "Format that total in title case for my report, please.",
    "Remind me what the total was again?",
]


def session_scorer(scorer_cls: Any, name: str) -> Any:
    """Build a built-in session scorer bound to the gateway model.

    `name` is the metric it logs under, so it is set explicitly rather than left
    to a default. `model` is a genuine pydantic field on these classes -- the
    attribute really does hold the value afterwards, and omitting it leaves the
    judge on MLflow's default model instead of the gateway's -- but basedpyright
    resolves a narrower __init__ than pydantic synthesises and reports
    "No parameter named model". Constructing through this factory keeps the
    checker useful on the rest of the file without four inline suppressions.
    """
    return scorer_cls(name=name, model=SESSION_MODEL)


SESSION_SCORERS = [
    session_scorer(ConversationCompleteness, "conversation_completeness"),
    session_scorer(UserFrustration, "user_frustration"),
    session_scorer(KnowledgeRetention, "knowledge_retention"),
    session_scorer(ConversationalToolCallEfficiency, "conversational_tool_call_efficiency"),
]


@mlflow.trace(name="conversation_turn")
def run_turn(agent: Any, history: list[Any], session_id: str) -> dict:
    """One turn, wrapped so we own the root span and can stamp the session id.

    The agent's own spans nest underneath this one, so each turn is a single
    trace that carries the session id -- which is what makes the turns
    groupable into a session later.
    """
    mlflow.update_current_trace(session_id=session_id)
    return agent.invoke({"messages": history})


def run_session(agent: Any) -> list[Trace]:
    """Drive one multi-turn conversation and collect a trace per turn.

    A session-level scorer takes `session=list[Trace]` -- the ordered turns of
    ONE conversation -- and it validates that every trace carries a session_id.
    Collecting traces is not enough: without the id you get
    "All traces in 'session' must have a session_id".
    """
    session_id = f"quality-metrics-{uuid.uuid4().hex[:8]}"
    trace_ids: list[str] = []
    history: list[Any] = []
    for turn in CONVERSATION:
        history.append({"role": "user", "content": turn})
        result = run_turn(agent, history, session_id)
        history = result["messages"]
        if trace_id := mlflow.get_last_active_trace_id():
            trace_ids.append(trace_id)

    # Traces are exported asynchronously. Fetching one straight after the turn
    # that produced it can race the exporter, so flush before reading them back.
    mlflow.flush_trace_async_logging()
    return [trace for tid in trace_ids if (trace := mlflow.get_trace(tid))]


TRUTHY = {"yes", "true", "pass", "y"}


def to_metric(value: Any) -> float:
    """Normalise a session judge's answer to a number MLflow can log.

    These judges answer with a STRING as often as with a number -- one run of
    this lesson returns "yes", "none" and "no" from three different scorers --
    and `bool("no")` is True, so coercing with bool() would score every failure
    as a pass. Strings are compared against a whitelist instead.

    Note the polarity is NOT uniform across scorers: 1.0 is the good outcome for
    conversation_completeness, knowledge_retention and tool_call_efficiency, but
    user_frustration is good at 0.0. Each is logged under its own metric name for
    that reason -- summing them would add a metric that improves as it falls to
    three that improve as they rise.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in TRUTHY else 0.0
    return 0.0


def score_session(traces: list[Trace]) -> dict[str, float]:
    """Run every session-level scorer over the whole conversation."""
    scores: dict[str, float] = {}
    for session_scorer in SESSION_SCORERS:
        feedback = session_scorer(session=traces)
        value = getattr(feedback, "value", feedback)
        rationale = " ".join(str(getattr(feedback, "rationale", "") or "").split())
        scores[session_scorer.name] = to_metric(value)
        print(f"    {session_scorer.name:<34} {str(value):<8} {rationale[:60]}")
    return scores


# ── Main ───────────────────────────────────────────────────────────────────── #


def main() -> None:
    print("=" * 70)
    print("  L2-M2.1.3 — Agent Quality Metrics and Session Scorers")
    print("  4 custom single-turn scorers, then 4 session-level scorers")
    print("=" * 70)

    # Configuration A: temperature=0.3 (more deterministic)
    print(f"\n{'=' * 70}\n  Configuration A: temperature=0.3\n{'=' * 70}")
    agent_a = build_agent(temperature=0.3)
    with mlflow.start_run(run_name="config_a_temp_0.3"):
        mlflow.log_params(
            {
                "temperature": 0.3,
                "model": "gemma-agent",
                "num_test_cases": len(EVAL_CASES),
            }
        )
        eval_a = run_evaluation(agent_a)
    ma = print_report(eval_a["results"], "Config A (temp=0.3)")

    # Configuration B: temperature=0.9 (more creative)
    print(f"\n{'=' * 70}\n  Configuration B: temperature=0.9\n{'=' * 70}")
    agent_b = build_agent(temperature=0.9)
    with mlflow.start_run(run_name="config_b_temp_0.9"):
        mlflow.log_params(
            {
                "temperature": 0.9,
                "model": "gemma-agent",
                "num_test_cases": len(EVAL_CASES),
            }
        )
        eval_b = run_evaluation(agent_b)
    mb = print_report(eval_b["results"], "Config B (temp=0.9)")

    compare_configs(ma, mb, "temp=0.3", "temp=0.9")

    # ── Part 2: the dimension the four scorers above cannot reach ───────────── #
    print(f"\n{'=' * 70}\n  Part 2: session-level scorers (multi-turn)\n{'=' * 70}")
    print(f"  driving a {len(CONVERSATION)}-turn conversation through the agent...")
    session_traces = run_session(agent_a)
    print(f"  captured {len(session_traces)} traces, one per turn\n")
    print(f"    {'scorer':<34} {'value':<8} rationale")
    print(f"    {'-' * 34} {'-' * 8} {'-' * 60}")
    with mlflow.start_run(run_name="session_level_scoring"):
        mlflow.log_params({"turns": len(CONVERSATION), "session_model": SESSION_MODEL})
        session_scores = score_session(session_traces)
        mlflow.log_metrics({f"session/{k}": v for k, v in session_scores.items()})

    print("\n  Every scorer in Part 1 sees one question and one answer. These four")
    print("  see the whole conversation -- which is the only way to ask whether the")
    print("  task was ever finished, or whether turn 4 remembered turn 1.")

    print(f"\n{'=' * 70}")
    print("  Done! View results: http://127.0.0.1:5555")
    print("  Experiment: L2/M2_agent_evaluation/1_instruments/3_quality_metrics")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
