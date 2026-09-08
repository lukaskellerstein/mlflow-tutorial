"""L2-M2.1.2.3 -- Judging a Whole Conversation.

The two lessons before this one PRODUCE conversations and grade nothing. This is
the one that turns a conversation into a verdict, and it does so two ways:

  words   -- discover_issues() returns issues with a severity and a root cause
  numbers -- session-level scorers return one value for the whole conversation

Every scorer in 1_turn/ takes `(inputs, outputs)` -- one question, one answer.
These take `session=list[Trace]`, and that changes what can be asked. Three
failures only exist at conversation scope:

  * the agent answers every question adequately and never finishes the task
  * the agent forgets on turn 4 what the user told it on turn 1
  * the user gets visibly more frustrated with every reply

The scripted conversation below ends by referring back to turn 1 WITHOUT
restating it. That is the only way to see whether the agent retained anything.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Feedback, Trace
from mlflow.genai.scorers import (
    ConversationalGuidelines,
    ConversationalRoleAdherence,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    KnowledgeRetention,
    UserFrustration,
    scorer,
)
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
JUDGE_ALIAS = "gemma-judge"

# Session scorers resolve their model through the litellm LIBRARY (which MLflow
# uses internally -- not the proxy this repo used to run), and it reads these.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
SESSION_MODEL = f"openai:/{JUDGE_ALIAS}"

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/2_conversation/3_judging_conversations"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog()

ORDERS = {
    "A1001": "shipped, arriving Thursday",
    "A1002": "held at the warehouse, payment not confirmed",
    "A1003": "delivered on Monday",
}
POLICIES = {
    "returns": "30 days from delivery, receipt required (P-101).",
    "warranty": "12 months against manufacturing defects (P-204).",
    "shipping": "Return shipping is free for faulty items (P-330).",
}


@tool
def order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A1001."""
    return ORDERS.get(order_id.strip().upper(), f"No order found with id {order_id}.")


@tool
def policy_lookup(topic: str) -> str:
    """Look up store policy. Topics: returns, warranty, shipping."""
    return POLICIES.get(topic.strip().lower(), f"No policy on file for '{topic}'.")


SYSTEM_PROMPT = (
    "You are a retail support agent. Use the tools to answer questions about orders and policy. "
    "Never invent an order status or a policy -- look it up. Keep replies under three sentences."
)

llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
AGENT = create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)

# Turn 4 refers back to turn 1 without restating the delivery day. No single-turn
# scorer can ask whether that worked.
CONVERSATION = [
    "What is the status of order A1001?",
    "How long do I have to return something?",
    "And who pays the return shipping if the item is faulty?",
    "Remind me which day my order was arriving?",
]

AGENT_GUIDELINES = [
    "Always cite the policy reference code (for example P-101) when quoting a policy.",
    "Never invent an order status.",
]

# -- Session scorers --------------------------------------------------------- #


def build(scorer_cls: Any, name: str, **kwargs: Any) -> Any:
    """Construct a built-in session scorer bound to the gateway model.

    `model` is a genuine pydantic field on these classes, but basedpyright
    resolves a narrower __init__ than pydantic synthesises and reports
    "No parameter named model". Building through this factory keeps the checker
    useful on the rest of the file without seven inline suppressions.
    """
    return scorer_cls(name=name, model=SESSION_MODEL, **kwargs)


# All seven built-in session scorers MLflow ships. Each takes `model=`, so every
# one of them is a JUDGE -- a scorer that decides with an LLM.
SESSION_SCORERS = [
    build(ConversationCompleteness, "conversation_completeness"),
    build(UserFrustration, "user_frustration"),
    build(KnowledgeRetention, "knowledge_retention"),
    build(ConversationalToolCallEfficiency, "conversational_tool_call_efficiency"),
    build(ConversationalRoleAdherence, "conversational_role_adherence"),
    build(ConversationalSafety, "conversational_safety"),
    build(ConversationalGuidelines, "conversational_guidelines", guidelines=AGENT_GUIDELINES),
]


@scorer
def turn_count(session: list[Trace]) -> Feedback:
    """Your own session scorer -- no LLM anywhere in it.

    A `@scorer` function becomes session-level purely because its parameter is
    named `session`. MLflow checks for that name and sets
    `is_session_level_scorer` accordingly, so this is the whole opt-in.

    It is also the counter-example to "session scorer means LLM judge". This one
    counts, so it is a scorer and not a judge.
    """
    return Feedback(value=float(len(session)), rationale=f"{len(session)} turns in this session.")


CUSTOM_SESSION_SCORERS = [turn_count]

# -- Driving the conversation ------------------------------------------------ #


@mlflow.trace(name="conversation_turn")
def run_turn(history: list[Any], session_id: str) -> dict:
    """One turn, wrapped so we own the root span and can stamp the session id.

    The agent's own spans nest underneath this one, so each turn is a single
    trace that carries the session id -- which is what makes the turns
    groupable into a session later.
    """
    mlflow.update_current_trace(session_id=session_id)
    return AGENT.invoke({"messages": history})


def run_session() -> list[Trace]:
    """Drive one multi-turn conversation and collect a trace per turn.

    A session-level scorer takes `session=list[Trace]` -- the ordered turns of
    ONE conversation -- and it validates that every trace carries a session_id.
    Collecting traces is not enough: without the id you get
    "All traces in 'session' must have a session_id".
    """
    session_id = f"judging-{uuid.uuid4().hex[:8]}"
    trace_ids: list[str] = []
    history: list[Any] = []
    for i, turn in enumerate(CONVERSATION, 1):
        history.append({"role": "user", "content": turn})
        print(f"    turn {i} user  : {turn}")
        result = run_turn(history, session_id)
        history = result["messages"]
        reply = " ".join(str(getattr(history[-1], "content", "")).split())
        print(f"           agent : {reply[:70]}")
        if trace_id := mlflow.get_last_active_trace_id():
            trace_ids.append(trace_id)

    # Traces are exported asynchronously. Fetching one straight after the turn
    # that produced it can race the exporter, so flush before reading them back.
    mlflow.flush_trace_async_logging()
    return [trace for tid in trace_ids if (trace := mlflow.get_trace(tid))]


TRUTHY = {"yes", "true", "pass", "y"}


def to_metric(value: Any) -> float:
    """Normalise a session judge's answer to a number MLflow can log.

    These judges answer with a STRING as often as with a number -- one run
    returns "yes", "none" and "no" from three different scorers -- and
    `bool("no")` is True, so coercing with bool() would score every failure as a
    pass. Strings are compared against a whitelist instead.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in TRUTHY else 0.0
    return 0.0


def score_session(traces: list[Trace]) -> dict[str, float]:
    """Run every session scorer over the whole conversation."""
    scores: dict[str, float] = {}
    failed: list[str] = []
    print(f"\n    {'scorer':<36} {'value':<8} rationale")
    print(f"    {'-' * 36} {'-' * 8} {'-' * 58}")
    for session_scorer in SESSION_SCORERS + CUSTOM_SESSION_SCORERS:
        try:
            feedback = session_scorer(session=traces)
        except Exception as exc:
            # A judge that could not parse its OWN answer. This happens with a
            # small local model: the judge asks for structured output and gets
            # prose back. Note what is NOT done here -- no metric is logged.
            # A judge that failed to answer is not a score of zero, and
            # recording it as one would quietly drag the average down.
            failed.append(session_scorer.name)
            print(f"    {session_scorer.name:<36} {'ERROR':<8} {str(exc)[:58]}")
            continue
        value = getattr(feedback, "value", feedback)
        rationale = " ".join(str(getattr(feedback, "rationale", "") or "").split())
        scores[session_scorer.name] = to_metric(value)
        print(f"    {session_scorer.name:<36} {str(value):<8} {rationale[:58]}")

    if failed:
        print(f"\n    {len(failed)} judge(s) could not parse their own answer: {', '.join(failed)}")
        print("    No metric was logged for those. A judge that did not answer is")
        print("    NOT a zero -- scoring it as one would quietly lower the average.")
    return scores


def discover(traces: list[Trace]) -> None:
    """The other half of judging: issues in words, not numbers."""
    # Not in mlflow.genai.discovery's __all__ -- test_agent() is the supported
    # entry point and imports it exactly this way. Called directly here so you
    # meet stage 4 on its own before meeting the wrapper in 4_test_agent.
    from mlflow.genai.discovery.pipeline import discover_issues

    print(f"\n{'=' * 70}\n  Judging in WORDS: discover_issues()\n{'=' * 70}")
    result = discover_issues(traces=traces, model=SESSION_MODEL, max_issues=5)
    issues = list(getattr(result, "issues", None) or [])
    print(f"  {len(issues)} issues found")
    for issue in issues:
        print(f"\n    severity   : {getattr(issue, 'severity', '?')}")
        print(f"    categories : {getattr(issue, 'categories', '?')}")
        print(f"    summary    : {' '.join(str(getattr(issue, 'summary', '')).split())[:70]}")
    if not issues:
        print("\n  Zero issues is not proof the conversation was good. Discovery is")
        print("  itself an LLM judge, and a judge that finds nothing may simply have")
        print("  failed to look. Read the numbers below beside this, never instead.")


def main() -> None:
    print("=" * 70)
    print("  L2-M2.1.2.3 -- Judging a Whole Conversation")
    print("=" * 70)

    print(f"\nStep 1: drive a {len(CONVERSATION)}-turn conversation")
    print("  (turn 4 refers back to turn 1 without restating it)\n")
    traces = run_session()
    print(f"\n  captured {len(traces)} traces, one per turn, all sharing a session id")

    print(f"\n{'=' * 70}\n  Judging in NUMBERS: session-level scorers\n{'=' * 70}")
    with mlflow.start_run(run_name="session_level_scoring"):
        mlflow.log_params({"turns": len(CONVERSATION), "session_model": SESSION_MODEL})
        scores = score_session(traces)
        mlflow.log_metrics({f"session/{k}": v for k, v in scores.items()})

    print("\n  WARNING: these do not share a value vocabulary OR a polarity.")
    print("  `user_frustration` is GOOD at 0.0; the rest are good at 1.0. Never sum")
    print("  them into one session score -- you would be adding a metric that")
    print("  improves as it falls to six that improve as they rise. Each is logged")
    print("  separately under session/<name> for exactly this reason.")

    discover(traces)

    print(f"\n{'=' * 70}")
    print(f"  turns judged     : {len(traces)}")
    print(f"  session scorers  : {len(SESSION_SCORERS)} built-in + {len(CUSTOM_SESSION_SCORERS)} custom")
    print("\n  Next: 4_test_agent runs the simulator and discover_issues in ONE call,")
    print("  from nothing but the agent itself.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
