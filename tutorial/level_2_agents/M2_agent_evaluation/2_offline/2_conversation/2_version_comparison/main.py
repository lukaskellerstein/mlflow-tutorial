"""L2-M2.2.2.2 -- Comparing Two Agent Versions Across Conversations.

1_turn/3_version_comparison paired two versions on identical cases and read
win/loss/tie per case. This lesson asks the same question -- ship v2 or keep v1
-- at conversation scope, where that method sometimes works and sometimes
cannot.

Which one you get depends entirely on WHO PLAYS THE USER:

  SCRIPTED user   the turns are fixed strings you wrote. Both versions face
                  the identical conversation, so pairing works exactly as it
                  does at turn scope.                              -> Part 1

  SIMULATED user  a model plays the user and REACTS to what the agent said.
                  v2 answers turn 1 differently, so the user asks a different
                  turn 2, and by turn 3 the two versions are not answering the
                  same conversation at all. Pairing is meaningless.  -> Part 2

That is not a detail. It decides which statistics you are allowed to use, and
it is the reason this lesson exists separately from the turn one.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from typing import Any, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Trace
from mlflow.genai.scorers import ConversationCompleteness, KnowledgeRetention
from mlflow.genai.simulators import ConversationSimulator
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
JUDGE_ALIAS = "gemma-judge"

os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
SESSION_MODEL = f"openai:/{JUDGE_ALIAS}"

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/2_conversation/2_version_comparison"
K = 2  # simulated runs per version in Part 2

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog()

ORDERS = {"A1001": "shipped, arriving Thursday", "A1002": "held at the warehouse, payment not confirmed"}
POLICIES = {
    "returns": "30 days from delivery, receipt required (P-101).",
    "shipping": "Return shipping is free for faulty items (P-330).",
}


@tool
def order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A1001."""
    return ORDERS.get(order_id.strip().upper(), f"No order found with id {order_id}.")


@tool
def policy_lookup(topic: str) -> str:
    """Look up store policy. Topics: returns, shipping."""
    return POLICIES.get(topic.strip().lower(), f"No policy on file for '{topic}'.")


# The only difference between the versions, exactly as in the turn lesson.
VERSIONS = {
    "v1": "You are a retail support agent. Use the tools. Keep replies under three sentences.",
    "v2": (
        "You are a retail support agent. Use the tools. Keep replies under three sentences. "
        "Before you finish, confirm the customer has everything they asked for."
    ),
}

# Part 1 input: the user side is FIXED TEXT. Both versions get these exact turns.
SCRIPTED = [
    {
        "name": "delayed_order",
        "turns": ["What is the status of order A1002?", "So when will it ship?", "And which order was that again?"],
    },
    {
        "name": "faulty_return",
        "turns": ["My item is faulty.", "How long do I have to return it?", "Who pays the shipping?"],
    },
]

# Part 2 input: a GOAL. The user side is generated, and reacts.
SIMULATED = [
    {
        "goal": "Find out why order A1002 has not shipped and what to do about it",
        "persona": "An impatient customer who asks short follow-up questions",
    }
]

SESSION_SCORERS = [
    (ConversationCompleteness, "conversation_completeness"),
    (KnowledgeRetention, "knowledge_retention"),
]
TRUTHY = {"yes", "true", "pass", "y"}


def build_agent(prompt: str) -> Any:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=[order_status, policy_lookup], system_prompt=prompt)


def scorers() -> list[Any]:
    """Built through a factory: `model` is a pydantic field basedpyright cannot see."""
    return [cast(Any, cls)(name=name, model=SESSION_MODEL) for cls, name in SESSION_SCORERS]


def to_metric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in TRUTHY else 0.0
    return 0.0


@mlflow.trace(name="conversation_turn")
def run_turn(agent: Any, history: list[Any], session_id: str) -> dict[str, Any]:
    mlflow.update_current_trace(session_id=session_id)
    return agent.invoke(cast(Any, {"messages": history}))


def score_session(traces: list[Trace]) -> float:
    """Mean of the session scorers over one conversation. NaN if all judges failed."""
    values: list[float] = []
    for scorer in scorers():
        try:
            values.append(to_metric(getattr(scorer(session=traces), "value", 0.0)))
        except Exception:
            continue
    return statistics.mean(values) if values else float("nan")


def run_scripted(agent: Any, scenario: dict[str, Any]) -> float:
    session_id = f"scripted-{scenario['name']}-{uuid.uuid4().hex[:6]}"
    trace_ids: list[str] = []
    history: list[Any] = []
    for turn in scenario["turns"]:
        history.append({"role": "user", "content": turn})
        history = run_turn(agent, history, session_id)["messages"]
        if tid := mlflow.get_last_active_trace_id():
            trace_ids.append(tid)
    mlflow.flush_trace_async_logging()
    return score_session([t for tid in trace_ids if (t := mlflow.get_trace(tid))])


def run_simulated(agent: Any) -> tuple[float, str]:
    """One simulated conversation. Returns its score and its first user turn."""

    def predict_fn(input: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:  # noqa: A002 - the contract
        return agent.invoke(cast(Any, {"messages": input}))

    sim = ConversationSimulator(test_cases=SIMULATED, max_turns=3, user_model=SESSION_MODEL)
    sessions = sim.simulate(predict_fn)
    traces = list(sessions[0]) if sessions else []
    return score_session(traces), first_user_turn(traces)


def first_user_turn(traces: list[Trace]) -> str:
    """The opening user message of a conversation, for the divergence table.

    A trace's request arrives as a JSON STRING on some paths and a dict on
    others. Reading only the dict form silently yields "" -- which is exactly
    what an empty column in the table below would mean.
    """
    if not traces:
        return ""
    payload: Any = traces[0].data.request
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return ""
    msgs = (payload.get("messages") or payload.get("input") or []) if isinstance(payload, dict) else payload
    for msg in msgs or []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            return " ".join(str(msg.get("content", "")).split())
    return ""


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n))


def main() -> None:
    print("=" * 76)
    print("  L2-M2.2.2.2 -- Comparing Two Agent Versions Across Conversations")
    print("=" * 76)

    with mlflow.start_run(run_name="conversation_version_comparison"):
        mlflow.log_params({"model": MODEL_ALIAS, "judge": JUDGE_ALIAS, "k": K})

        # -- Part 1: scripted user -> pairing is legitimate ------------------- #
        print(f"\n{'=' * 76}\n  Part 1: SCRIPTED user -- both versions face identical turns\n{'=' * 76}")
        paired: dict[str, list[float]] = {"v1": [], "v2": []}
        for label, prompt in VERSIONS.items():
            agent = build_agent(prompt)
            for scenario in SCRIPTED:
                score = run_scripted(agent, scenario)
                paired[label].append(score)
                print(f"    {label}  {scenario['name']:<16} {score:.2f}")

        print(f"\n  {'conversation':<20} {'v1':>6} {'v2':>6}  result")
        print(f"  {'-' * 20} {'-' * 6} {'-' * 6}  ------")
        wins = losses = ties = 0
        for scenario, a, b in zip(SCRIPTED, paired["v1"], paired["v2"]):
            if b > a:
                verdict, wins = "v2 WIN", wins + 1
            elif b < a:
                verdict, losses = "v2 LOSS", losses + 1
            else:
                verdict, ties = "tie", ties + 1
            print(f"  {scenario['name']:<20} {a:>6.2f} {b:>6.2f}  {verdict}")
        p = sign_test(wins, losses)
        print(f"\n  paired: {wins} win / {losses} loss / {ties} tie   sign test p = {p:.3f}")
        print("\n  This is legitimate ONLY because the user turns were fixed strings.")
        if wins == losses == 0:
            print("\n  Everything tied, and that is worth reading rather than skipping.")
            print("  Session scorers answer yes/no -- they are COARSE. Two competent")
            print("  versions tie on a two-conversation suite almost every time.")
            print("  A tie means the suite lacks the RESOLUTION to separate them, not")
            print("  that the versions are equivalent. Add conversations, or score per")
            print("  turn as well -- see 3_failure_analysis.")
        mlflow.log_metrics({"paired_wins": wins, "paired_losses": losses, "paired_p": p})

        # -- Part 2: simulated user -> pairing collapses ---------------------- #
        print(f"\n{'=' * 76}\n  Part 2: SIMULATED user -- the same goal, {K} runs per version\n{'=' * 76}")
        dist: dict[str, list[float]] = {"v1": [], "v2": []}
        openings: dict[str, list[str]] = {"v1": [], "v2": []}
        for label, prompt in VERSIONS.items():
            agent = build_agent(prompt)
            for run_i in range(1, K + 1):
                score, first = run_simulated(agent)
                dist[label].append(score)
                openings[label].append(first)
                print(f"    {label}  run {run_i}/{K}  score {score:.2f}   opened: {first[:44]}")

        print("\n  Look at the 'opened' column. Same goal, different first turns --")
        print("  the user is a model. Run 1 of v1 and run 1 of v2 are NOT the same")
        print("  conversation, so there is nothing to pair and no win/loss to count.")

        print(f"\n  {'version':<10} {'runs':>5} {'mean':>7} {'spread':>8}")
        print(f"  {'-' * 10} {'-' * 5} {'-' * 7} {'-' * 8}")
        summary: dict[str, float] = {}
        for label in ("v1", "v2"):
            clean = [v for v in dist[label] if v == v]
            mean = statistics.mean(clean) if clean else float("nan")
            spread = (max(clean) - min(clean)) if len(clean) > 1 else 0.0
            summary[label] = mean
            print(f"  {label:<10} {len(clean):>5} {mean:>7.2f} {spread:>8.2f}")
            mlflow.log_metrics({f"sim_{label}_mean": mean, f"sim_{label}_spread": spread})

        delta = summary["v2"] - summary["v1"]
        print(f"\n  delta of means: {delta:+.2f}")
        print("\n  Now the honest part. With 2 runs per version you cannot tell that")
        print("  delta from noise -- the spread column is the same size as the effect.")
        print("  A distribution comparison needs enough runs that the spread is small")
        print("  relative to the difference you care about, and 2 is never enough.")
        print("  Budget for it before you promise a conversation-level A/B result.")

    print(f"\n{'=' * 76}\n  Which method, and when\n{'=' * 76}")
    print("  SCRIPTED user   -> paired, win/loss/tie, sign test.")
    print("                     Cheap and sensitive. Cannot discover anything you")
    print("                     did not write into the script.")
    print("  SIMULATED user  -> distributions over k runs per version.")
    print("                     Finds behaviour you never thought to script, and")
    print("                     costs k times as much to reach the same confidence.")
    print("\n  Use scripted conversations for the regression gate you run on every")
    print("  commit, and simulated ones for the periodic sweep that looks for what")
    print("  the gate cannot see. They are not competing options.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 76)


if __name__ == "__main__":
    main()
