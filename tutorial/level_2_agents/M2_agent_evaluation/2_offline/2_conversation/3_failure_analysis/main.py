"""L2-M2.2.2.3 -- Where Conversations Break Down: Funnel and Slices.

1_turn/4_failure_analysis read a set of results two ways: by slice, and by
cause. This lesson does the same at conversation scope, and adds the reading
that only exists here.

A session scorer returns ONE verdict for a conversation of N turns. That is
enough to fail a build and useless for fixing it -- "the conversation failed"
does not say whether the agent fell over on turn 1 or held up until turn 4.
The funnel says which.

  Part 1  FUNNEL   at which turn do conversations stop being healthy?
  Part 2  SLICES   which kind of conversation dies, and from what cause?

Turn 3 of every scenario below deliberately refers back to turn 1 without
restating it. That is the cliff -- and a per-turn view is the only way to see
that conversations survive to turn 2 and fall off at turn 3.

Deterministic scoring throughout. Localising a failure is about reading
results, so it works with whatever produced them.
"""

from __future__ import annotations

import os
import uuid
from collections import Counter
from typing import Any, cast

import mlflow
import mlflow.langchain
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/2_conversation/3_failure_analysis"

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
    "Keep replies under three sentences."
)

# THE AGENT UNDER TEST HAS A CONTEXT BUDGET, and that is the point of the
# lesson rather than a flaw in it. Truncating history is one of the most common
# production choices there is -- it caps token cost per turn and keeps latency
# flat as a conversation grows. It is also never free, and the funnel is how
# you find out what it cost. Raise this number and watch the cliff move.
HISTORY_WINDOW = 4

# Each turn carries what its answer must contain, so every turn is scored, not
# just the conversation. `slice` groups conversations; turn 3 is always a
# callback to turn 1 with the subject omitted.
CONVERSATIONS: list[dict[str, Any]] = [
    {
        "slice": "order_then_recall",
        "turns": [
            {"say": "What is the status of order A1001?", "want": "shipped"},
            {"say": "How long do I have to return something?", "want": "30 days"},
            {"say": "Remind me which day that order was arriving?", "want": "thursday"},
        ],
    },
    {
        "slice": "order_then_recall",
        "turns": [
            {"say": "Has order A1003 arrived?", "want": "delivered"},
            {"say": "What does the warranty cover?", "want": "12 months"},
            {"say": "Remind me when that order arrived?", "want": "monday"},
        ],
    },
    {
        "slice": "policy_only",
        "turns": [
            {"say": "How long is the return window?", "want": "30 days"},
            {"say": "Who pays return shipping if faulty?", "want": "free"},
            {"say": "And how long was the return window again?", "want": "30 days"},
        ],
    },
    {
        "slice": "policy_only",
        "turns": [
            {"say": "What does the warranty cover?", "want": "12 months"},
            {"say": "How long do I have to return an item?", "want": "30 days"},
            {"say": "Remind me the warranty length?", "want": "12 months"},
        ],
    },
]


def answer_of(result: dict[str, Any]) -> str:
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") == "ai" and content and not getattr(msg, "tool_calls", None):
            return str(content)
    return ""


def tools_of(result: dict[str, Any], seen: int) -> list[str]:
    """Tool names used in THIS turn -- messages after the ones already counted."""
    return [
        call["name"]
        for msg in result["messages"][seen:]
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None)
        for call in msg.tool_calls
    ]


@mlflow.trace(name="conversation_turn")
def run_turn(agent: Any, history: list[Any], session_id: str) -> dict[str, Any]:
    mlflow.update_current_trace(session_id=session_id)
    return agent.invoke(cast(Any, {"messages": history}))


def classify(turn_index: int, passed: bool, tools_used: list[str]) -> str:
    """Why this turn failed. The causes are conversation-specific on purpose."""
    if passed:
        return ""
    if turn_index >= 3 and not tools_used:
        # The callback turn, answered from memory with no lookup. Either the
        # agent forgot, or it invented. Both are context failures.
        return "lost_earlier_context"
    if not tools_used:
        return "no_tool_called"
    return "tool_called_answer_wrong"


def run_all() -> pd.DataFrame:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    agent = create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)

    rows: list[dict[str, Any]] = []
    for convo_i, convo in enumerate(CONVERSATIONS, 1):
        session_id = f"funnel-{uuid.uuid4().hex[:8]}"
        history: list[Any] = []
        seen = 0
        print(f"\n  conversation {convo_i} [{convo['slice']}]")
        for turn_i, turn in enumerate(convo["turns"], 1):
            history.append({"role": "user", "content": turn["say"]})
            # Only the most recent HISTORY_WINDOW messages reach the model.
            result = run_turn(agent, history[-HISTORY_WINDOW:], session_id)
            answer = answer_of(result)
            tools_used = tools_of(result, seen)
            history, seen = result["messages"], len(result["messages"])
            passed = str(turn["want"]).lower() in answer.lower()
            rows.append(
                {
                    "conversation": convo_i,
                    "slice": convo["slice"],
                    "turn": turn_i,
                    "passed": passed,
                    "cause": classify(turn_i, passed, tools_used),
                }
            )
            print(f"    turn {turn_i}  {turn['say'][:44]:<44} {'PASS' if passed else 'FAIL'}")
    mlflow.flush_trace_async_logging()
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 76)
    print("  L2-M2.2.2.3 -- Where Conversations Break Down")
    print("=" * 76)

    with mlflow.start_run(run_name="conversation_failure_analysis"):
        mlflow.log_params({"model": MODEL_ALIAS, "conversations": len(CONVERSATIONS)})
        df = run_all()

        # -- The verdict a session scorer would give -------------------------- #
        per_convo = df.groupby("conversation")["passed"].all()
        healthy = int(cast(int, per_convo.sum()))
        print(f"\n{'=' * 76}\n  What a session scorer tells you\n{'=' * 76}")
        print(f"\n  {healthy}/{len(per_convo)} conversations fully succeeded.")
        print("  One bit per conversation. True, and not actionable -- it does not")
        print("  say WHERE they went wrong.")

        # -- Part 1: the funnel ----------------------------------------------- #
        print(f"\n{'=' * 76}\n  Part 1: the funnel -- still healthy at turn N\n{'=' * 76}")
        total = len(CONVERSATIONS)
        alive = set(range(1, total + 1))
        max_turn = int(cast(int, df["turn"].max()))
        print(f"\n  {'turn':<6} {'still healthy':>14} {'survival':>10} {'dropped here':>13}")
        print(f"  {'-' * 6} {'-' * 14} {'-' * 10} {'-' * 13}")
        worst_turn, worst_drop = 0, 0
        for turn_i in range(1, max_turn + 1):
            failed_here = {
                int(r["conversation"])
                for _, r in df[(df["turn"] == turn_i) & (~df["passed"])].iterrows()
                if int(r["conversation"]) in alive
            }
            alive -= failed_here
            print(f"  {turn_i:<6} {len(alive):>14} {len(alive) / total:>10.0%} {len(failed_here):>13}")
            mlflow.log_metric(f"funnel/alive_after_turn_{turn_i}", len(alive))
            if len(failed_here) > worst_drop:
                worst_turn, worst_drop = turn_i, len(failed_here)

        if worst_drop:
            print(f"\n  The cliff is TURN {worst_turn}: {worst_drop} of {total} conversations died there.")
            print("  That is a place in the code, not a vague quality problem.")
            mlflow.set_tag("cliff_turn", str(worst_turn))
        else:
            print("\n  No cliff -- every conversation survived. Widen the suite until")
            print("  something breaks, or you are measuring nothing.")

        print("\n  Read 'still healthy' as a funnel, not a per-turn pass rate. Once a")
        print("  conversation has broken it stays broken, because the customer has")
        print("  already had the bad experience -- a later good turn does not undo it.")

        # -- Part 2: slices and causes ---------------------------------------- #
        print(f"\n{'=' * 76}\n  Part 2: which conversations, and why\n{'=' * 76}")
        print(f"\n  {'slice':<20} {'convos':>7} {'turn pass rate':>15}")
        print(f"  {'-' * 20} {'-' * 7} {'-' * 15}")
        for name, group in df.groupby("slice"):
            rate = float(cast(float, group["passed"].mean()))
            n_convos = group["conversation"].nunique()
            print(f"  {str(name):<20} {n_convos:>7} {rate:>15.2f}")
            mlflow.log_metric(f"slice/{name}", rate)

        failures = df[~df["passed"]]
        if failures.empty:
            print("\n  No failed turns to classify.")
        else:
            causes = Counter(failures["cause"])
            total_f = sum(causes.values())
            print(f"\n  {'cause':<28} {'n':>3} {'share':>7}")
            print(f"  {'-' * 28} {'-' * 3} {'-' * 7}")
            for cause, n in causes.most_common():
                print(f"  {cause:<28} {n:>3} {n / total_f:>6.0%}")
                mlflow.log_metric(f"cause/{cause}", n)
            top = causes.most_common(1)[0]
            mlflow.set_tag("top_cause", top[0])

        mlflow.log_table(df, artifact_file="turns.json")

    print(f"\n{'=' * 76}\n  What this reading gives you that a session verdict cannot\n{'=' * 76}")
    print("  A session scorer answers 'did this conversation succeed?'.")
    print("  The funnel answers 'at which turn did it stop succeeding?'.")
    print("\n  The second question is the one you can act on. 'Conversations die at")
    print("  turn 3, and the cause is lost context' names a fix. 'Completeness is")
    print("  0.5' does not.")
    print("\n  Both readings are cheap: they need a per-turn pass flag and a turn")
    print("  number, which any multi-turn suite already has. What they need that")
    print("  is easy to omit is that you SCORE EVERY TURN, not just the session.")
    print("  Score only the session and this analysis is impossible afterwards.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 76)


if __name__ == "__main__":
    main()
