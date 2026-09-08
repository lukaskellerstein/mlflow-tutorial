"""L2-M2.2.1.3 -- Comparing Two Versions of One Agent.

1_architecture_comparison compares three DIFFERENT things -- a chain, a ReAct
agent, a graph. This lesson compares two versions of the SAME thing: you
rewrote the system prompt, and you want to know whether to ship it.

That is a different statistical question, and the difference is the lesson.

  independent comparison   3 architectures, each with its own mean
  PAIRED comparison        1 agent, 2 versions, the SAME cases in both

Because both versions answer the identical cases, you can compare case by case
instead of only mean against mean. That buys two things a mean cannot give you:

  1. WIN / LOSS / TIE per case. A mean that rises by 0.05 might be three cases
     improving, or one case improving a lot while two get worse. Those call for
     opposite decisions, and the mean cannot tell them apart.
  2. A SIGNIFICANCE TEST that works on the 6-20 cases a real suite has. The sign
     test asks: if the versions were truly equal, how often would chance alone
     produce a split this lopsided?

Scoring here is deterministic on purpose. A judge would add variance to BOTH
versions, and on a suite this small that noise swamps the effect you are trying
to measure. See 4_failure_analysis for what to do once you need a judge.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/1_turn/3_version_comparison"

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


# THE ONLY THING THAT DIFFERS. Same model, same tools, same cases, same scorer --
# so any change in the numbers is attributable to this and nothing else. That
# discipline is what makes a comparison a comparison rather than an anecdote.
VERSIONS = {
    "v1": (
        "You are a retail support agent. Use the tools to answer questions about "
        "orders and policy. Keep replies under three sentences."
    ),
    "v2": (
        "You are a retail support agent. Use the tools to answer questions about "
        "orders and policy. ALWAYS quote the policy reference code (for example "
        "P-101) whenever you state a policy. Keep replies under three sentences."
    ),
}

# `cites_policy` is where v2 is meant to win. `max_words` is where its extra
# instruction may cost it. A suite with only the first kind of case cannot
# detect a regression, and would make every prompt change look like an
# improvement.
CASES: list[dict[str, Any]] = [
    {"q": "How long do I have to return something?", "contains": "30 days", "cites_policy": True, "max_words": 60},
    {"q": "What does the warranty cover?", "contains": "12 months", "cites_policy": True, "max_words": 60},
    {"q": "Who pays return shipping on a faulty item?", "contains": "free", "cites_policy": True, "max_words": 60},
    {"q": "What is the status of order A1001?", "contains": "shipped", "cites_policy": False, "max_words": 40},
    {"q": "Has order A1003 arrived?", "contains": "delivered", "cites_policy": False, "max_words": 40},
    {"q": "Is order A1002 on its way?", "contains": "payment", "cites_policy": False, "max_words": 40},
]

POLICY_CODES = ("P-101", "P-204", "P-330")


def build_agent(system_prompt: str) -> Any:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=[order_status, policy_lookup], system_prompt=system_prompt)


def answer_of(result: dict) -> str:
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") == "ai" and content and not getattr(msg, "tool_calls", None):
            return str(content)
    return ""


def score_case(answer: str, case: dict[str, Any]) -> float:
    """Deterministic, 0.0-1.0. Three components, equally weighted.

    Deterministic matters more here than anywhere else in the module: the whole
    method rests on the two versions seeing identical conditions, and a judge
    would re-roll on every call for both of them.
    """
    got_answer = 1.0 if str(case["contains"]).lower() in answer.lower() else 0.0
    cited = any(code in answer for code in POLICY_CODES)
    got_citation = 1.0 if cited == bool(case["cites_policy"]) else 0.0
    concise = 1.0 if len(answer.split()) <= int(case["max_words"]) else 0.0
    return round((got_answer + got_citation + concise) / 3.0, 3)


def run_version(label: str, system_prompt: str) -> list[dict[str, Any]]:
    """Run every case once, in the SAME order, and score deterministically."""
    agent = build_agent(system_prompt)
    rows: list[dict[str, Any]] = []
    print(f"\n  [{label}] running {len(CASES)} cases...")
    for case in CASES:
        started = time.time()
        answer = answer_of(agent.invoke(cast(Any, {"messages": [{"role": "user", "content": case["q"]}]})))
        rows.append(
            {
                "q": case["q"],
                "answer": answer,
                "score": score_case(answer, case),
                "latency_s": round(time.time() - started, 2),
            }
        )
        print(f"    {case['q'][:44]:<44} {rows[-1]['score']:.2f}")
    return rows


def sign_test(wins: int, losses: int) -> float:
    """Two-sided sign test p-value. Ties are excluded, which is the point of it.

    Under the null ("the versions are equal") each non-tied case is a coin flip.
    This asks how often chance alone would produce a split at least this
    lopsided. On six cases it almost never clears 0.05 -- and that is the
    honest answer, not a defect in the test.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1))
    return min(1.0, 2 * tail / (2**n))


def main() -> None:
    print("=" * 74)
    print("  L2-M2.2.1.3 -- Comparing Two Versions of One Agent (paired)")
    print("=" * 74)

    with mlflow.start_run(run_name="version_comparison") as parent:
        results: dict[str, list[dict[str, Any]]] = {}
        for label, prompt in VERSIONS.items():
            with mlflow.start_run(run_name=label, nested=True):
                mlflow.log_params({"version": label, "model": MODEL_ALIAS, "cases": len(CASES)})
                rows = run_version(label, prompt)
                results[label] = rows
                mean = sum(r["score"] for r in rows) / len(rows)
                mlflow.log_metrics({"mean_score": mean, "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows)})

        v1, v2 = results["v1"], results["v2"]
        mean1 = sum(r["score"] for r in v1) / len(v1)
        mean2 = sum(r["score"] for r in v2) / len(v2)

        # -- The part a mean cannot give you --------------------------------- #
        print(f"\n{'=' * 74}\n  Per-case: the same cases, so they can be paired\n{'=' * 74}")
        print(f"  {'case':<46} {'v1':>5} {'v2':>5}  result")
        print(f"  {'-' * 46} {'-' * 5} {'-' * 5}  ------")
        wins = losses = ties = 0
        for a, b in zip(v1, v2):
            if b["score"] > a["score"]:
                verdict, wins = "v2 WIN", wins + 1
            elif b["score"] < a["score"]:
                verdict, losses = "v2 LOSS", losses + 1
            else:
                verdict, ties = "tie", ties + 1
            print(f"  {a['q'][:46]:<46} {a['score']:>5.2f} {b['score']:>5.2f}  {verdict}")

        p = sign_test(wins, losses)
        print(f"\n  means      : v1 {mean1:.3f}  ->  v2 {mean2:.3f}   (delta {mean2 - mean1:+.3f})")
        print(f"  paired     : {wins} win / {losses} loss / {ties} tie")
        print(f"  sign test  : p = {p:.3f} over {wins + losses} non-tied cases")

        mlflow.log_metrics(
            {
                "v1_mean": mean1,
                "v2_mean": mean2,
                "delta": mean2 - mean1,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_p": p,
            }
        )

        verdict = "PROMOTE v2" if (wins > losses and p < 0.05) else "KEEP v1 (not proven)"
        print(f"\n  VERDICT: {verdict}")
        mlflow.set_tag("verdict", verdict)
        print(f"\n  parent run: {parent.info.run_id}")

    # -- Why this is not the same as comparing two means --------------------- #
    print(f"\n{'=' * 74}\n  What the pairing bought\n{'=' * 74}")
    print("  A delta of +0.05 in the mean is compatible with all of these:")
    print("    - six cases each a little better        -> ship it")
    print("    - three better, three worse             -> you changed behaviour,")
    print("                                               you did not improve it")
    print("    - one much better, two slightly worse   -> you traded, and you")
    print("                                               should know what for")
    print("  The win/loss/tie column above tells them apart. The mean cannot.")
    print("\n  And read the p-value honestly. With six cases the sign test almost")
    print("  never clears 0.05, which is the correct answer: a six-case suite")
    print("  cannot prove a small improvement. Either collect more cases or")
    print("  accept that you are shipping on judgement, not on evidence.")
    print("\n  NOT to be confused with L2-M3.2 (Configuration Optimization). That")
    print("  SEARCHES a space of models and tool budgets for a good setting. This")
    print("  DECIDES between two candidates you already have.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 74)


if __name__ == "__main__":
    main()
