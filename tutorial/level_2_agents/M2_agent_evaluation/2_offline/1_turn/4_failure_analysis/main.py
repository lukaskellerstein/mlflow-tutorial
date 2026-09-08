"""L2-M2.2.1.4 -- Reading an Evaluation: Slices and Failure Taxonomy.

Every lesson so far produces a NUMBER. This one produces a DECISION about what
to fix next, from results you already have. No new metric, no new scorer -- two
ways of reading the same result set.

  Part 1  SLICES        which segment is failing?
  Part 2  TAXONOMY      why is it failing, and which cause is worth fixing?

Part 1 exists because the aggregate lies. A suite reporting 0.75 can be 1.00 on
three slices and 0.10 on a fourth, and the mean says nothing about which. Ship
on the mean and you ship a product that is perfect for most users and broken
for one group -- and you will hear about it from them, not from your dashboard.

Part 2 exists because "12 cases failed" is not actionable and "9 of the 12
failed for one reason" is. Ranking causes by frequency is what turns an
evaluation into a work queue.

Deterministic throughout. Both techniques are about reading results, so they
work on whatever scorer produced them -- deterministic here, a judge in your
own suite.
"""

from __future__ import annotations

import os
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

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/1_turn/4_failure_analysis"

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

# Every case carries its SLICE. That single field is the whole of Part 1, and it
# costs nothing to add -- which is the point. A suite without slice labels can
# never be broken down afterwards, so add them before you need them.
CASES: list[dict[str, Any]] = [
    # -- order lookups on ids that exist -------------------------------------- #
    {"slice": "known_order", "q": "Status of order A1001?", "want": "shipped", "tool": "order_status"},
    {"slice": "known_order", "q": "Has order A1003 arrived?", "want": "delivered", "tool": "order_status"},
    {"slice": "known_order", "q": "Is order A1002 shipped yet?", "want": "payment", "tool": "order_status"},
    # -- policy questions ------------------------------------------------------ #
    {"slice": "policy", "q": "How long is the return window?", "want": "30 days", "tool": "policy_lookup"},
    {"slice": "policy", "q": "How long is the warranty?", "want": "12 months", "tool": "policy_lookup"},
    {"slice": "policy", "q": "Who pays return shipping if faulty?", "want": "free", "tool": "policy_lookup"},
    # -- ids that do NOT exist. This is the slice that hurts. ------------------ #
    {"slice": "unknown_order", "q": "Status of order Z9999?", "want": "no order", "tool": "order_status"},
    {"slice": "unknown_order", "q": "Where is my order B2222?", "want": "no order", "tool": "order_status"},
    {"slice": "unknown_order", "q": "Has order Q7777 shipped?", "want": "no order", "tool": "order_status"},
]


def answer_of(result: dict[str, Any]) -> str:
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") == "ai" and content and not getattr(msg, "tool_calls", None):
            return str(content)
    return ""


def tools_of(result: dict[str, Any]) -> list[str]:
    return [
        call["name"]
        for msg in result["messages"]
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None)
        for call in msg.tool_calls
    ]


def classify_failure(row: dict[str, Any]) -> str:
    """Name the CAUSE of one failure. Deterministic, and ordered deliberately.

    The order matters: a case can exhibit several symptoms, and the taxonomy is
    only useful if each failure lands in exactly one bucket. Most specific
    cause first, so the count you rank on is a count of root causes rather than
    a count of symptoms.
    """
    if not row["tools_used"]:
        return "no_tool_called"
    if row["expected_tool"] not in row["tools_used"]:
        return "wrong_tool_selected"
    if row["slice"] == "unknown_order" and "no order" not in row["answer"].lower():
        # Right tool, and the tool said the id was unknown -- the agent then
        # answered as though it were known. This is the expensive one.
        return "ignored_tool_result"
    return "fact_missing_from_answer"


def run_suite() -> pd.DataFrame:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    agent = create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)

    rows: list[dict[str, Any]] = []
    print(f"\n  running {len(CASES)} cases...")
    for case in CASES:
        result = agent.invoke(cast(Any, {"messages": [{"role": "user", "content": case["q"]}]}))
        answer = answer_of(result)
        row = {
            "slice": case["slice"],
            "q": case["q"],
            "answer": answer,
            "tools_used": tools_of(result),
            "expected_tool": case["tool"],
            "passed": str(case["want"]).lower() in answer.lower(),
        }
        row["cause"] = "" if row["passed"] else classify_failure(row)
        rows.append(row)
        print(f"    {case['slice']:<14} {case['q'][:38]:<38} {'PASS' if row['passed'] else 'FAIL'}")
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 74)
    print("  L2-M2.2.1.4 -- Reading an Evaluation: Slices and Failure Taxonomy")
    print("=" * 74)

    with mlflow.start_run(run_name="failure_analysis"):
        mlflow.log_params({"model": MODEL_ALIAS, "cases": len(CASES)})
        df = run_suite()

        overall = float(cast(float, df["passed"].mean()))
        print(f"\n{'=' * 74}\n  The number you would normally report\n{'=' * 74}")
        print(f"\n  overall pass rate: {overall:.2f}  ({int(df['passed'].sum())}/{len(df)})")
        print("\n  Ship on that and you are done. Part 1 is why you should not be.")

        # -- Part 1: slices --------------------------------------------------- #
        print(f"\n{'=' * 74}\n  Part 1: the same results, sliced\n{'=' * 74}")
        print(f"\n  {'slice':<16} {'n':>3} {'pass rate':>10}   {'vs overall':>10}")
        print(f"  {'-' * 16} {'-' * 3} {'-' * 10}   {'-' * 10}")
        worst_slice, worst_rate = "", 2.0
        for name, group in df.groupby("slice"):
            rate = float(cast(float, group["passed"].mean()))
            print(f"  {str(name):<16} {len(group):>3} {rate:>10.2f}   {rate - overall:>+10.2f}")
            mlflow.log_metric(f"slice/{name}", rate)
            if rate < worst_rate:
                worst_slice, worst_rate = str(name), rate

        mlflow.log_metrics({"overall_pass_rate": overall, "worst_slice_pass_rate": worst_rate})
        print(f"\n  The aggregate {overall:.2f} is an average of these, weighted by slice size.")
        print(f"  Worst slice: '{worst_slice}' at {worst_rate:.2f}.")
        if worst_rate < overall:
            print("  Every user whose question lands in that slice sees the low number,")
            print("  not the average. They do not experience your mean.")

        print("\n  Read the n column before acting. A slice of 3 that scores 0.00 is a")
        print("  lead, not a finding -- three cases can miss by chance. A slice of 30")
        print("  at 0.40 is a finding. Slice size is what separates the two.")

        # -- Part 2: taxonomy -------------------------------------------------- #
        print(f"\n{'=' * 74}\n  Part 2: the same failures, by cause\n{'=' * 74}")
        failures = df[~df["passed"]]
        if failures.empty:
            print("\n  No failures this run. Nothing to rank -- which is a good problem,")
            print("  and also a reason to widen the suite until something fails.")
        else:
            causes = Counter(failures["cause"])
            total = sum(causes.values())
            print(f"\n  {'cause':<28} {'n':>3} {'share':>7} {'cumulative':>11}")
            print(f"  {'-' * 28} {'-' * 3} {'-' * 7} {'-' * 11}")
            running = 0
            for cause, n in causes.most_common():
                running += n
                print(f"  {cause:<28} {n:>3} {n / total:>6.0%} {running / total:>11.0%}")
                mlflow.log_metric(f"cause/{cause}", n)
            top_cause, top_n = causes.most_common(1)[0]
            mlflow.set_tag("top_failure_cause", top_cause)
            print(f"\n  {top_n} of {total} failures share one cause: '{top_cause}'.")
            print("  That is your next piece of work. Not the first failure you")
            print("  happened to read -- the one that explains the most of them.")

        mlflow.log_table(df.drop(columns=["answer"]), artifact_file="results.json")
        print("\n  Full results logged with log_table(); the slice and cause columns")
        print("  are what make them re-readable later.")

    print(f"\n{'=' * 74}\n  Two warnings\n{'=' * 74}")
    print("  1. A SLICE YOU DID NOT LABEL CANNOT BE ANALYSED. The `slice` field")
    print("     costs nothing at authoring time and is impossible to add")
    print("     retroactively without re-reading every case. Label by anything you")
    print("     might later suspect: length, language, tool needed, customer tier.")
    print("\n  2. AN AGGREGATE CAN MOVE THE OPPOSITE WAY TO EVERY SLICE. If two")
    print("     versions are measured on different slice MIXES, one can win")
    print("     overall while losing on every single slice -- Simpson's paradox.")
    print("     3_version_comparison avoids it by running both versions on the")
    print("     identical cases. Comparing runs with different case mixes does not.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 74)


if __name__ == "__main__":
    main()
