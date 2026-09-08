"""L2-M2.1.1.1 -- Hand-Written Agent Test Suites.

The suite you write by hand, in full: structured cases, pass/fail on both the
answer and the tool calls, one nested MLflow run per case, and a regression
baseline that a later run is compared against.

It works. Step 3 breaks the agent on purpose and the suite catches it, which is
exactly what a test suite is for. What it cannot do is the subject of the next
three lessons:

  - every case is ONE turn                     -> L2-M2.1.2.1, simulation
  - every case exists because you thought of it -> L2-M2.1.2.4, generation
  - the suite is a Python list, not an asset    -> L2-M2.1.3.1, datasets
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from test_framework import (
    AgentTestRunner,
    TestCase,
    build_results_dataframe,
    compare_to_baseline,
    load_baseline,
    print_summary,
    save_baseline,
)

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/1_turn/1_hand_written_tests"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment(EXPERIMENT)
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


def build_agent(tools: list[Any]) -> Any:
    """Build the agent under test. The tool list is a parameter so Step 3 can
    ship a version with one of them missing."""
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


# The suite. Six cases, every one of them written by a person who sat down and
# imagined a user. That is both the strength and the ceiling.
SUITE: list[TestCase] = [
    TestCase(
        name="order_shipped",
        input="What is the status of order A1001?",
        expected_output="shipped",
        expected_tools=["order_status"],
        difficulty="easy",
    ),
    TestCase(
        name="order_delivered",
        input="Has order A1003 arrived yet?",
        expected_output="delivered",
        expected_tools=["order_status"],
        difficulty="easy",
    ),
    TestCase(
        name="order_held",
        input="Is order A1002 on its way?",
        expected_output="payment",
        expected_tools=["order_status"],
        difficulty="medium",
    ),
    TestCase(
        name="return_window",
        input="How long do I have to return something?",
        expected_output="30 days",
        expected_tools=["policy_lookup"],
        difficulty="easy",
    ),
    TestCase(
        name="warranty_length",
        input="What does the warranty cover and for how long?",
        expected_output="12 months",
        expected_tools=["policy_lookup"],
        difficulty="medium",
    ),
    TestCase(
        name="faulty_return_cost",
        input="My item is faulty. Who pays for the return shipping?",
        expected_output="free",
        expected_tools=["policy_lookup"],
        difficulty="hard",
    ),
]


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.1.1  Hand-Written Agent Test Suites")
    print("=" * 70)

    # -- 1. Run the suite against the agent as shipped ----------------------- #
    print(f"\nStep 1: run {len(SUITE)} hand-written cases against v1")
    agent_v1 = build_agent([order_status, policy_lookup])

    with mlflow.start_run(run_name="suite_v1_both_tools") as run_v1:
        results_v1 = AgentTestRunner(agent_v1, SUITE).run_suite()
        df_v1 = build_results_dataframe(results_v1)
        pass_rate_v1 = float(df_v1["passed"].mean())

        mlflow.log_params({"model": MODEL_ALIAS, "tools": "order_status,policy_lookup", "cases": len(SUITE)})
        mlflow.log_metrics({"pass_rate": pass_rate_v1, "cases": len(SUITE)})
        mlflow.log_table(data=df_v1, artifact_file="results.json")
        # -- 2. Freeze it as the baseline ---------------------------------- #
        save_baseline(df_v1, run_v1.info.run_id)
        baseline_run_id = run_v1.info.run_id

    print_summary(results_v1, SUITE)

    # A case whose tools were right but whose text did not contain the expected
    # substring is usually the MATCHER failing, not the agent: "we cover the
    # return shipping" is a correct answer that does not contain "free".
    matcher_suspects = [r for r in results_v1 if r.tool_usage_correct and not r.output_correct]
    for r in matcher_suspects:
        print(f"\n  {r.test_name} called the right tool but missed '{r.expected_output}':")
        print(f"    {r.agent_output[:100]}")
        print("    That is the substring matcher, not the agent. A judge would pass it.")

    print(f"\nStep 2: baseline frozen on run {baseline_run_id[:8]} as an artifact")
    print("  It lives in MLflow, not on your disk -- CI can read it by run id.")

    # -- 3. Ship a regression, and catch it --------------------------------- #
    print("\n" + "=" * 70)
    print("Step 3: v2 -- somebody dropped policy_lookup in a refactor")
    print("=" * 70)
    agent_v2 = build_agent([order_status])

    with mlflow.start_run(run_name="suite_v2_policy_tool_removed"):
        results_v2 = AgentTestRunner(agent_v2, SUITE).run_suite()
        df_v2 = build_results_dataframe(results_v2)

        mlflow.log_params({"model": MODEL_ALIAS, "tools": "order_status", "cases": len(SUITE)})
        mlflow.log_metrics({"pass_rate": float(df_v2["passed"].mean()), "cases": len(SUITE)})
        mlflow.log_table(data=df_v2, artifact_file="results.json")

        baseline = load_baseline(baseline_run_id)
        regressions, improvements = compare_to_baseline(df_v2, baseline)

    print_summary(results_v2, SUITE)

    # -- 4. What the suite cannot see --------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 4: the ceiling")
    print("=" * 70)
    print(f"  caught     : {len(regressions)} regressions, {len(improvements)} improvements")
    print("  did NOT catch, because no case can express it:")
    print("    - the agent that answers turn 1 well and contradicts itself at turn 3")
    print("    - the order id nobody thought to test, so no case exists for it")
    print("    - a correct answer phrased without the substring the case demands")
    print("\n  Those are structural. Writing more cases does not remove them:")
    print("    2_conversation/1  simulation  -- conversations instead of questions")
    print("    2_conversation/4  test_agent  -- cases nobody had to imagine")
    print("    3_dataset_store/1 datasets   -- the suite as a versioned asset")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
