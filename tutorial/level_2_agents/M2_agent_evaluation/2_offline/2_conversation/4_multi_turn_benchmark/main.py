"""L2-M2.2.2.4 -- A Multi-Turn Agent Benchmark.

`1_turn/5_swe_bench` and `1_turn/6_gaia` are single-turn: one task in, one answer
out, compared to a frozen expected answer. That is not laziness. A benchmark's
whole value is that its number means the same thing to everyone, and multi-turn
breaks that in three places at once.

This lesson builds a small multi-turn benchmark and fixes all three, using the
pattern tau-bench established:

  PROBLEM                                   FIX
  the conversation depends on the        -> PIN the user model in the benchmark
  user simulator, which is a model          spec, not in the caller's config
  judging conversation text needs a      -> score the FINAL STATE of a database
  judge, which is a model too               the agent had to mutate
  one multi-turn run is luck             -> run each task k times and report
                                            pass^k, not just pass@1

The third is the one people skip, and it is the one that matters most. An agent
that solves a task 3 times out of 4 is not a 75% agent you can ship -- it is an
agent that fails one customer in four, and pass^k is what makes that visible.

Nothing here calls an LLM judge. The metric is a dict comparison.
"""

from __future__ import annotations

import copy
import os
import statistics
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.genai.simulators import ConversationSimulator
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

# PART OF THE BENCHMARK SPEC, NOT A CONFIG KNOB.
# Everything else in this repo names the model it wants and lets the gateway
# resolve it. A benchmark cannot: change the user simulator and every
# conversation changes, so the score stops being comparable to anyone else's.
# Publishing this benchmark means publishing this line.
BENCHMARK_USER_MODEL = "gemma-judge"

os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
USER_MODEL = f"openai:/{BENCHMARK_USER_MODEL}"

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/2_conversation/4_multi_turn_benchmark"
K = 2  # trials per task; pass^K is the headline number
MAX_TURNS = 4

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog()

# The world the agent can CHANGE. Scoring reads this dict afterwards, which is
# why the metric needs no judge and is byte-for-byte reproducible.
INITIAL_DB: dict[str, dict[str, Any]] = {
    "A1001": {"status": "shipped", "address": "12 Oak Street", "cancellable": False},
    "A1002": {"status": "pending_payment", "address": "5 Elm Road", "cancellable": True},
    "A1003": {"status": "delivered", "address": "9 Pine Lane", "cancellable": False},
}


def build_tools(db: dict[str, dict[str, Any]]) -> list[Any]:
    """Tools bound to ONE task's database, so trials cannot contaminate each other."""

    @tool
    def get_order(order_id: str) -> str:
        """Look up an order by id, e.g. A1002. Returns status and address."""
        order = db.get(order_id.strip().upper())
        if not order:
            return f"No order found with id {order_id}."
        return f"{order_id}: status={order['status']}, address={order['address']}"

    @tool
    def cancel_order(order_id: str) -> str:
        """Cancel an order. Only orders that are not yet shipped can be cancelled."""
        order = db.get(order_id.strip().upper())
        if not order:
            return f"No order found with id {order_id}."
        if not order["cancellable"]:
            return f"Order {order_id} is {order['status']} and can no longer be cancelled."
        order["status"] = "cancelled"
        return f"Order {order_id} has been cancelled."

    @tool
    def update_address(order_id: str, new_address: str) -> str:
        """Change the shipping address on an order that has not shipped yet."""
        order = db.get(order_id.strip().upper())
        if not order:
            return f"No order found with id {order_id}."
        if order["status"] in ("shipped", "delivered"):
            return f"Order {order_id} is already {order['status']}; the address cannot change."
        order["address"] = new_address
        return f"Address for {order_id} updated to {new_address}."

    return [get_order, cancel_order, update_address]


SYSTEM_PROMPT = (
    "You are a retail support agent. Use the tools to look up and change orders. "
    "Never claim an action succeeded unless the tool said so. Keep replies under three sentences."
)

# Each task is a GOAL for the simulated user, plus the state the database must be
# in afterwards. The check is a plain function -- no model reads the transcript.
TASKS: list[dict[str, Any]] = [
    {
        "name": "cancel_pending_order",
        "scenario": {
            "goal": "Cancel order A1002 because you no longer want it",
            "persona": "A direct customer who states what they want and confirms once done",
        },
        "check": lambda db: db["A1002"]["status"] == "cancelled",
        "expected": "A1002.status == 'cancelled'",
    },
    {
        "name": "refuse_impossible_cancel",
        "scenario": {
            "goal": "Cancel order A1001, and if that is impossible find out why",
            "persona": "A polite customer who accepts a clear explanation",
        },
        # The agent PASSES by leaving the world alone. A benchmark that only
        # rewards mutation teaches agents to act when they should refuse.
        "check": lambda db: db["A1001"]["status"] == "shipped",
        "expected": "A1001.status unchanged ('shipped')",
    },
]


def run_trial(task: dict[str, Any]) -> tuple[bool, int]:
    """One trial: fresh database, fresh agent, one simulated conversation."""
    db = copy.deepcopy(INITIAL_DB)
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    agent = create_agent(llm, tools=build_tools(db), system_prompt=SYSTEM_PROMPT)

    def predict_fn(input: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:  # noqa: A002 - the contract
        from typing import cast

        return agent.invoke(cast(Any, {"messages": input}))

    simulator = ConversationSimulator(
        test_cases=[task["scenario"]],
        max_turns=MAX_TURNS,
        user_model=USER_MODEL,
    )
    traces = simulator.simulate(predict_fn)
    turns = sum(len(t) for t in traces)

    check: Callable[[dict[str, Any]], bool] = task["check"]
    return bool(check(db)), turns


def main() -> None:
    print("=" * 70)
    print("  L2-M2.2.2.4 -- A Multi-Turn Agent Benchmark")
    print(f"  {len(TASKS)} tasks x {K} trials, user model PINNED to {BENCHMARK_USER_MODEL}")
    print("=" * 70)

    results: dict[str, list[bool]] = {}
    turn_counts: list[int] = []
    # Bound here, not only inside the run block below: they are read after it.
    pass_at_1 = pass_pow_k = 0.0

    with mlflow.start_run(run_name=f"benchmark_k{K}"):
        mlflow.log_params(
            {
                "agent_model": MODEL_ALIAS,
                "benchmark_user_model": BENCHMARK_USER_MODEL,
                "k": K,
                "max_turns": MAX_TURNS,
                "tasks": len(TASKS),
            }
        )

        for task in TASKS:
            print(f"\n  task: {task['name']}")
            print(f"    goal     : {task['scenario']['goal']}")
            print(f"    expected : {task['expected']}")
            outcomes: list[bool] = []
            for trial in range(1, K + 1):
                passed, turns = run_trial(task)
                outcomes.append(passed)
                turn_counts.append(turns)
                print(f"    trial {trial}/{K}: {'PASS' if passed else 'FAIL'}  ({turns} turns)")
            results[task["name"]] = outcomes

        # -- The two numbers, and why they differ ---------------------------- #
        flat = [o for outs in results.values() for o in outs]
        pass_at_1 = statistics.mean(1.0 if o else 0.0 for o in flat)
        pass_pow_k = statistics.mean(1.0 if all(outs) else 0.0 for outs in results.values())

        print(f"\n{'=' * 70}\n  Results\n{'=' * 70}")
        print(f"  {'task':<28} {'trials':>10}  {'pass^k':>7}")
        print(f"  {'-' * 28} {'-' * 10}  {'-' * 7}")
        for name, outs in results.items():
            marks = "".join("P" if o else "F" for o in outs)
            print(f"  {name:<28} {marks:>10}  {'YES' if all(outs) else 'NO':>7}")

        print(f"\n  pass@1  : {pass_at_1:.2f}   (every trial counted on its own)")
        print(f"  pass^{K}  : {pass_pow_k:.2f}   (a task counts only if ALL {K} trials passed)")
        mlflow.log_metrics(
            {
                "pass_at_1": pass_at_1,
                f"pass_pow_{K}": pass_pow_k,
                "mean_turns": statistics.mean(turn_counts) if turn_counts else 0.0,
            }
        )

    print(f"\n{'=' * 70}\n  What this benchmark bought, and what it did not\n{'=' * 70}")
    print("  BOUGHT")
    print("    - a metric with no judge in it: the check is a dict comparison,")
    print("      so re-scoring an old transcript gives the same answer forever")
    print("    - a task that passes by NOT acting, so refusing correctly scores")
    print("    - pass^k, which separates 'solved it' from 'solved it once'")
    if pass_at_1 != pass_pow_k:
        print(f"\n    Look at the gap: pass@1 {pass_at_1:.2f} vs pass^{K} {pass_pow_k:.2f}.")
        print("    Same agent, same tasks. The first number is the one that")
        print("    flatters you, and the second is the one your users feel.")
    print("\n  NOT BOUGHT")
    print("    - comparability with ANOTHER team. That needs them to run the")
    print("      same user model, the same simulator version and the same k.")
    print("      A benchmark is a frozen spec, not a script -- which is why")
    print(f"      BENCHMARK_USER_MODEL is pinned in this file and '{BENCHMARK_USER_MODEL}'")
    print("      belongs in anything you publish alongside the number.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
