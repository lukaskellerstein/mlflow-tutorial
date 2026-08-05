"""L2-M2.1.1 -- Agent Test Generation and Simulation.

Hand-written test cases are where agent testing starts and where it stops
scaling. They are single-turn, they cover only what somebody already thought of,
and every new failure mode is another case a human has to write.

`test_framework.py` next to this file is that hand-rolled approach, kept
deliberately: Step 1 runs it, so you can see what it buys and where it stops.
Steps 2-4 then replace it with three things MLflow ships:

  1. ConversationSimulator  -- drives MULTI-TURN conversations from a goal and a
     persona, so you test the conversation and not one question.
  2. mlflow.genai.test_agent() -- asks the agent to describe itself, generates
     cases from that description, simulates them, and reports the issues found.
  3. mlflow.genai.create_dataset() -- promotes the result into a versioned
     dataset the rest of the module evaluates against.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.genai.simulators import ConversationSimulator
from pydantic import SecretStr
from test_framework import AgentTestRunner, TestCase, print_summary

GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-large"

# The simulator and test_agent resolve their own model through LiteLLM, which
# reads these. Assignments, not setdefault: a real OPENAI_API_KEY in the
# environment would win and every simulated turn would be rejected by the gateway
# with "Invalid proxy server token passed", a long way from its cause.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

SIM_MODEL = f"openai:/{MODEL_ALIAS}"
EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/1_agent_testing"
DATASET_NAME = "support_agent_regression"
MAX_TURNS = 3

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


def build_agent() -> Any:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)


AGENT = build_agent()

# Step 1's suite: three cases, hand-written, single-turn. This is the ceiling.
HAND_WRITTEN: list[TestCase] = [
    TestCase(
        name="order_shipped",
        input="What is the status of order A1001?",
        expected_output="shipped",
        expected_tools=["order_status"],
        difficulty="easy",
    ),
    TestCase(
        name="return_window",
        input="How long do I have to return something?",
        expected_output="30 days",
        expected_tools=["policy_lookup"],
        difficulty="easy",
    ),
    TestCase(
        name="order_held",
        input="Is order A1002 on its way?",
        expected_output="payment",
        expected_tools=["order_status"],
        difficulty="medium",
    ),
]


def field_of(item: Any, *names: str) -> str:
    """Read the first present field from a test case or issue.

    `test_agent` hands back plain dicts, not the pydantic models its source
    defines, and issues carry a title on some paths and a description on others.
    Reading either shape keeps this lesson working across both.
    """
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value:
            return " ".join(str(value).split())
    return " ".join(str(item).split())


def predict_fn(input: list[dict], **_kwargs: Any) -> dict:  # noqa: A002 - parameter name fixed by the simulator
    """The contract ConversationSimulator drives.

    It must accept `input` -- the conversation so far, as message dicts -- and
    return something MLflow can parse a reply out of. A LangGraph agent's native
    `{"messages": [...]}` is one of those shapes, so the agent's output is
    returned untouched and no adapter is needed.

    The simulator also passes `mlflow_session_id` as a keyword argument. This
    agent is stateless -- the whole conversation arrives in `input` every turn --
    so it is ignored here, but a stateful agent would key its memory on it.
    """
    return AGENT.invoke({"messages": input})


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.1  Agent Test Generation and Simulation")
    print("=" * 70)

    # ── 1. Where hand-written tests get you ─────────────────────────────────── #
    print("\nStep 1: the hand-rolled baseline (test_framework.py, single-turn)")
    with mlflow.start_run(run_name="hand_written_suite"):
        results = AgentTestRunner(AGENT, HAND_WRITTEN).run_suite()
    print_summary(results, HAND_WRITTEN)
    hand_pass = sum(r.passed for r in results)
    print("\n  Every case there is one a human already thought of, and every one is")
    print("  a single turn. Real users are neither.")

    # ── 2. Simulated multi-turn conversations ───────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 2: ConversationSimulator -- goals and personas, not questions")
    print("=" * 70)
    scenarios: list[dict[str, Any]] = [
        {
            "goal": "Find out whether order A1002 will arrive this week, and why it is delayed",
            "persona": "An impatient customer who asks short, blunt follow-up questions",
        },
        {
            "goal": "Return a faulty item bought two months ago and find out who pays return shipping",
            "persona": "A polite first-time customer who does not know the policy names",
            "simulation_guidelines": ["Do not say the word 'warranty' unless the agent says it first"],
        },
    ]
    simulator = ConversationSimulator(test_cases=scenarios, max_turns=MAX_TURNS, user_model=SIM_MODEL)
    sim_traces = simulator.simulate(predict_fn)

    print(f"  simulated {len(sim_traces)} conversations")
    for scenario, traces in zip(scenarios, sim_traces):
        print(f"    - {str(scenario['goal'])[:56]:<56} {len(traces)} turns traced")
    print("\n  Every turn is a trace, and the simulated user reacts to what the agent")
    print("  actually said -- so turn 3 only happens if turn 2 was any good.")

    # ── 3. Automated test generation + issue discovery ──────────────────────── #
    print("\n" + "=" * 70)
    print("Step 3: mlflow.genai.test_agent() -- generate, simulate, discover")
    print("=" * 70)
    print("  describing the agent, generating cases, simulating them...")
    result = mlflow.genai.test_agent(
        predict_fn,
        model=SIM_MODEL,
        num_test_cases=3,
        max_turns=MAX_TURNS,
        guidance="Focus on order ids that do not exist, and policy questions the tools cannot answer.",
    )

    description = " ".join(str(result.agent_description).split())
    print(f"\n  the agent described itself as:\n    {description[:150]}")
    print(f"\n  generated {len(result.test_cases)} test cases:")
    for case in result.test_cases:
        print(f"    - {field_of(case, 'goal')[:64]}")

    discovery = result.issues_result
    issues = list(getattr(discovery, "issues", None) or [])
    analysed = getattr(discovery, "total_traces_analyzed", None)
    print(f"\n  discovered {len(issues)} issues across {analysed} traces analysed")
    for issue in issues:
        print(f"    ! {field_of(issue, 'title', 'description', 'summary')[:64]}")
    if summary := getattr(discovery, "summary", None):
        print(f"    summary: {' '.join(str(summary).split())[:110]}")
    if not issues:
        # "0 issues" is NOT proof the agent is clean. Issue discovery is itself
        # LLM-judged, and individual judge calls can fail -- MLflow logs
        # "Some scorer invocations failed during evaluation" as a WARNING and
        # carries on, so a partial failure looks exactly like a clean result
        # from here. Read the warnings above, and open the triage run below to
        # see the per-trace assessments before believing the zero.
        print("    no issues reported -- check the log above for scorer failures")
        print(f"    triage run: {getattr(discovery, 'triage_run_id', 'n/a')}")

    # ── 4. Promote to a versioned dataset ───────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 4: promote what was found into a versioned dataset")
    print("=" * 70)
    dataset = mlflow.genai.create_dataset(
        name=DATASET_NAME,
        experiment_id=EXPERIMENT_ID,
        tags={"source": "test_agent + simulator", "lesson": "L2-M2.1.1"},
    )
    records: list[dict[str, Any]] = [
        {
            "inputs": {"messages": [{"role": "user", "content": tc.input}]},
            "expectations": {"contains": tc.expected_output, "tools": tc.expected_tools},
        }
        for tc in HAND_WRITTEN
    ]
    records += [
        {"inputs": {"messages": [{"role": "user", "content": field_of(case, "goal")}]}, "expectations": {}}
        for case in result.test_cases
    ]
    dataset.merge_records(records)
    print(f"  dataset '{dataset.name}' (id {dataset.dataset_id}) holds {len(records)} records")
    print("  Later lessons load it by name, so the regression suite accumulates")
    print("  instead of being rewritten once per lesson.")

    # ── 5. Record the run ───────────────────────────────────────────────────── #
    with mlflow.start_run(run_name="agent_test_generation"):
        mlflow.log_params(
            {
                "model": MODEL_ALIAS,
                "hand_written_cases": len(HAND_WRITTEN),
                "generated_cases": len(result.test_cases),
                "max_turns": MAX_TURNS,
            }
        )
        mlflow.log_metrics(
            {
                "hand_written_pass_rate": hand_pass / len(HAND_WRITTEN),
                "simulated_conversations": len(sim_traces),
                "issues_found": len(issues),
                "dataset_records": len(records),
            }
        )

    print("\n" + "=" * 70)
    print(f"  hand-written cases : {len(HAND_WRITTEN):>2}  (a human wrote them, single-turn)")
    print(f"  generated cases    : {len(result.test_cases):>2}  (test_agent wrote them, multi-turn)")
    print(f"  issues discovered  : {len(issues):>2}")
    print(f"  dataset records    : {len(records):>2}")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
