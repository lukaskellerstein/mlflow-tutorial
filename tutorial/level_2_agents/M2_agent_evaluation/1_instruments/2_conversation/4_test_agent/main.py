"""L2-M2.1.2.4 -- test_agent(): the Whole Pipeline in One Call.

L2-M2.1.1.1 needed a person to write every case. L2-M2.1.2.1 needed a person to write
every goal, or existing traffic to distil goals from. `mlflow.genai.test_agent()`
needs neither: it asks the agent to describe itself and writes the goals from
that description.

Four stages, one call:

  1. describe   -- the agent is asked what it does and what it refuses
  2. generate   -- test cases are written from that description
  3. simulate   -- each case is run as a multi-turn conversation
  4. discover   -- the resulting traces are LLM-judged for issues

Stage 1 is the one that makes this different from a random case generator. The
cases probe the agent's CLAIMED contract, which is why they reliably find the
boundaries -- unknown ids, out-of-scope questions, data it should refuse.
"""

from __future__ import annotations

import os
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
# test_agent uses ONE model for all four stages: describing, generating, playing
# the simulated user, and judging the traces afterwards. One knob, four jobs --
# and on this stack no single alias does all four well. Read this before you
# believe stage 1's output.
#
# STAGE 1 CAN DEGRADE ON THE LOCAL ALIAS, AND IT FAILS QUIETLY.
# `_describe_agent_from_response` calls get_chat_completions_with_structured_output,
# which json.loads the reply. Under the old LiteLLM + LMStudio stack the gateway
# STRIPPED `response_format` from the local deployments on purpose, so the parse
# always raised. That stripping is gone -- the MLflow AI Gateway forwards every
# parameter as sent and Unsloth honours a schema properly -- but any reply that
# does not parse still lands the same way: MLflow logs "Failed to describe agent
# from self-description" and falls back to its hardcoded stub:
#
#     "A conversational AI agent" / capabilities: ["general conversation"]
#
# The run still completes. That is the trap: every generated case then derives
# from that stub rather than from YOUR agent, and the describe stage -- the thing
# this lesson is about -- silently did nothing. Watch for that WARNING.
#
# THERE IS NO CLOUD ESCAPE HATCH ANY MORE, and that is deliberate. This comment
# used to point at `gemma-31b-free` and `gpt-mini`; every hosted alias is gone
# from the gateway, so the only knob left is which LOCAL model plays the tester.
#
# `gemma-31b-local` is the alternative worth trying if stage 1 comes back as the
# stub: it is the denser model and it does the structured-output job in
# L2-M2.1.2.2. It costs an auto-switch on every call this lesson makes, because
# the agent under test runs on the 26B and Unsloth holds one model at a time.
TESTER_ALIAS = "gemma-judge"

# test_agent resolves its model through the litellm LIBRARY (which MLflow uses
# internally -- not the proxy this repo used to run), and it reads these. Assignments,
# not setdefault: a real OPENAI_API_KEY in the environment would win and every
# call would be rejected by the gateway with "Invalid proxy server token
# passed", a long way from its cause.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

TESTER_MODEL = f"openai:/{TESTER_ALIAS}"
EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/2_conversation/4_test_agent"
NUM_TEST_CASES = 3
MAX_TURNS = 3
MAX_ISSUES = 5

# What MLflow returns when stage 1 gives up. Copied from
# mlflow.genai.agent_tester._resolve_agent_description, which falls back to
# description="A conversational AI agent" after both the self-description and
# the trace-based attempts fail. There is no flag on the result that says so.
PLACEHOLDER_DESCRIPTION = "A conversational AI agent"

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

# What a person wrote in L2-M2.1.1.1, for comparison at the end. Nothing imports
# it -- every lesson is a standalone leaf, so it is copied.
HAND_WRITTEN_TOPICS = [
    "status of a known order",
    "delivery of a known order",
    "why a known order is held",
    "the return window",
    "the warranty length",
    "who pays return shipping",
]


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


def predict_fn(input: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:  # noqa: A002 - the parameter name is the contract
    """The same contract the simulator drives in L2-M2.1.2.1: `input` xor
    `messages`, and a return shape MLflow can read a reply out of."""
    # `invoke` is typed for langchain's InputAgentState, which lives at
    # langchain.agents.middleware.types -- too deep an import to put in a lesson
    # for one annotation. A message dict is what it actually wants.
    return AGENT.invoke(cast(Any, {"messages": input}))


def field_of(item: Any, *names: str) -> str:
    """Read the first present field from a generated case or a discovered issue.

    `test_agent` hands generated cases back as plain DICTS, not the pydantic
    models its own source defines -- so `case.goal` raises AttributeError while
    `case["goal"]` works. Issues arrive as objects on some paths. Reading either
    shape keeps this lesson working across both.
    """
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value:
            return " ".join(str(value).split())
    return " ".join(str(item).split())


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.2.4  test_agent() -- the Whole Pipeline in One Call")
    print("=" * 70)

    # -- 1. One call, four stages -------------------------------------------- #
    print(f"\nStep 1: test_agent() -- {NUM_TEST_CASES} cases, up to {MAX_TURNS} turns each")
    print("  describing the agent, generating cases, simulating, judging...")
    print("  (expect several minutes; it has not hung)")

    result = mlflow.genai.test_agent(
        predict_fn,
        model=TESTER_MODEL,
        num_test_cases=NUM_TEST_CASES,
        max_turns=MAX_TURNS,
        max_issues=MAX_ISSUES,
        # Guidance steers stage 2. Without it the generator covers a broad mix
        # of what the agent says it can do; with it, you aim at what you suspect.
        guidance="Focus on order ids that do not exist, and policy questions the tools cannot answer.",
    )

    # -- 2. Stage 1: what the agent says it is ------------------------------- #
    print("\n" + "=" * 70)
    print("Step 2: the self-description every case is derived from")
    print("=" * 70)
    description = " ".join(str(result.agent_description).split())
    for i in range(0, min(len(description), 400), 100):
        print(f"  {description[i : i + 100]}")

    # Stage 1 has three tiers and only the first one is the one this lesson is
    # about: ask the agent, else read existing traces, else return the constant
    # below. All three produce an AgentDescription, and only a WARNING in the log
    # distinguishes them -- so a run where the agent never described itself looks
    # identical here to one where it did.
    if described_itself := PLACEHOLDER_DESCRIPTION not in description:
        print("\n  The agent described itself, so the cases below come from its own words.")
    else:
        print(f"\n  !! That is MLflow's PLACEHOLDER: '{PLACEHOLDER_DESCRIPTION}'.")
        print("     Stage 1 failed and was swallowed -- look for the WARNING")
        print("     'Failed to describe agent from self-description' above.")
        print("     A local model that answers with the wrong JSON keys does this.")
        print("     The cases below were therefore NOT derived from this agent;")
        print("     whatever quality they have came from `guidance` instead.")

    # -- 3. Stage 2: the cases nobody wrote ---------------------------------- #
    print("\n" + "=" * 70)
    print(f"Step 3: {len(result.test_cases)} generated cases")
    print("=" * 70)
    for case in result.test_cases:
        print(f"\n  goal    : {field_of(case, 'goal')[:80]}")
        if persona := (case.get("persona") if isinstance(case, dict) else None):
            print(f"  persona : {' '.join(str(persona).split())[:80]}")

    # -- 4. Stages 3 and 4: traces, and what the judge made of them ---------- #
    print("\n" + "=" * 70)
    print("Step 4: issues discovered in the simulated traces")
    print("=" * 70)
    discovery = result.issues_result
    issues = list(getattr(discovery, "issues", None) or [])
    analysed = getattr(discovery, "total_traces_analyzed", 0)
    print(f"  {len(issues)} issues across {analysed} traces analysed")

    for issue in issues:
        print(f"\n  ! {field_of(issue, 'name', 'title', 'description')[:70]}")
        if severity := getattr(issue, "severity", None):
            print(f"    severity   : {severity}")
        if categories := getattr(issue, "categories", None):
            print(f"    categories : {', '.join(str(c) for c in categories)}")
        for cause in getattr(issue, "root_causes", None) or []:
            print(f"    root cause : {' '.join(str(cause).split())[:70]}")

    if summary := getattr(discovery, "summary", None):
        print(f"\n  summary: {' '.join(str(summary).split())[:150]}")

    triage_run_id = getattr(discovery, "triage_run_id", "n/a")
    if not issues:
        # "0 issues" is NOT proof the agent is clean. Discovery is itself
        # LLM-judged, and individual judge calls can fail -- MLflow logs
        # "Some scorer invocations failed during evaluation" as a WARNING and
        # carries on, so a partial failure looks exactly like a clean result
        # from here.
        print("\n  Zero issues. Before believing it, read the log above for scorer")
        print(f"  failures and open the triage run: {triage_run_id}")

    # -- 5. What a person would not have written ----------------------------- #
    print("\n" + "=" * 70)
    print("Step 5: generated versus hand-written")
    print("=" * 70)
    print("  L2-M2.1.1.1's six cases, by topic:")
    for topic in HAND_WRITTEN_TOPICS:
        print(f"    - {topic}")
    print("\n  Every one is a HAPPY PATH on data that exists. The generator aims")
    print("  at the edges instead: unknown ids, questions the tools cannot")
    print("  answer, data it must refuse.")

    # -- 6. Record the run --------------------------------------------------- #
    with mlflow.start_run(run_name="test_agent_generation"):
        mlflow.log_params(
            {
                "model": MODEL_ALIAS,
                "tester_model": TESTER_ALIAS,
                "num_test_cases": NUM_TEST_CASES,
                "max_turns": MAX_TURNS,
                "max_issues": MAX_ISSUES,
                # Log WHICH tier of stage 1 produced the description. Without
                # this the run record cannot tell a real self-description from
                # the placeholder, and neither can you, six months later.
                "self_description": "agent" if described_itself else "placeholder",
            }
        )
        mlflow.log_metrics(
            {
                "generated_cases": len(result.test_cases),
                "traces_analysed": analysed,
                "issues_found": len(issues),
                "described_itself": int(described_itself),
            }
        )
        mlflow.set_tag("triage_run_id", str(triage_run_id))

    print("\n" + "=" * 70)
    print(f"  generated cases  : {len(result.test_cases):>2}  (nobody wrote them)")
    print(f"  traces analysed  : {analysed:>2}")
    print(f"  issues found     : {len(issues):>2}")
    print(f"  self-description : {'the agent' if described_itself else 'PLACEHOLDER (stage 1 failed)'}")
    print("\n  test_agent logs its simulation and triage runs to ITS OWN")
    print("  experiment, not the one you set -- look for 'simulation-<id>' runs.")
    print("\n  Next: 3_dataset_store turns all of this -- hand-written cases, simulated")
    print("  goals and generated cases -- into one versioned dataset.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
