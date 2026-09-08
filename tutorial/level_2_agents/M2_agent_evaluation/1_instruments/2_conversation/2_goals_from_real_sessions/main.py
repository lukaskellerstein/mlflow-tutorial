"""L2-M2.1.2.2 -- Goals and Personas Distilled from Real Sessions.

1_conversation_simulation ran the engine forward: a goal you wrote became a
conversation. That does not scale. Every scenario still costs a person the time
to imagine it, and people imagine the paths they already thought of.

`generate_test_cases()` runs the other way. It reads conversations that already
happened and infers, for each one, the goal the user was pursuing and the
persona they were pursuing it with. Nobody writes them.

Note what does and does not reverse here. The SIMULATOR only ever runs forward:

    goal --simulate--> conversation          always this direction

What this lesson adds is a second way to obtain the goal:

    conversations --generate_test_cases()--> goals --simulate--> conversations

So "backward" describes where the goal came from, not a second execution mode.
Point step 2 at real production sessions instead of the seeded ones and your
users have written your test suite.
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.genai.simulators import ConversationSimulator, generate_test_cases
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
SIMULATED_USER_ALIAS = "gemma-judge"

# The simulator and the distiller both resolve their model through the litellm LIBRARY
# (which MLflow uses internally -- not the proxy this repo used to run),
# which reads these. Assignments, not setdefault: a real OPENAI_API_KEY in the
# environment would win and every call would be rejected by the gateway with
# "Invalid proxy server token passed", a long way from its cause.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

SIM_MODEL = f"openai:/{SIMULATED_USER_ALIAS}"

# The distiller is the DENSER local model, and the reason is not "bigger is
# better".
#
# `generate_test_cases()` asks for STRUCTURED OUTPUT: it sends a pydantic schema
# as `response_format`, then calls `_GoalAndPersona.model_validate_json()` on
# whatever comes back. That parse is unforgiving -- MLflow's JUDGE path runs
# `_strip_markdown_code_blocks` before json.loads, but the DISTILLATION path
# calls `model_validate_json()` straight on the raw string. A reply that is
# fenced, truncated or merely chatty drops every session, and the failure is
# logged at DEBUG -- an empty list and no visible error. So watch the count this
# lesson prints: zero goals is the symptom, not a crash.
#
# HISTORY, because this line used to name a cloud alias: under LiteLLM +
# LMStudio the gateway STRIPPED `response_format` from the local deployments
# (`additional_drop_params`), because LMStudio honoured a schema by compiling a
# decoding grammar that masks EOS and a model that missed a closing quote
# rambled to max_tokens. Local structured output could not work, so the
# distiller escaped to OpenRouter. Both halves of that are gone: the MLflow AI
# Gateway forwards every parameter as sent, Unsloth honours a schema and returns
# clean JSON with `finish_reason: stop`, and there is no hosted provider left in
# the stack to escape to.
DISTILL_ALIAS = "gemma-31b-local"
DISTILL_MODEL = f"openai:/{DISTILL_ALIAS}"

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/2_conversation/2_goals_from_real_sessions"

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

# The seed stands in for yesterday's production traffic. In a real system you
# would skip step 1 entirely and search the experiment your agent already logs
# to. It is here so the lesson runs on a fresh machine with an empty server.
SEED_SCENARIOS: list[dict[str, Any]] = [
    {
        "goal": "Find out whether order A1002 will arrive this week, and why it is delayed",
        "persona": "An impatient customer who asks short, blunt follow-up questions",
    },
    {
        "goal": "Return a faulty item bought two months ago and find out who pays return shipping",
        "persona": "A polite first-time customer who does not know the policy names",
    },
]


def predict_fn(input: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:  # noqa: A002 - the parameter name is the contract
    """The simulator contract: `input` XOR `messages` in, a readable reply out."""
    # `invoke` is typed for langchain's InputAgentState, which lives at
    # langchain.agents.middleware.types -- too deep an import to put in a lesson
    # for one annotation. A message dict is what it actually wants.
    return AGENT.invoke(cast(Any, {"messages": input}))


def messages_of(payload: Any) -> list[Any]:
    """Read the message list out of a trace's request or response payload."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        return payload.get("messages") or payload.get("input") or []
    return payload if isinstance(payload, list) else []


def last_message(messages: list[Any], *labels: str) -> str:
    """The last message whose role (or LangChain `type`) is one of `labels`."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in labels or msg.get("type") in labels:
            if content := msg.get("content"):
                return " ".join(str(content).split())
    return ""


def turn_texts(trace: Any) -> tuple[str, str]:
    """Pull THIS turn's question and reply out of one turn's trace."""
    asked = last_message(messages_of(trace.data.request), "user", "human")
    replied = last_message(messages_of(trace.data.response), "assistant", "ai")
    return asked, replied


def short(value: Any, width: int = 80) -> str:
    """Collapse whitespace and trim, so a multi-line LLM answer stays one line."""
    return " ".join(str(value or "").split())[:width]


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.2.2  Goals and Personas Distilled from Real Sessions")
    print("=" * 70)

    # -- 1. Stand in for existing traffic ------------------------------------ #
    print("\nStep 1: seed some conversations to stand in for yesterday's traffic")
    print("  (in production you skip this and search your live experiment)")
    seeder = ConversationSimulator(test_cases=SEED_SCENARIOS, max_turns=3, user_model=SIM_MODEL)
    with mlflow.start_run(run_name="seed_traffic"):
        seed_traces = seeder.simulate(predict_fn)
        mlflow.log_params({"model": MODEL_ALIAS, "user_model": SIMULATED_USER_ALIAS})
        mlflow.log_metrics({"seeded_conversations": len(seed_traces)})
    print(f"  seeded {len(seed_traces)} conversations")

    # -- 2. Read them back as sessions --------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 2: search_sessions() -- one conversation, one session")
    print("=" * 70)
    sessions = mlflow.search_sessions(locations=[EXPERIMENT_ID], max_results=len(SEED_SCENARIOS))
    for session in sessions:
        print(f"  session {str(session.id)[:16]:<16} {len(session)} traces")
    print("\n  This is the only input generate_test_cases() needs. It never sees")
    print("  the agent, the tools, or the goal that produced these.")

    # -- 3. Distil ----------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 3: generate_test_cases() -- sessions back into goals")
    print("=" * 70)
    print(f"  one LLM call per session via {DISTILL_ALIAS}, inferring goal and persona...")
    distilled = generate_test_cases(sessions, model=DISTILL_MODEL)

    # COUNT FIRST, then read. Distillation is an LLM call per session, and a
    # session whose answer will not parse is dropped -- so the function returns a
    # SHORTER list, never an error. Reading the contents without checking the
    # length is how "it produced nothing" reads as "it produced nothing wrong".
    print(f"\n  {len(distilled)} of {len(sessions)} sessions produced a case")
    for case in distilled:
        print(f"\n  goal     : {short(case.get('goal'))}")
        print(f"  persona  : {short(case.get('persona'))}")
        if guides := case.get("simulation_guidelines"):
            print(f"  guides   : {short(guides)}")

    print("\n  Every key above is one the simulator accepts directly. `persona` is")
    print("  the ONE user inferred from that conversation, not a participant list.")

    if not distilled:
        print("\n  Every session was dropped. That is not a crash and not a bug report --")
        print("  it is what a silent LLM failure looks like from the caller's side.")

    # -- 4. Close the loop --------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 4: feed a distilled goal straight back into the simulator")
    print("=" * 70)
    replay_traces: list[list[Any]] = []
    if not distilled:
        print("  skipped -- step 3 returned no case to replay")
    else:
        replay = ConversationSimulator(test_cases=distilled[:1], max_turns=2, user_model=SIM_MODEL)
        with mlflow.start_run(run_name="simulate_distilled_goal"):
            replay_traces = replay.simulate(predict_fn)
            mlflow.log_params({"source": "generate_test_cases", "max_turns": 2})
            mlflow.log_metrics({"conversations": len(replay_traces), "turns_total": sum(len(t) for t in replay_traces)})

        for traces in replay_traces:
            for i, trace in enumerate(traces, 1):
                asked, replied = turn_texts(trace)
                print(f"  turn {i}  user  : {asked[:70]}")
                print(f"          agent : {replied[:70]}")

    print("\n" + "=" * 70)
    print(f"  sessions read            : {len(sessions)}")
    print(f"  goals distilled back out : {len(distilled)}")
    print(f"  replayed conversations   : {len(replay_traces)}")
    print("\n  Still nothing here is graded. Both lessons produce conversations;")
    print("  3_judging_conversations is the one that turns them into numbers.")
    print("\n  Next: 3_judging_conversations scores a whole conversation, then")
    print("  4_test_agent removes the last human input -- not even a goal.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
