"""L2-M2.1.2.1 -- Conversation Simulation: the engine.

A hand-written case (1_turn/1_hand_written_tests) is one question. A real user
has a GOAL, and pursues it over several turns, reacting to what the agent
actually said. Bugs that only appear at turn 3 are invisible to a suite of
single-turn cases.

`ConversationSimulator` replaces the question with a goal and a persona. A second
model plays the user, so the conversation is real: turn 3 only happens because of
what the agent said at turn 2.

Two things this lesson wants you to leave with:

  * The simulator has exactly TWO sides -- one simulated user and your agent.
    `persona` describes that single user. It is not a list of participants, and
    there is no multi-party mode.
  * MLflow calls the input `test_cases`, but a simulator test case is a
    SCENARIO, not an assertion. It holds a goal, not a right answer, so it
    cannot pass or fail. Something else has to judge the result -- that is
    3_judging_conversations.

This lesson only runs the engine forward: a goal you write becomes a
conversation. Where the goal comes from when you do NOT write it is
2_goals_from_real_sessions and 4_test_agent.
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
from mlflow.genai.simulators import ConversationSimulator
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
# The agent and the model playing the user are named separately on purpose. They
# resolve to the same model today, but "be the agent" and "be a difficult
# customer" are different jobs and will not always want the same one.
SIMULATED_USER_ALIAS = "gemma-judge"

# The simulator resolves its model through the litellm LIBRARY (which MLflow uses
# internally -- not the proxy this repo used to run), and it reads these.
# Assignments, not setdefault: a real OPENAI_API_KEY in the environment would win
# and every simulated turn would be rejected by the gateway with
# "Invalid proxy server token passed", a long way from its cause.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

SIM_MODEL = f"openai:/{SIMULATED_USER_ALIAS}"
EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/2_conversation/1_conversation_simulation"
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

llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
AGENT = create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)


def predict_fn(input: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:  # noqa: A002 - the parameter name is the contract
    """The contract the simulator drives.

    Three rules, all enforced by MLflow before the first turn runs:

    1. It takes `input` XOR `messages` -- the conversation so far, as message
       dicts. Both, or neither, raises.
    2. It returns something MLflow can pull an assistant reply out of: a plain
       string, an OpenAI `{"choices": [...]}` response, or LangGraph's native
       `{"messages": [...]}`. The third is what this agent already returns, so
       no adapter is needed.
    3. It receives `mlflow_session_id`, the same value for every turn of one
       conversation. This agent is stateless -- the whole history arrives in
       `input` each turn -- so it is ignored. A thread-based agent would key its
       memory on it.
    """
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
        # The simulator sends `input` on some turns and `messages` on others,
        # because predict_fn accepts either. Read whichever is present.
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
    """Pull THIS turn's question and reply out of one turn's trace.

    Both payloads carry the whole conversation so far, not just this turn, so
    printing them raw shows turn 1 three times over. Take the last message of
    each side instead.
    """
    asked = last_message(messages_of(trace.data.request), "user", "human")
    replied = last_message(messages_of(trace.data.response), "assistant", "ai")
    return asked, replied


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.2.1  Conversation Simulation -- the engine")
    print("=" * 70)

    # -- 1. Goals and personas, not questions -------------------------------- #
    print("\nStep 1: two scenarios -- a goal, ONE persona, and optional guidelines")
    scenarios: list[dict[str, Any]] = [
        {
            "goal": "Find out whether order A1002 will arrive this week, and why it is delayed",
            "persona": "An impatient customer who asks short, blunt follow-up questions",
        },
        {
            "goal": "Return a faulty item bought two months ago and find out who pays return shipping",
            "persona": "A polite first-time customer who does not know the policy names",
            # Guidelines constrain the simulated USER, not the agent. This one
            # forces the agent to surface the warranty itself instead of being
            # handed the word -- which is the behaviour actually under test.
            "simulation_guidelines": ["Do not say the word 'warranty' unless the agent says it first"],
        },
    ]
    for s in scenarios:
        print(f"  - goal    : {s['goal']}")
        print(f"    persona : {s['persona']}")
    print("\n  `goal` is the only required key. `persona`, `context`, `expectations`")
    print("  and `simulation_guidelines` are optional. `persona` is ONE user --")
    print("  the simulator has two sides, that user and your agent.")

    # -- 2. Simulate --------------------------------------------------------- #
    print(f"\nStep 2: simulate, up to {MAX_TURNS} turns each")
    print("  (a second model plays the user -- this takes a few minutes)")
    simulator = ConversationSimulator(test_cases=scenarios, max_turns=MAX_TURNS, user_model=SIM_MODEL)
    with mlflow.start_run(run_name="simulate_written_goals"):
        sim_traces = simulator.simulate(predict_fn)
        mlflow.log_params({"model": MODEL_ALIAS, "user_model": SIMULATED_USER_ALIAS, "max_turns": MAX_TURNS})
        mlflow.log_metrics(
            {
                "conversations": len(sim_traces),
                "turns_total": sum(len(t) for t in sim_traces),
            }
        )

    print(f"\n  simulated {len(sim_traces)} conversations")
    for scenario, traces in zip(scenarios, sim_traces):
        print(f"\n  goal: {str(scenario['goal'])[:60]}")
        for i, trace in enumerate(traces, 1):
            req, resp = turn_texts(trace)
            print(f"    turn {i}  user  : {req[:70]}")
            print(f"            agent : {resp[:70]}")

    # -- 3. What a session is ------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Step 3: one conversation = one session = several traces")
    print("=" * 70)
    sessions = mlflow.search_sessions(locations=[EXPERIMENT_ID], max_results=len(scenarios))
    for session in sessions:
        print(f"  session {str(session.id)[:16]:<16} {len(session)} traces")
    print("\n  Every turn is its own trace. The session id is what ties them")
    print("  together, and it is the unit a session-level scorer reads.")

    # -- 4. What this lesson did NOT do -------------------------------------- #
    print("\n" + "=" * 70)
    print("What the simulator did not do")
    print("=" * 70)
    print("  It did not grade anything. It asked an LLM 'has the goal been")
    print("  achieved?' only to know when to STOP a conversation -- that answer")
    print("  never becomes a score, and it never reaches a run.")
    print("\n  You probably saw warnings above reading 'Could not parse response")
    print("  for goal achievement check'. Read one: the JSON is valid, and the")
    print("  model simply wrapped it in a ```json fence that MLflow's parser does")
    print("  not strip. Because that check only decides when to STOP, a failed")
    print("  parse costs you nothing but extra turns -- the conversation runs to")
    print("  max_turns instead of ending early. Nothing scored is affected.")
    print("\n  A simulator test case holds a GOAL, not a right answer, so there is")
    print("  nothing to compare against. Judging what came out is a separate step.")

    print("\n" + "=" * 70)
    print(f"  conversations simulated : {len(sim_traces)}")
    print(f"  turns traced            : {sum(len(t) for t in sim_traces)}")
    print(f"  sessions                : {len(sessions)}")
    print("\n  Next: 2_goals_from_real_sessions runs the same engine, but the goal")
    print("  is distilled out of conversations that already happened.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
