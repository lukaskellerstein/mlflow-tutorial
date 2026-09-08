"""L2-M2.3.2.1 -- Online Session Scoring.

`1_turn/1_online_scoring` samples production TRACES and judges each one on its
own. This lesson samples production SESSIONS and judges the whole conversation.

The difference is not cosmetic. A support agent whose every individual reply
scores well can still be failing, and only the session view can say so:

    turn-level online   "was this reply good?"          -> answered per trace
    session-level online "did this customer get helped?" -> answered per session

MLflow supports both on the same machinery. `Scorer.is_session_level_scorer`
picks the path, and the server's sampler branches on it -- the scorer is handed
a list of traces sharing one session id instead of a single trace.

Two traps, both of which cost real time:

  1. `.start()` demands a GATEWAY model. A scorer built with
     `model="openai:/gemma-judge"` registers happily and then fails to start
     with "does not use a gateway model". The judge runs INSIDE the MLflow
     server, which has neither your base URL nor your key.
  2. `delete_scorer(name=...)` is not enough -- it raises asking for a version.
     Pass `version="all"`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# The MLflow AI Gateway -- the tracking server itself, not a provider
# directly. It serves the AGENT here and the JUDGE server-side, from the
# same alias list in infra/mlflow/gateway/seed_gateway.py.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
JUDGE_ALIAS = "gemma-judge"

EXPERIMENT = "L2/M2_agent_evaluation/3_online/2_conversation/1_online_session_scoring"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

# Without traces there is nothing to sample, and the scorer would sit STARTED
# and score nothing forever.
mlflow.langchain.autolog(log_traces=True)

# The judge runs INSIDE the MLflow server, so it names a gateway ENDPOINT
# rather than a base URL. `openai:/gemma-judge` would register happily and
# then fail to start: an `openai:/` model is resolved client-side, and the
# server has no client. See check_judge_endpoint() below.

SESSION_SCORER_NAME = "production_conversation_completeness"

ORDERS = {
    "A1001": "shipped, arriving Thursday",
    "A1002": "held at the warehouse, payment not confirmed",
}
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


SYSTEM_PROMPT = (
    "You are a retail support agent. Use the tools to answer questions about orders and policy. "
    "Never invent an order status or a policy. Keep replies under three sentences."
)

llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
AGENT = create_agent(llm, tools=[order_status, policy_lookup], system_prompt=SYSTEM_PROMPT)

# Stand-in production traffic. Each entry is one customer's whole conversation.
LIVE_CONVERSATIONS = [
    ["What is the status of order A1001?", "And when exactly does that arrive?"],
    ["My item arrived broken.", "Who pays to send it back?", "Remind me the policy code?"],
]


def check_judge_endpoint() -> str:
    """Confirm the server holds the endpoint the judge will name.

    A judge started with `.start()` runs INSIDE the MLflow server: it samples its
    own traces on its own schedule, long after this script has exited. So it
    cannot borrow the base URL or key the agent above uses -- it needs a model
    the server itself owns, named `gateway:/<endpoint>`.

    Nothing has to be built here. infra/mlflow/gateway/seed_gateway.py defines
    `gemma-judge` as a gateway endpoint and the stack seeds it on every
    `podman compose up -d`. This is a check, because a missing endpoint would
    otherwise surface hours later as a SCORER_ERROR on a sampled trace, with
    nothing to say why.
    """
    from mlflow.tracking._tracking_service.utils import _get_store

    names = {e.name for e in _get_store().list_gateway_endpoints()}
    if JUDGE_ALIAS not in names:
        raise SystemExit(
            f"The gateway has no endpoint named '{JUDGE_ALIAS}'.\n"
            "Seed it:  cd infra && podman compose up -d\n"
            "Then read what it built:  podman compose logs mlflow-seed"
        )
    print(f"  gateway endpoint '{JUDGE_ALIAS}' is present")
    return JUDGE_ALIAS


@mlflow.trace(name="support_turn")
def run_turn(history: list[Any], session_id: str) -> dict:
    """One production turn, stamped with the session id that groups it."""
    mlflow.update_current_trace(session_id=session_id)
    return AGENT.invoke(cast(Any, {"messages": history}))


def generate_traffic() -> list[str]:
    """Serve the stand-in customers, one session per conversation."""
    session_ids: list[str] = []
    for convo in LIVE_CONVERSATIONS:
        session_id = f"prod-{uuid.uuid4().hex[:8]}"
        session_ids.append(session_id)
        history: list[Any] = []
        for turn in convo:
            history.append({"role": "user", "content": turn})
            history = run_turn(history, session_id)["messages"]
        print(f"    session {session_id}  {len(convo)} turns")
    mlflow.flush_trace_async_logging()
    return session_ids


def register_session_scorer(endpoint: str) -> Any:
    """Register a SESSION-level scorer and start it sampling live sessions."""
    from mlflow.genai.scorers import ConversationCompleteness, ScorerSamplingConfig

    # Built through a local alias so basedpyright does not report "No parameter
    # named model" -- `model` is a real pydantic field, but the checker resolves
    # a narrower __init__ than pydantic synthesises.
    scorer_cls: Any = ConversationCompleteness
    # gateway:/ not openai:/ -- the judge runs inside the MLflow server.
    scorer = scorer_cls(name=SESSION_SCORER_NAME, model=f"gateway:/{endpoint}")
    print(f"  kind          : {scorer.kind}")
    print(f"  session-level : {scorer.is_session_level_scorer}")

    registered = scorer.register(name=SESSION_SCORER_NAME)
    print(f"  registered    : {registered.name}")

    # sample_rate matters MORE here than at turn level. A turn judge reads one
    # trace; a session judge reads the whole conversation, so its prompt grows
    # with conversation length and its cost per sampled unit is higher.
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
    print(f"  started       : status={started.status} sample_rate={started.sample_rate}")
    return started


def show_server_state() -> None:
    """Read the scorer back from the server, not from the local object."""
    from mlflow.genai.scorers import list_scorers

    for scorer in list_scorers():
        if scorer.name == SESSION_SCORER_NAME:
            print(f"  server-side: name={scorer.name} session_level={scorer.is_session_level_scorer}")
            print(
                f"              status={getattr(scorer, 'status', '?')} sample_rate={getattr(scorer, 'sample_rate', '?')}"
            )
            return
    print("  scorer not found on the server")


def cleanup() -> None:
    """Stop and remove the scorer so re-running the lesson starts clean."""
    from mlflow.genai.scorers import delete_scorer, list_scorers

    for scorer in list_scorers():
        if scorer.name != SESSION_SCORER_NAME:
            continue
        try:
            scorer.stop()
        except Exception as exc:
            print(f"  stop skipped: {str(exc)[:60]}")
        # `version` is REQUIRED. delete_scorer(name=...) alone raises
        # "You must set `version` argument to either an integer or 'all'".
        delete_scorer(name=SESSION_SCORER_NAME, version="all")
        print(f"  deleted '{SESSION_SCORER_NAME}'")
        return


def main() -> None:
    print("=" * 70)
    print("  L2-M2.3.2.1 -- Online Session Scoring")
    print("=" * 70)

    cleanup()  # start from a known state

    print("\nStep 1: the gateway endpoint the judge will run on")
    endpoint = check_judge_endpoint()

    print("\nStep 2: register a SESSION-level scorer and start it")
    register_session_scorer(endpoint)

    print(f"\nStep 3: serve {len(LIVE_CONVERSATIONS)} customers (the 'production' traffic)")
    session_ids = generate_traffic()

    print("\nStep 4: what the server holds")
    show_server_state()

    with mlflow.start_run(run_name="online_session_scoring"):
        mlflow.log_params(
            {
                "scorer": SESSION_SCORER_NAME,
                "endpoint": endpoint,
                "sample_rate": 1.0,
                "sessions_served": len(session_ids),
            }
        )
        mlflow.log_metric("sessions_served", len(session_ids))

    print("\nStep 5: scoring happens SERVER-SIDE, on the server's own schedule")
    print("  Sampling is asynchronous. The assessments appear on the session in")
    print("  the MLflow UI once the server has picked the sessions up -- there is")
    print("  no local call to await, which is the whole point of online scoring.")
    time.sleep(5)
    print("  waited 5s; check the UI rather than expecting output here")

    print(f"\n{'=' * 70}\n  Turn-level online vs session-level online\n{'=' * 70}")
    print("  Same registration, same start(), same sampling machinery.")
    print("  What changes is the UNIT the server hands the judge:")
    print("    turn-level    -> one trace          'was this reply good?'")
    print("    session-level -> a list of traces   'did this customer get helped?'")
    print("\n  And the economics change with it. A session judge reads every turn")
    print("  of a conversation, so its prompt -- and its cost -- grow with")
    print("  conversation length. Sample lower here than you would at turn level.")

    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("  Leaving the scorer STARTED so you can watch it work. Re-running")
    print("  this lesson cleans it up first.")
    print("=" * 70)


if __name__ == "__main__":
    main()
