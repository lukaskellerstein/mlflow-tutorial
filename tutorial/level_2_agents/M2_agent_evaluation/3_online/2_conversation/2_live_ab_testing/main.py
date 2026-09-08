"""L2-M2.3.2.2 -- A/B Testing on Live Sessions: Sticky Assignment.

1_turn/2_live_ab_testing split live REQUESTS between two versions and scored a
sample of the traces. This lesson splits live CONVERSATIONS, and one thing
changes that makes it a different problem rather than the same one bigger.

At turn scope, a bad assignment costs you a noisy data point.
At session scope, a bad assignment FABRICATES a data point.

Assign per request and a customer's turn 1 is served by v1 while their turn 2
is served by v2. The conversation that results was produced by NEITHER version.
Score it and you have added to your results a measurement of something that
does not exist -- and it is counted against whichever arm you happen to tag it
with. That is worse than noise: noise averages out, and this does not.

So the variant is chosen ONCE, when the session opens, and every turn of that
session is served and tagged with it. That is what "sticky" means, and it is
the whole lesson.

Two consequences worth budgeting for, both in the closing notes:
  - a session judge reads every turn, so cost grows with conversation length
  - a session cannot be judged until it is over, so results arrive later
"""

from __future__ import annotations

import hashlib
import statistics
import time
import uuid
from collections import defaultdict
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

EXPERIMENT = "L2/M2_agent_evaluation/3_online/2_conversation/2_live_ab_testing"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog(log_traces=True)

# The judge runs INSIDE the MLflow server, so it names a gateway ENDPOINT
# rather than a base URL. `openai:/gemma-judge` would register happily and
# then fail to start: an `openai:/` model is resolved client-side, and the
# server has no client. See check_judge_endpoint() below.
SESSION_JUDGE_NAME = "ab_conversation_completeness"

KNOWLEDGE = {
    "returns": "Returns are accepted within 30 days of delivery (P-101).",
    "warranty": "The warranty covers manufacturing defects for 12 months (P-204).",
    "shipping": "Return shipping is free for faulty items (P-330).",
}


@tool
def lookup_policy(topic: str) -> str:
    """Look up store policy. Topics: returns, warranty, shipping."""
    return KNOWLEDGE.get(topic.strip().lower(), f"No policy on file for '{topic}'.")


VARIANTS = {
    "v1": "You are a support agent. Answer using the tool. Keep it short.",
    "v2": (
        "You are a support agent. Answer using the tool. Keep it short, and before "
        "you finish, check the customer has everything they asked for."
    ),
}

# Stand-in production traffic: whole conversations, one per customer.
LIVE_SESSIONS = [
    ("cust-1", ["How long do I have to return something?", "And if the item is faulty?"]),
    ("cust-2", ["What does the warranty cover?", "How long does that last?"]),
    ("cust-3", ["Is return shipping free?", "Even after 20 days?"]),
    ("cust-4", ["Can I return an opened item?", "What about the warranty then?"]),
]


def assign_variant(session_id: str) -> str:
    """Bucket a SESSION -- not a request -- and do it stably.

    Built-in hash() is salted per process, so it would reassign sessions after
    every restart. md5 here is a stability choice, not a security one.
    """
    digest = hashlib.md5(session_id.encode(), usedforsecurity=False).hexdigest()
    return "v1" if int(digest, 16) % 2 == 0 else "v2"


def build_agent(prompt: str) -> Any:
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=[lookup_policy], system_prompt=prompt)


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


def cleanup() -> None:
    from mlflow.genai.scorers import delete_scorer, list_scorers

    for scorer in list_scorers():
        if scorer.name != SESSION_JUDGE_NAME:
            continue
        try:
            scorer.stop()
        except Exception:
            pass
        delete_scorer(name=SESSION_JUDGE_NAME, version="all")  # `version` is required
        print(f"  removed previous '{SESSION_JUDGE_NAME}'")
        return


def start_session_judge(endpoint: str) -> Any:
    """ONE session-level judge for BOTH arms."""
    from mlflow.genai.scorers import ConversationCompleteness, ScorerSamplingConfig

    scorer_cls: Any = ConversationCompleteness  # `model` is a pydantic field the checker cannot see
    scorer = scorer_cls(name=SESSION_JUDGE_NAME, model=f"gateway:/{endpoint}")
    print(f"  session-level : {scorer.is_session_level_scorer}")
    registered = scorer.register(name=SESSION_JUDGE_NAME)
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
    print(f"  started       : status={started.status} sample_rate={started.sample_rate}")
    return started


@mlflow.trace(name="support_turn")
def run_turn(agent: Any, history: list[Any], session_id: str, variant: str) -> dict[str, Any]:
    """One turn. The session id groups it; the variant tag is the same all session."""
    mlflow.update_current_trace(session_id=session_id, tags={"variant": variant})
    return agent.invoke(cast(Any, {"messages": history}))


def serve_session(agents: dict[str, Any], customer: str, turns: list[str]) -> tuple[str, str]:
    """Serve one whole conversation on ONE arm, chosen before the first turn."""
    session_id = f"{customer}-{uuid.uuid4().hex[:6]}"
    variant = assign_variant(session_id)  # decided ONCE, here
    history: list[Any] = []
    for turn in turns:
        history.append({"role": "user", "content": turn})
        history = run_turn(agents[variant], history, session_id, variant)["messages"]
    return session_id, variant


def read_session_scores() -> dict[str, list[float]]:
    """Group sampled traces into sessions, then split the session scores by arm."""
    traces = mlflow.search_traces(
        locations=[EXPERIMENT_ID], max_results=200, return_type="list", order_by=["timestamp DESC"]
    )
    by_variant: dict[str, list[float]] = defaultdict(list)
    seen_sessions: set[str] = set()
    for trace in traces:
        tags = trace.info.tags or {}
        variant = tags.get("variant")
        session_id = tags.get("mlflow.trace.session")
        if not variant or not session_id or session_id in seen_sessions:
            continue
        for assessment in trace.info.assessments or []:
            if assessment.name != SESSION_JUDGE_NAME:
                continue
            value = getattr(getattr(assessment, "feedback", None), "value", None)
            score = 1.0 if (value is True or str(value).strip().lower() == "yes") else 0.0
            by_variant[variant].append(score)
            seen_sessions.add(session_id)
            break
    return dict(by_variant)


def main() -> None:
    print("=" * 76)
    print("  L2-M2.3.2.2 -- A/B Testing on Live Sessions: Sticky Assignment")
    print("=" * 76)

    cleanup()
    print("\nStep 1: the gateway endpoint the judge runs on")
    endpoint = check_judge_endpoint()

    print("\nStep 2: register ONE session-level judge for both arms")
    start_session_judge(endpoint)

    print(f"\nStep 3: serve {len(LIVE_SESSIONS)} whole conversations")
    agents = {label: build_agent(prompt) for label, prompt in VARIANTS.items()}
    with mlflow.start_run(run_name=f"session_ab_{uuid.uuid4().hex[:6]}"):
        mlflow.log_params(
            {
                "model": MODEL_ALIAS,
                "judge": SESSION_JUDGE_NAME,
                "arms": ",".join(VARIANTS),
                "sessions": len(LIVE_SESSIONS),
            }
        )
        assigned: list[tuple[str, str, int]] = []
        for customer, turns in LIVE_SESSIONS:
            session_id, variant = serve_session(agents, customer, turns)
            assigned.append((session_id, variant, len(turns)))
            print(f"    {customer}  session {session_id[-6:]}  arm {variant}  {len(turns)} turns (all on {variant})")
        mlflow.flush_trace_async_logging()

        print("\n  Every turn of a session went to one arm. That is the invariant.")

        print("\nStep 4: wait for server-side session scoring, then split by arm")
        scores: dict[str, list[float]] = {}
        for attempt in range(1, 13):
            time.sleep(10)
            scores = read_session_scores()
            got = sum(len(v) for v in scores.values())
            print(f"    poll {attempt:>2}: {got} scored session(s)")
            if got >= len(LIVE_SESSIONS):
                break

        if not scores:
            print("\n  No session assessments yet. The judge is STARTED and every trace")
            print("  is tagged -- a session simply takes longer to score than a trace,")
            print("  because it cannot be judged until it looks finished. Check the UI.")
        else:
            print(f"\n  {'arm':<6} {'sessions':>9} {'mean':>7}")
            print(f"  {'-' * 6} {'-' * 9} {'-' * 7}")
            for arm in sorted(scores):
                mean = statistics.mean(scores[arm])
                print(f"  {arm:<6} {len(scores[arm]):>9} {mean:>7.2f}")
                mlflow.log_metrics({f"{arm}_mean": mean, f"{arm}_sessions": len(scores[arm])})

    print(f"\n{'=' * 76}\n  Why assignment must be sticky\n{'=' * 76}")
    print("  Assign per REQUEST instead of per session and a customer's turn 1 is")
    print("  served by v1 while their turn 2 is served by v2. The conversation you")
    print("  then score was produced by NEITHER version.")
    print("\n  That is not noise. Noise averages out over enough traffic; a")
    print("  fabricated conversation does not -- it is a measurement of a system")
    print("  you never shipped, counted against whichever arm you tagged it with.")
    print("\n  Turn-scope A/B does not have this failure mode, which is exactly why")
    print("  it is a different lesson and not the same one at a larger size.")

    print(f"\n{'=' * 76}\n  Two costs to budget for\n{'=' * 76}")
    print("  COST PER UNIT. A turn judge reads one trace. A session judge reads")
    print("  every turn, so its prompt grows with conversation length. Sample")
    print("  lower here than you would at turn level.")
    print("\n  TIME TO SIGNAL. A trace can be judged the moment it closes; a session")
    print("  cannot be judged until it looks finished. Turn A/B tells you a reply")
    print("  got worse within minutes. Session A/B tells you customers stopped")
    print("  being helped -- and only after they have stopped talking.")
    print("\n  Run both arms of both. They fail at different times and neither")
    print("  substitutes for the other.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 76)


if __name__ == "__main__":
    main()
