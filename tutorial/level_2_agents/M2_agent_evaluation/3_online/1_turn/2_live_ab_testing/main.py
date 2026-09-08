"""L2-M2.3.1.2 -- A/B Testing Two Versions on Live Traffic.

2_offline/1_turn/3_version_comparison compared v1 and v2 on a dataset you
chose. This lesson runs both versions AT THE SAME TIME against traffic nobody
chose, and that difference is not cosmetic:

  offline comparison   your cases, known answers, every case scored, paired
  live A/B             real questions, no ground truth, a SAMPLE scored,
                       and the two versions never see the same request

You cannot pair, because request 7 goes to exactly one variant. So you compare
distributions, and the thing that makes that legitimate is that assignment is
random with respect to the question -- each variant gets the same MIX of easy
and hard traffic, in expectation.

The mechanism is three lines of real work:

  1. assign each request to a variant, deterministically per user
  2. tag the trace with the variant it was served by
  3. let ONE registered judge score a sample of both, then split by tag

Point 3 is the part people get wrong. Two judges, or a judge retuned between
variants, and you are measuring the judges rather than the agents.
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

EXPERIMENT = "L2/M2_agent_evaluation/3_online/1_turn/2_live_ab_testing"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog(log_traces=True)

# The judge runs INSIDE the MLflow server, so it names a gateway ENDPOINT
# rather than a base URL. `openai:/gemma-judge` would register happily and
# then fail to start: an `openai:/` model is resolved client-side, and the
# server has no client. See check_judge_endpoint() below.
AB_JUDGE_NAME = "ab_answer_quality"

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
        "You are a support agent. Answer using the tool. Keep it short, and always "
        "quote the policy reference code (for example P-101) when you state a policy."
    ),
}

# Stand-in production traffic: a user id and whatever they asked. Note that the
# same user appears more than once -- which is what makes sticky assignment
# observable.
LIVE_TRAFFIC = [
    ("user-a", "How long do I have to return something?"),
    ("user-b", "What does the warranty cover?"),
    ("user-c", "Who pays return shipping on a faulty item?"),
    ("user-a", "And the warranty length?"),
    ("user-d", "Can I return after 40 days?"),
    ("user-b", "Is return shipping free?"),
]


def assign_variant(user_id: str) -> str:
    """Bucket a user into a variant, stably across processes and deploys.

    DO NOT USE THE BUILT-IN hash() FOR THIS. Python salts str hashing per
    process (PYTHONHASHSEED), so hash("user-a") % 2 gives a different answer
    after every restart -- users would silently flip variants between deploys,
    and your A/B result would be measuring the flipping rather than the change.
    md5 is not a security choice here, it is a STABILITY choice.
    """
    digest = hashlib.md5(user_id.encode(), usedforsecurity=False).hexdigest()
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
        if scorer.name != AB_JUDGE_NAME:
            continue
        try:
            scorer.stop()
        except Exception:
            pass
        # `version` is REQUIRED -- name alone raises.
        delete_scorer(name=AB_JUDGE_NAME, version="all")
        print(f"  removed previous '{AB_JUDGE_NAME}'")
        return


def start_judge(endpoint: str) -> Any:
    """ONE judge for BOTH variants. This is the part that must not be duplicated."""
    from mlflow.genai.scorers import ScorerSamplingConfig

    judge = mlflow.genai.make_judge(
        name=AB_JUDGE_NAME,
        instructions=(
            "You are reviewing a live support answer.\n"
            "The request is in {{ inputs }} and the agent's reply is in {{ outputs }}.\n"
            "Is the reply accurate, complete and genuinely useful? Answer true or false."
        ),
        model=f"gateway:/{endpoint}",
        feedback_value_type=bool,
    )
    registered = judge.register(name=AB_JUDGE_NAME)
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
    print(f"  one judge for both arms: status={started.status} sample_rate={started.sample_rate}")
    return started


@mlflow.trace(name="support_request")
def serve(agents: dict[str, Any], user_id: str, question: str) -> str:
    """Serve one live request, and TAG the trace with the arm that served it.

    The tag is the entire A/B mechanism. Without it the assessments arrive in
    one undifferentiated pile and the experiment is unrecoverable -- there is no
    way to work out afterwards which version produced which answer.
    """
    variant = assign_variant(user_id)
    mlflow.update_current_trace(tags={"variant": variant, "user_id": user_id})
    result = agents[variant].invoke(cast(Any, {"messages": [{"role": "user", "content": question}]}))
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") == "ai" and content and not getattr(msg, "tool_calls", None):
            return str(content)
    return ""


def read_assessments_by_variant() -> dict[str, list[float]]:
    """Pull the sampled traces back and split their scores by the variant tag."""
    traces = mlflow.search_traces(
        locations=[EXPERIMENT_ID], max_results=100, return_type="list", order_by=["timestamp DESC"]
    )
    by_variant: dict[str, list[float]] = defaultdict(list)
    for trace in traces:
        variant = (trace.info.tags or {}).get("variant")
        if not variant:
            continue
        for assessment in trace.info.assessments or []:
            if assessment.name != AB_JUDGE_NAME:
                continue
            value = getattr(getattr(assessment, "feedback", None), "value", None)
            if isinstance(value, bool):
                by_variant[variant].append(1.0 if value else 0.0)
    return dict(by_variant)


def main() -> None:
    print("=" * 76)
    print("  L2-M2.3.1.2 -- A/B Testing Two Versions on Live Traffic")
    print("=" * 76)

    cleanup()
    print("\nStep 1: the gateway endpoint the judge runs on")
    endpoint = check_judge_endpoint()

    print("\nStep 2: register ONE judge and start it sampling")
    start_judge(endpoint)

    print("\nStep 3: assignment -- stable per user, before any traffic")
    for user_id in sorted({u for u, _ in LIVE_TRAFFIC}):
        print(f"    {user_id} -> {assign_variant(user_id)}")
    print("\n  Same user, same arm, every time and after every restart. A user who")
    print("  flips arms mid-experiment pollutes both.")

    print(f"\nStep 4: serve {len(LIVE_TRAFFIC)} live requests")
    agents = {label: build_agent(prompt) for label, prompt in VARIANTS.items()}
    with mlflow.start_run(run_name=f"ab_{uuid.uuid4().hex[:6]}"):
        mlflow.log_params(
            {"model": MODEL_ALIAS, "judge": AB_JUDGE_NAME, "arms": ",".join(VARIANTS), "requests": len(LIVE_TRAFFIC)}
        )
        for user_id, question in LIVE_TRAFFIC:
            answer = serve(agents, user_id, question)
            print(f"    {user_id} [{assign_variant(user_id)}]  {question[:40]:<40} -> {answer[:34]}")
        mlflow.flush_trace_async_logging()

        print("\nStep 5: wait for the server to score the sample, then split by tag")
        print("  (scoring is server-side and asynchronous -- this is a poll, not a call)")
        scores: dict[str, list[float]] = {}
        for attempt in range(1, 13):
            time.sleep(10)
            scores = read_assessments_by_variant()
            got = sum(len(v) for v in scores.values())
            print(f"    poll {attempt:>2}: {got} scored trace(s)")
            if got >= len(LIVE_TRAFFIC):
                break

        if not scores:
            print("\n  No assessments yet. The scorer is STARTED and the traces are")
            print("  tagged -- the server simply has not caught up. Check the UI;")
            print("  nothing below would be wrong, just early.")
        else:
            print(f"\n  {'arm':<6} {'scored':>7} {'mean':>7}")
            print(f"  {'-' * 6} {'-' * 7} {'-' * 7}")
            for arm in sorted(scores):
                mean = statistics.mean(scores[arm])
                print(f"  {arm:<6} {len(scores[arm]):>7} {mean:>7.2f}")
                mlflow.log_metrics({f"{arm}_mean": mean, f"{arm}_n": len(scores[arm])})

    print(f"\n{'=' * 76}\n  Reading a live A/B honestly\n{'=' * 76}")
    print("  YOU CANNOT PAIR. Request 7 was served by exactly one arm, so there is")
    print("  no matched pair and no win/loss/tie. Compare distributions only.")
    print("\n  THE SAMPLE IS SMALL AND THE EFFECT IS SMALL. Six requests cannot")
    print("  separate two prompts. Real A/B tests run for days precisely because")
    print("  the traffic has to accumulate before the difference clears the noise.")
    print("\n  WHAT ONLINE BUYS YOU that offline cannot: the traffic MIX is real.")
    print("  Your offline suite contains the questions you thought of. This")
    print("  contains the questions people actually ask, in the proportions they")
    print("  actually ask them -- including the ones you would never have written.")
    print("\n  The scorer is left STARTED so you can watch it work. Re-running this")
    print("  lesson removes it first.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 76)


if __name__ == "__main__":
    main()
