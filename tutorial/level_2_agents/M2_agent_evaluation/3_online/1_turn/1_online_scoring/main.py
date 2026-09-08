"""
L2-M2.3.1.1 — Online Scoring on Production Traces

Offline evaluation (M2.2) answers "is this version good enough to ship?" It
cannot answer "is what shipped still good?" -- for that the traces come from
production, there are no expectations to compare against, coverage is sampled
because every judge call costs a model call, and the SERVER pulls the trigger on
a schedule rather than you.

  Part 1: register a judge against a gateway model and start it sampling
  Part 2: send live traffic, then read the server-side scorer state
  Part 3: retune sampling with update(), including a filter_string
  Part 4: read the assessments back off the sampled traces
  Part 5: stop the scorer, and the four axes that separate the two modes

One catch decides the whole design: start() refuses any judge whose model is not
a GATEWAY model, because scoring runs server-side and the server needs its own
credentialed endpoint -- it cannot borrow the API key from your shell.

Builds on L2-M2.1.1.2 (Judges) and L2-M2.2.1.2 (Offline Gates).
"""

import time
from typing import Any

import mlflow
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
# The agent and the thing grading it are named separately on purpose: both
# resolve to the same model today, but a judge and an agent are different
# jobs and will not always want the same one. Splitting them here means that
# change is a config edit, not a re-read of this lesson.
JUDGE_ALIAS = "gemma-judge"

EXPERIMENT = "L2/M2_agent_evaluation/3_online/1_turn/1_online_scoring"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

# Without this there are no traces, and with no traces there is nothing for the
# server to sample -- the scorer would sit ACTIVE and score nothing forever. This
# is the one line that makes the rest of the lesson observable.
mlflow.langchain.autolog(log_traces=True)

# The judge runs INSIDE the MLflow server, so it names a gateway ENDPOINT
# rather than a base URL. `openai:/gemma-judge` would register happily and
# then fail to start: an `openai:/` model is resolved client-side, and the
# server has no client. See check_judge_endpoint() below.

ONLINE_JUDGE_NAME = "production_answer_quality"


# ---------------------------------------------------------------------------
# The application under observation
# ---------------------------------------------------------------------------
KNOWLEDGE = {
    "python": "Python is a high-level programming language known for readability. "
    "It supports multiple paradigms including OOP, functional, and procedural.",
    "mlflow": "MLflow is an open-source platform for the ML lifecycle. "
    "It provides tracking, model registry, evaluation, and deployment.",
    "langgraph": "LangGraph builds stateful multi-actor LLM applications "
    "using graph-based workflows with nodes, edges, and state.",
}


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information on a topic."""
    q = query.lower()
    results = [v for k, v in KNOWLEDGE.items() if k in q]
    return results[0] if results else f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3' or '10 * 5'."""
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        return str(eval(expression))  # nosec: reached only for whitelisted arithmetic chars
    return "Invalid expression — only basic arithmetic is supported."


def build_agent():
    """The agent whose live traffic will be scored."""
    llm = ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )
    return create_agent(llm, [search_knowledge, calculate])


# ---------------------------------------------------------------------------
# Part 1: the gateway endpoint the judge will run on
# ---------------------------------------------------------------------------
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


def register_and_start(endpoint: str):
    """Register the judge, then start it sampling live traces."""
    from mlflow.genai.scorers import ScorerSamplingConfig

    print("\n" + "=" * 60)
    print("  Part 1: Register a judge and start it")
    print("=" * 60)

    judge = mlflow.genai.make_judge(
        name=ONLINE_JUDGE_NAME,
        instructions=(
            "You are reviewing a live support answer.\n"
            "The request is in {{ inputs }} and the agent's reply is in {{ outputs }}.\n"
            "Is the reply accurate and genuinely useful to the user? Answer true or false."
        ),
        model=f"gateway:/{endpoint}",
        feedback_value_type=bool,
    )

    # Registration is the step that makes online scoring possible at all. An
    # inline @scorer function is DECORATOR kind -- it deserialises via exec() and
    # cannot be registered against a non-Databricks tracking URI, so it can never
    # run server-side. make_judge produces INSTRUCTIONS kind, which can.
    registered = judge.register(name=ONLINE_JUDGE_NAME)
    print(f"  registered '{registered.name}' on model gateway:/{endpoint}")

    # sample_rate is the whole economic argument for online scoring: judging is a
    # model call per trace, so cost scales with TRAFFIC, not with dataset size.
    # 20% is a deliberate choice, not a default.
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
    print(f"  started: status={started.status} sample_rate={started.sample_rate}")
    return started


# ---------------------------------------------------------------------------
# Part 2: live traffic
# ---------------------------------------------------------------------------
def send_live_traffic(agent: Any, questions: list[str]) -> None:
    print("\n" + "=" * 60)
    print("  Part 2: Send live traffic")
    print("=" * 60)
    print("  These traces are what the server samples. No expected answers exist")
    print("  for any of them -- that is what makes this online, not offline.\n")

    for question in questions:
        agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"    -> {question}")
    mlflow.flush_trace_async_logging()

    fresh = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)
    print(f"\n  server-side state: status={fresh.status} sample_rate={fresh.sample_rate}")
    print("  The scheduler picks active scorers up on its own cadence, so assessments")
    print("  appear on sampled traces shortly -- not synchronously with this script.")


# ---------------------------------------------------------------------------
# Part 3: retune sampling
# ---------------------------------------------------------------------------
def retune_sampling() -> None:
    """Sampling is not set once. update() changes rate and filter in place."""
    from mlflow.genai.scorers import ScorerSamplingConfig

    print("\n" + "=" * 60)
    print("  Part 3: Retune sampling with update()")
    print("=" * 60)

    scorer = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)

    raised = scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
    print(f"  raised rate: sample_rate={raised.sample_rate}")

    # filter_string is the other half of cost control, and the more useful half:
    # score the traffic that matters instead of a random slice of everything. It
    # takes the same syntax as mlflow.search_traces().
    filtered = scorer.update(
        sampling_config=ScorerSamplingConfig(
            sample_rate=0.5,
            filter_string="attributes.status = 'OK'",
        )
    )
    print(f"  scoped to successful traces: filter_string={filtered.filter_string!r}")
    print("\n  Rate answers 'how much can I afford?'. Filter answers 'what is worth")
    print("  paying for?' -- 50% of checkout traffic beats 5% of everything.")


# ---------------------------------------------------------------------------
# Part 4: read the assessments back
# ---------------------------------------------------------------------------
def read_assessments(wait_seconds: int = 180) -> None:
    """Assessments land on traces asynchronously. Poll, then summarise.

    The wait is long on purpose. The server scheduler decides when to sample, and
    Part 5 stops the scorer -- so a short wait here does not merely miss the
    assessments, it prevents them from ever being produced for this run. If you
    shorten this, expect Part 4 to be empty every time.
    """
    print("\n" + "=" * 60)
    print("  Part 4: Read assessments back off the traces")
    print("=" * 60)
    print(f"  Waiting up to {wait_seconds}s for the server scheduler...\n")

    deadline = time.time() + wait_seconds
    scored: list[Any] = []
    while time.time() < deadline:
        traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=50, return_type="list")
        scored = [t for t in traces if getattr(t.info, "assessments", None)]
        if scored:
            break
        time.sleep(5)

    if not scored:
        print("  No assessments within the wait. THREE causes, and they are")
        print("  different -- check them in this order:")
        print("    1. Judge ERROR. An assessment can exist and carry no value. Look")
        print("       for one whose feedback has an `error` rather than a `value`:")
        print("       that is the judge failing, not the scheduler being slow, and")
        print("       waiting longer will never fix it.")
        print("    2. Sampling -- at 20-50%, most traces are never judged at all.")
        print("    3. Cadence  -- the scheduler had not run yet. Part 5 stops the")
        print("       scorer, so for THIS run those traces will never be scored.")
        print("\n  Cause 1 is the one that looks like the others and is not. A real")
        print("  example from this repo: the MLflow server rejected its OWN gateway")
        print(
            "  callback with 'Invalid Host header - possible DNS rebinding attack",
        )
        print("  detected', because 0.0.0.0:5000 was missing from")
        print("  MLFLOW_SERVER_ALLOWED_HOSTS in infra/compose.yml. Every judge")
        print("  failed, silently, and this message blamed sampling for it.")
        print("\n  To see assessments: comment out stop_scorer() in main(), re-run,")
        print("  and check the experiment in the MLflow UI a few minutes later.")
        print("  Remember to stop the scorer afterwards -- it bills per trace.")
        return

    print(f"  {len(scored)} trace(s) carry assessments:\n")
    for trace in scored:
        for assessment in trace.info.assessments:
            value = getattr(assessment, "value", None)
            feedback = getattr(assessment, "feedback", None)
            if feedback is not None:
                value = getattr(feedback, "value", value)
            print(f"    {trace.info.trace_id[:16]}…  {assessment.name} = {value}")

    print("\n  This is the quality trend, accumulating on its own. Level 3 consumes")
    print("  exactly these assessments in Grafana dashboards and alerts.")


# ---------------------------------------------------------------------------
# Part 5: stop
# ---------------------------------------------------------------------------
def stop_scorer() -> None:
    print("\n" + "=" * 60)
    print("  Part 5: Stop the scorer")
    print("=" * 60)

    scorer = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)
    stopped = scorer.stop()
    print(f"  stopped: status={stopped.status}  (left running, it would score forever)")
    print("  stop() sets the sample rate to 0 but keeps the scorer registered, so")
    print("  start() can resume it later without re-registering.")

    print("\n  offline vs online, on four axes:")
    print(f"    {'':<12}{'offline':<26}{'online'}")
    print(f"    {'input':<12}{'curated dataset':<26}{'production traces'}")
    print(f"    {'truth':<12}{'expectations':<26}{'none'}")
    print(f"    {'coverage':<12}{'every case':<26}{'sampled'}")
    print(f"    {'trigger':<12}{'you, in CI':<26}{'the server, on a schedule'}")


def main() -> None:
    agent = build_agent()

    print("=" * 60)
    print("  L2-M2.3.1.1 — Online Scoring on Production Traces")
    print("=" * 60)

    endpoint = check_judge_endpoint()
    register_and_start(endpoint)

    send_live_traffic(
        agent,
        [
            "What is 25 * 4?",
            "What does the knowledge base say about mlflow?",
            "Explain what LangGraph is used for.",
            "What is 144 / 12?",
        ],
    )

    retune_sampling()
    read_assessments()
    stop_scorer()

    print("\n" + "=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5555")
    print(f"  Experiment: {EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
