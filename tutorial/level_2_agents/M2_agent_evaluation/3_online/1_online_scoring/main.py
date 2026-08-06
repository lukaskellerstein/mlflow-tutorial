"""
L2-M2.3.1 — Online Scoring on Production Traces

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

Builds on L2-M2.1.2 (Judges) and L2-M2.2.2 (Offline Gates).
"""

import time
from typing import Any

import mlflow
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# The LiteLLM gateway from infra/ serves the AGENT directly. The judge reaches
# the same proxy, but only through an MLflow gateway endpoint -- see
# ensure_gateway_endpoint() below for why it cannot use these constants.
LITELLM_URL = "http://localhost:4000/v1"
LITELLM_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-agent"
# The agent and the thing grading it are named separately on purpose: both
# resolve to the same model today, but a judge and an agent are different
# jobs and will not always want the same one. Splitting them here means that
# change is a config edit, not a re-read of this lesson.
JUDGE_ALIAS = "gemma-judge"

EXPERIMENT = "L2/M2_agent_evaluation/3_online/1_online_scoring"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

# Without this there are no traces, and with no traces there is nothing for the
# server to sample -- the scorer would sit ACTIVE and score nothing forever. This
# is the one line that makes the rest of the lesson observable.
mlflow.langchain.autolog(log_traces=True)

# MLflow AI Gateway objects the judge runs on.
#
# This endpoint points back at the SAME LiteLLM proxy the agent uses, but by its
# CONTAINER name: the MLflow server dials it over the compose network, where
# "localhost" would be the MLflow container itself.
MLFLOW_SIDE_GATEWAY_URL = "http://litellm:4000/v1"
GATEWAY_SECRET_NAME = "litellm-tutorial"
GATEWAY_MODEL_NAME = "litellm-gemma-large"
GATEWAY_ENDPOINT_NAME = "tutorial-gemma-endpoint"
UPSTREAM_MODEL = JUDGE_ALIAS

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
        base_url=LITELLM_URL,
        api_key=SecretStr(LITELLM_KEY),
        temperature=0.0,
    )
    return create_agent(llm, [search_knowledge, calculate])


# ---------------------------------------------------------------------------
# Part 1: the gateway endpoint the judge will run on
# ---------------------------------------------------------------------------
def ensure_gateway_endpoint() -> str:
    """Build (or reuse) secret -> model definition -> endpoint. Returns its name.

    This points at the same LiteLLM proxy the agent uses, so the judge and the
    thing it judges run on one model. Two details make that work, and both are
    easy to get wrong:

    1. The key is `api_base`, inside `auth_config`. `base_url` is NOT a synonym --
       it is accepted and silently ignored, as is an `api_base` placed in
       `secret_value` instead. Either mistake sends the request to the provider's
       own API, and the first symptom is an authentication error about a key you
       never sent, which points nowhere near the actual cause.
    2. `provider="openai"`, because LiteLLM speaks the OpenAI protocol. The
       provider name selects the request format, not the destination -- the
       destination is `api_base`.
    """
    from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
    from mlflow.tracking._tracking_service.utils import _get_store

    store = _get_store()

    existing = {e.name: e for e in store.list_gateway_endpoints()}
    if GATEWAY_ENDPOINT_NAME in existing:
        print(f"  reusing gateway endpoint '{GATEWAY_ENDPOINT_NAME}'")
        return GATEWAY_ENDPOINT_NAME

    secrets = {s.secret_name: s for s in store.list_secret_infos()}
    secret = secrets.get(GATEWAY_SECRET_NAME) or store.create_gateway_secret(
        secret_name=GATEWAY_SECRET_NAME,
        secret_value={"api_key": LITELLM_KEY},
        provider="openai",
        auth_config={"api_base": MLFLOW_SIDE_GATEWAY_URL},
    )

    defs = {d.name: d for d in store.list_gateway_model_definitions()}
    model_def = defs.get(GATEWAY_MODEL_NAME) or store.create_gateway_model_definition(
        name=GATEWAY_MODEL_NAME,
        secret_id=secret.secret_id,
        provider="openai",
        model_name=UPSTREAM_MODEL,
    )

    store.create_gateway_endpoint(
        name=GATEWAY_ENDPOINT_NAME,
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                # The ENUM, not the string "PRIMARY" -- a string fails deep in
                # proto serialisation with 'str' object has no attribute 'to_proto'.
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1,
                fallback_order=0,
            )
        ],
    )
    print(f"  created gateway endpoint '{GATEWAY_ENDPOINT_NAME}' -> {MLFLOW_SIDE_GATEWAY_URL} ({UPSTREAM_MODEL})")
    return GATEWAY_ENDPOINT_NAME


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
        print("  No assessments within the wait. Two causes, and they are different:")
        print("    1. Sampling -- at 20-50%, most traces are never judged at all.")
        print("    2. Cadence  -- the scheduler had not run yet. Part 5 stops the")
        print("       scorer, so for THIS run those traces will never be scored.")
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
    print("  L2-M2.3.1 — Online Scoring on Production Traces")
    print("=" * 60)

    endpoint = ensure_gateway_endpoint()
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
