"""
L1-M4.3.1 — Online Scoring for LLM Applications

Everything in M4.1 and M4.2 was OFFLINE: a curated dataset, expected answers,
every case scored, and you decide when it runs. That answers "is this version
good enough to ship?"

This lesson answers the other question: "is what shipped still good?" The input is
production traffic, there is no ground truth to compare against, coverage is
sampled because every judge call costs a model call, and the SERVER pulls the
trigger on a schedule rather than you.

  Part 1: register a judge against a gateway model and start it sampling
  Part 2: send live traffic through a plain LLM app
  Part 3: retune sampling with update()
  Part 4: read the assessments back off the traces
  Part 5: stop, and the four axes

Nothing about this is agent-specific. `scorer.start()` samples **traces**, so a
single chat.completions.create() call qualifies exactly as an agent does -- which
is why this lesson lives in Level 1 rather than waiting for Level 2.
"""

import time
from typing import Any

import mlflow
from openai import OpenAI

# The LiteLLM gateway from infra/, not a provider directly. The aliases below are
# defined in infra/litellm/config.yaml, which also owns the fallback order and
# each model's context window. Swapping model or provider is a change there,
# never here.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin

MODEL_NAME = "gemma-chat"

EXPERIMENT = "L1/M4_evaluation/3_online/1_online_scoring"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

# No traces means nothing for the server to sample. This one line is what makes
# the whole lesson observable.
mlflow.openai.autolog(log_traces=True)

# The JUDGE cannot reuse the client above. Scoring happens inside the MLflow
# server, so the judge needs a credentialed gateway endpoint the server owns --
# it cannot borrow a base URL or key from your shell.
#
# That endpoint points back at the SAME LiteLLM proxy, but by its CONTAINER name:
# the MLflow server dials it over the compose network, so "localhost" would be
# the MLflow container itself. This is the one place in the tutorial where the
# gateway URL is not localhost, and getting it wrong fails at scoring time, not
# at setup time.
MLFLOW_SIDE_GATEWAY_URL = "http://litellm:4000/v1"
GATEWAY_SECRET_NAME = "litellm-tutorial"
GATEWAY_MODEL_NAME = "litellm-gemma-large"
GATEWAY_ENDPOINT_NAME = "tutorial-gemma-endpoint"
UPSTREAM_MODEL = "gemma-judge"  # the JUDGE runs server-side; the app above is gemma-chat

ONLINE_JUDGE_NAME = "l1_production_answer_quality"


# ---------------------------------------------------------------------------
# The application under observation
# ---------------------------------------------------------------------------
def answer(question: str) -> str:
    """A plain LLM call -- no agent, no tools, no framework."""
    client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a concise, accurate assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def check_gateway() -> None:
    """Fail early and clearly rather than deep inside an OpenAI client error.

    The request carries the key because /v1/models is authenticated: an
    unauthenticated probe gets a 401, and urllib raises HTTPError for it --
    a subclass of URLError. Catching URLError alone would therefore report a
    perfectly healthy gateway as unreachable.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"{GATEWAY_URL}/models", headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"The gateway answered {exc.code} at {GATEWAY_URL}. It is running, so this\n"
            f"is a config problem -- check GATEWAY_KEY against LITELLM_MASTER_KEY in infra/.env."
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"The LiteLLM gateway is not reachable at {GATEWAY_URL}.\n"
            "Start the stack:  cd infra && podman compose up -d"
        ) from exc


# ---------------------------------------------------------------------------
# Part 1: the gateway endpoint the judge runs on
# ---------------------------------------------------------------------------
def ensure_gateway_endpoint() -> str:
    """Build (or reuse) secret -> model definition -> endpoint. Returns its name."""
    from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
    from mlflow.tracking._tracking_service.utils import _get_store

    store = _get_store()

    existing = {e.name: e for e in store.list_gateway_endpoints()}
    if GATEWAY_ENDPOINT_NAME in existing:
        print(f"  reusing gateway endpoint '{GATEWAY_ENDPOINT_NAME}'")
        return GATEWAY_ENDPOINT_NAME

    # provider="openai" because LiteLLM speaks the OpenAI protocol. The base URL
    # belongs in auth_config, NOT in secret_value: a base URL placed in
    # secret_value is silently ignored and the server calls api.openai.com
    # instead, which surfaces as an OpenAI "Incorrect API key" 401 that says
    # nothing about the real mistake.
    secrets = {s.secret_name: s for s in store.list_secret_infos()}
    secret = secrets.get(GATEWAY_SECRET_NAME) or store.create_gateway_secret(
        secret_name=GATEWAY_SECRET_NAME,
        secret_value={"api_key": GATEWAY_KEY},
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
                # The ENUM, not the string "PRIMARY".
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1,
                fallback_order=0,
            )
        ],
    )
    print(f"  created gateway endpoint '{GATEWAY_ENDPOINT_NAME}' -> {MLFLOW_SIDE_GATEWAY_URL} ({UPSTREAM_MODEL})")
    return GATEWAY_ENDPOINT_NAME


def register_and_start(endpoint: str) -> None:
    from mlflow.genai.scorers import ScorerSamplingConfig

    print("\n" + "=" * 60)
    print("  Part 1: Register a judge and start it sampling")
    print("=" * 60)

    judge = mlflow.genai.make_judge(
        name=ONLINE_JUDGE_NAME,
        instructions=(
            "You are reviewing a live answer from a production assistant.\n"
            "The user asked {{ inputs }} and the assistant replied {{ outputs }}.\n"
            "Is the reply accurate and genuinely useful? Answer true or false."
        ),
        model=f"gateway:/{endpoint}",
        feedback_value_type=bool,
    )

    # Registration is what makes online scoring possible. An inline @scorer is
    # DECORATOR kind -- it deserialises via exec() and cannot be registered
    # against a local tracking server, so it can never run server-side.
    registered = judge.register(name=ONLINE_JUDGE_NAME)
    print(f"  registered '{registered.name}' on model gateway:/{endpoint}")

    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
    print(f"  started: status={started.status} sample_rate={started.sample_rate}")


# ---------------------------------------------------------------------------
# Part 2: live traffic
# ---------------------------------------------------------------------------
def send_live_traffic(questions: list[str]) -> None:
    print("\n" + "=" * 60)
    print("  Part 2: Send live traffic")
    print("=" * 60)
    print("  No expected answers exist for any of these. That absence is exactly")
    print("  what makes this online rather than offline.\n")

    for question in questions:
        reply = answer(question)
        print(f"    Q: {question}")
        print(f"    A: {reply.strip()[:70]}...")
    mlflow.flush_trace_async_logging()

    traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=50, return_type="list")
    print(f"\n  {len(traces)} trace(s) now in the experiment -- the sampling pool.")


# ---------------------------------------------------------------------------
# Part 3: retune sampling
# ---------------------------------------------------------------------------
def retune_sampling() -> None:
    from mlflow.genai.scorers import ScorerSamplingConfig

    print("\n" + "=" * 60)
    print("  Part 3: Retune sampling with update()")
    print("=" * 60)

    scorer = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)

    lowered = scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
    print(f"  lowered rate: sample_rate={lowered.sample_rate}")

    filtered = scorer.update(
        sampling_config=ScorerSamplingConfig(
            sample_rate=0.2,
            filter_string="attributes.status = 'OK'",
        )
    )
    print(f"  scoped to successful traces: filter_string={filtered.filter_string!r}")
    print("\n  sample_rate answers 'how much can I afford?' -- judge cost scales with")
    print("  TRAFFIC, not dataset size. filter_string answers 'what is worth paying")
    print("  for?', and it takes the same syntax as mlflow.search_traces().")


# ---------------------------------------------------------------------------
# Part 4: read the assessments back
# ---------------------------------------------------------------------------
def read_assessments(wait_seconds: int = 180) -> None:
    """The wait is long on purpose.

    The server scheduler decides when to sample, and Part 5 stops the scorer -- so
    a short wait here does not merely miss the assessments, it prevents them from
    ever being produced for this run.
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
            feedback = getattr(assessment, "feedback", None)
            value = getattr(feedback, "value", None) if feedback else getattr(assessment, "value", None)
            print(f"    {trace.info.trace_id[:16]}...  {assessment.name} = {value}")

    print("\n  That is a quality trend accumulating without you asking for it.")


# ---------------------------------------------------------------------------
# Part 5: stop
# ---------------------------------------------------------------------------
def stop_scorer() -> None:
    print("\n" + "=" * 60)
    print("  Part 5: Stop the scorer")
    print("=" * 60)

    scorer = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)
    stopped = scorer.stop()
    print(f"  stopped: status={stopped.status}")
    print("  stop() sets the sample rate to 0 but keeps the scorer registered, so")
    print("  start() resumes it later without re-registering.")

    print("\n  offline vs online, on four axes:")
    print(f"    {'':<12}{'offline':<26}{'online'}")
    print(f"    {'input':<12}{'curated dataset':<26}{'production traces'}")
    print(f"    {'truth':<12}{'expectations':<26}{'none'}")
    print(f"    {'coverage':<12}{'every case':<26}{'sampled'}")
    print(f"    {'trigger':<12}{'you, in CI':<26}{'the server, on a schedule'}")


def main() -> None:
    print("=" * 60)
    print("  L1-M4.3.1 — Online Scoring for LLM Applications")
    print("=" * 60)

    check_gateway()
    endpoint = ensure_gateway_endpoint()
    register_and_start(endpoint)

    send_live_traffic(
        [
            "What is MLflow used for, in one sentence?",
            "Name two benefits of experiment tracking.",
            "What is the capital of Japan?",
            "Explain what a model registry does.",
        ]
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
