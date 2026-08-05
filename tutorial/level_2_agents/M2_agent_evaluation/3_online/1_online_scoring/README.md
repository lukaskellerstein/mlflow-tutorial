# L2-M2.3.1 — Online Scoring on Production Traces

**Level:** AI Agents
**Duration:** 90 min

## Overview

Offline evaluation tells you whether a version is good enough to ship. It cannot tell you whether what shipped is *still* good — for that you need to score real traffic, which has no expected answers, arrives continuously, and costs a model call every time you judge it. This lesson registers a judge against an MLflow AI Gateway model, starts it sampling live traces, retunes the sampling, reads the assessments back, and stops it.

## Prerequisites

- Completed: L2-M2.1.2 (Judges) — registered vs. inline judges
- Completed: L2-M2.2.2 (Offline Gates) — the other half of the question
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- **`OPENROUTER_API_KEY` exported in your shell** — the MLflow gateway secret is created from the environment, never from a file in this repo

## Concepts

### The four axes

| | offline | online |
|:--|:--|:--|
| **input** | curated dataset | production traces |
| **ground truth** | expectations | none |
| **coverage** | every case | sampled |
| **trigger** | you, in CI | the server, on a schedule |

Neither replaces the other. Offline evaluation cannot tell you that real users ask things your dataset never imagined; online scoring cannot tell you whether a change is safe *before* you ship it.

### Why the judge must be registered

An inline `@scorer` function is `DECORATOR` kind. It deserialises via `exec()`, and MLflow refuses to register that against a non-Databricks tracking URI — so it can never run server-side. `make_judge()` produces `INSTRUCTIONS` kind, which registers against a local server and therefore *can* be scheduled. Registration is not bookkeeping here; it is the thing that makes online scoring possible at all.

### Why the judge needs a gateway model

`start()` refuses any judge whose model is not a **gateway** model. Scoring runs inside the MLflow server, and the server needs its own credentialed endpoint — it cannot borrow the API key from your shell. That is why this lesson builds a secret → model definition → endpoint chain even though the agent itself is perfectly happy talking to LiteLLM.

One trap worth knowing: `create_gateway_secret(auth_config={"base_url": ...})` accepts a base URL and silently ignores it. The request still goes to the provider's own API, and the first symptom is an authentication error about a key you never sent. Use `provider="openrouter"`, which is supported natively.

### Rate versus filter

`sample_rate` answers "how much can I afford?" — judging costs a model call per trace, so cost scales with **traffic**, not with dataset size. `filter_string` answers the better question: "what is worth paying for?" Scoring 50% of checkout traffic beats scoring 5% of everything.

## Step-by-Step

### Step 1: Build the gateway endpoint

```python
store.create_gateway_endpoint(
    name=GATEWAY_ENDPOINT_NAME,
    model_configs=[
        GatewayEndpointModelConfig(
            model_definition_id=model_def.model_definition_id,
            linkage_type=GatewayModelLinkageType.PRIMARY,  # the enum, not "PRIMARY"
            weight=1,
            fallback_order=0,
        )
    ],
)
```

Passing the string `"PRIMARY"` instead of the enum fails deep in proto serialisation with `'str' object has no attribute 'to_proto'`.

### Step 2: Register the judge and start it

```python
judge = mlflow.genai.make_judge(
    name=ONLINE_JUDGE_NAME,
    instructions="... {{ inputs }} ... {{ outputs }} ...",
    model=f"gateway:/{endpoint}",
    feedback_value_type=bool,
)
registered = judge.register(name=ONLINE_JUDGE_NAME)
started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
```

### Step 3: Send live traffic

The traces the agent produces are what the server samples. Nothing here supplies an expected answer — that absence is what makes it online.

### Step 4: Retune with `update()`

```python
scorer.update(
    sampling_config=ScorerSamplingConfig(
        sample_rate=0.5,
        filter_string="attributes.status = 'OK'",
    )
)
```

`filter_string` takes the same syntax as `mlflow.search_traces()`.

### Step 5: Read assessments back, then stop

Assessments attach to traces asynchronously — the scheduler runs on its own cadence, so they do not appear synchronously with the script. `stop()` sets the sample rate to 0 but keeps the scorer registered, so `start()` can resume it later.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/3_online/1_online_scoring
uv sync
uv run python main.py
```

## Expected Output

```text
  Part 1: Register a judge and start it
  registered 'production_answer_quality' on model gateway:/tutorial-gemma-endpoint
  started: status=ScorerStatus.STARTED sample_rate=0.2

  Part 2: Send live traffic
    -> What is 25 * 4?
    ...
  server-side state: status=ScorerStatus.STARTED sample_rate=0.2

  Part 3: Retune sampling with update()
  raised rate: sample_rate=0.5
  scoped to successful traces: filter_string="attributes.status = 'OK'"

  Part 4: Read assessments back off the traces
  Part 5: Stop the scorer
  stopped: status=ScorerStatus.STOPPED
```

Part 4 frequently reports no assessments, and the reason matters more than the fact. There are two distinct causes:

1. **Sampling.** At a 20–50% rate most traces are never judged at all — working as designed.
2. **Scheduler cadence.** The server had not sampled yet when the wait expired. Because Part 5 then stops the scorer, those traces will never be scored *for that run*.

The wait is set to 180s for this reason; shortening it guarantees an empty Part 4. To actually watch assessments appear, comment out `stop_scorer()` in `main()`, re-run, and check the experiment in the MLflow UI a few minutes later — then stop the scorer, because it bills per trace.

## Key Takeaways

- Online scoring answers a question offline evaluation structurally cannot.
- Only a **registered** judge can run online; an inline `@scorer` never can.
- The judge needs a **gateway** model because scoring happens server-side.
- `sample_rate` controls cost; `filter_string` controls relevance. Use both.
- Assessments arrive asynchronously — absence shortly after a run is expected.

## Next Steps

This closes L2-M2. Continue to **L2-M3 — Agent Optimization**, which consumes the scorers built here and in M2.1 to actually change the agent. The assessments this lesson produces are also the raw material for **L3-M1**, where they become Grafana dashboards and alerting on quality regression.
