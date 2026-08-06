# L1-M4.3.1 — Online Scoring for LLM Applications

**Level:** Models
**Duration:** 60 min

## Overview

Everything in M4.1 and M4.2 was offline evaluation: a curated dataset, expected answers, and you deciding when it runs. That answers *"is this version good enough to ship?"* It structurally cannot answer *"is what shipped still good?"* — real traffic has no expected answers. This lesson registers a judge, has the MLflow server score a sampled share of your app's live traces on a schedule, and reads the resulting assessments back.

## Prerequisites

- Completed: L1-M4.1.1 (Evaluation Fundamentals) and L1-M4.2.1 (GenAI Custom Metrics)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway up (`cd infra && podman compose up -d`), with LMStudio
  serving `google/gemma-4-26b-a4b` behind the `gemma-chat` alias
- **`OPENROUTER_API_KEY` exported in your shell** — the gateway secret is built from the environment, never from a file in this repo

## Concepts

### The four axes

| | offline | online |
|:--|:--|:--|
| **input** | curated dataset | production traces |
| **ground truth** | expectations | none |
| **coverage** | every case | sampled |
| **trigger** | you, in CI | the server, on a schedule |

Neither replaces the other. Offline evaluation cannot tell you that real users ask things your dataset never imagined; online scoring cannot tell you whether a change is safe *before* you ship it.

### This is not an agent feature

It is easy to assume online scoring belongs with agents. It does not. `scorer.start()` samples **traces**, and a trace is produced by any instrumented call — a single `chat.completions.create()` counts exactly as much as a multi-step agent. That is why this lesson sits in Level 1, right beside the offline lessons it completes.

### Registration is the enabling step

An inline `@scorer` function is `DECORATOR` kind: it deserialises via `exec()`, and MLflow refuses to register that against a non-Databricks tracking URI. It therefore can never run server-side. `make_judge()` produces `INSTRUCTIONS` kind, which registers fine against your local server and *can* be scheduled.

### Why the judge needs a gateway model, even though the app does not

Your app talks to the LiteLLM gateway with a base URL and a key from your own process. The judge cannot: scoring runs **inside the MLflow server**, which has neither. So the judge needs a gateway endpoint the server owns — a secret, a model definition and an endpoint, built once and reused. It points back at the same LiteLLM proxy, but by its container name (`http://litellm:4000/v1`), because the server dials it over the compose network.

## Step-by-Step

### Step 1: Turn on tracing

```python
mlflow.openai.autolog(log_traces=True)
```

Without traces there is nothing to sample, and the scorer would sit active scoring nothing.

### Step 2: Register a judge on a gateway model and start it

```python
judge = mlflow.genai.make_judge(
    name=ONLINE_JUDGE_NAME,
    instructions="... {{ inputs }} ... {{ outputs }} ...",
    model=f"gateway:/{endpoint}",
    feedback_value_type=bool,
)
registered = judge.register(name=ONLINE_JUDGE_NAME)
registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
```

### Step 3: Send live traffic

Plain LLM calls. Nothing supplies an expected answer — that absence is what makes this online.

### Step 4: Retune with `update()`

```python
scorer.update(
    sampling_config=ScorerSamplingConfig(
        sample_rate=0.2,
        filter_string="attributes.status = 'OK'",
    )
)
```

`sample_rate` controls cost — judging is a model call per trace, so spend scales with **traffic**, not dataset size. `filter_string` controls relevance and takes the same syntax as `mlflow.search_traces()`.

### Step 5: Read assessments, then stop

`stop()` sets the sample rate to 0 but keeps the scorer registered, so `start()` resumes it later.

## Running the Lesson

```bash
cd tutorial/level_1_models/M4_evaluation/3_online/1_online_scoring
uv sync
uv run python main.py
```

## Expected Output

```text
  Part 1: Register a judge and start it sampling
  registered 'l1_production_answer_quality' on model gateway:/tutorial-gemma-endpoint
  started: status=ScorerStatus.STARTED sample_rate=0.5

  Part 2: Send live traffic
    Q: What is MLflow used for, in one sentence?
    A: MLflow is an open-source platform for managing the machine learning...
  4 trace(s) now in the experiment -- the sampling pool.

  Part 3: Retune sampling with update()
  lowered rate: sample_rate=0.2

  Part 4: Read assessments back off the traces
  Part 5: Stop the scorer
  stopped: status=ScorerStatus.STOPPED
```

Part 4 frequently reports no assessments, and the reason matters more than the fact. There are two distinct causes:

1. **Sampling.** At a 20–50% rate most traces are never judged at all — working as designed.
2. **Scheduler cadence.** The server had not sampled yet when the wait expired. Because Part 5 then stops the scorer, those traces will never be scored *for that run*.

The wait is set to 180s for this reason. To actually watch assessments appear, comment out `stop_scorer()` in `main()`, re-run, and check the experiment in the MLflow UI a few minutes later — then stop the scorer, because it bills per trace.

## Key Takeaways

- Online scoring answers a question offline evaluation structurally cannot.
- It is **not** agent-specific — `scorer.start()` samples traces, and one LLM call makes one.
- Only a **registered** judge can run online; an inline `@scorer` never can.
- The judge needs a **gateway** model because scoring happens server-side.
- `sample_rate` controls cost, `filter_string` controls relevance.

## Next Steps

This completes L1-M4. Continue to **L1-M5 — Prompt Registry and Management**, then **L1-M7 — Optimization**, which uses the scorers from this module to actually improve the model's behaviour. The agent-scale version of this lesson is **L2-M2.3.1**.
