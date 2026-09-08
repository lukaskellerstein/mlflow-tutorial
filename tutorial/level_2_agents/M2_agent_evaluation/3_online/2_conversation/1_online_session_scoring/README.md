# L2-M2.3.2.1 — Online Session Scoring

**Level:** AI Agents
**Duration:** 45 min

## Overview

`1_turn/1_online_scoring` samples production **traces** and judges each one on
its own. This lesson samples production **sessions** and judges the whole
conversation — the question no single trace can answer, asked of traffic you did
not choose.

## Prerequisites

- Completed: [`../../1_turn/1_online_scoring`](../../1_turn/1_online_scoring/),
  [`../../../1_instruments/2_conversation/3_judging_conversations`](../../../1_instruments/2_conversation/3_judging_conversations/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### What changes, and what does not

```text
turn-level online     "was this reply good?"           -> one trace
session-level online  "did this customer get helped?"  -> a list of traces
```

Registration, `start()`, and the sampling machinery are **identical**.
`Scorer.is_session_level_scorer` is the only thing that changes the server's
behaviour — it decides which unit the judge is handed.

### Built-in scorers register; decorated ones do not

An inline `@scorer` is `DECORATOR` kind. It deserialises via `exec()`, so an
open-source tracking server refuses to store it, so it can never run online.

`ConversationCompleteness` and its six siblings are `BUILTIN` kind, and
`BUILTIN` is in MLflow's `_ALLOWED_SCORERS_FOR_REGISTRATION`. That is what makes
this lesson possible, and it was verified by running it, not assumed.

## Step-by-Step

### Step 1: The gateway endpoint

The judge runs **inside** the MLflow server, on its own schedule, long after this
script has exited. So it cannot borrow the base URL the agent above uses — it
names a gateway **endpoint** the server resolves against its own database:

```python
scorer = scorer_cls(name=SESSION_SCORER_NAME, model="gateway:/gemma-judge")
```

Nothing has to be built. `infra/mlflow/gateway/seed_gateway.py` defines `gemma-judge`
and the stack seeds it into the gateway on every `podman compose up -d`, so this
step is a check rather than a setup. `openai:/gemma-judge` would register happily
and then refuse to start: it is resolved client-side, and the server has no
client.

### Step 2: Register and start

```python
scorer = ConversationCompleteness(
    name=SESSION_SCORER_NAME,
    model=f"gateway:/{endpoint}",  # gateway:/ NOT openai:/
)
registered = scorer.register(name=SESSION_SCORER_NAME)
registered.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
```

### Step 3: Serve traffic, one session per customer

Each turn is stamped with the session id that groups it:

```python
mlflow.update_current_trace(session_id=session_id)
```

### Step 4–5: Read the server state, then wait

Scoring is asynchronous and server-side. There is no local call to await — that
is the entire point of online scoring.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/3_online/2_conversation/1_online_session_scoring
uv sync
uv run python main.py
```

## Expected Output

```text
Step 2: register a SESSION-level scorer and start it
  kind          : ScorerKind.BUILTIN
  session-level : True
  registered    : production_conversation_completeness
  started       : status=ScorerStatus.STARTED sample_rate=1.0
```

Assessments appear on the sessions in the MLflow UI once the server has picked
them up. Nothing is printed locally, because nothing is computed locally.

> [!warning]
> **Two traps, both of which cost real time.**
>
> 1. **`.start()` demands a gateway model.** A scorer built with
>    `model="openai:/gemma-judge"` registers happily and then fails with
>    *"does not use a gateway model"*. Registration succeeding is not evidence
>    that starting will.
> 2. **`delete_scorer(name=...)` is not enough.** It raises *"You must set
>    `version` argument to either an integer or 'all'"*. Pass `version="all"`.

## Key Takeaways

- Session scorers **can** run online. `BUILTIN` kind registers against an
  open-source tracking server.
- The code is nearly identical to turn-level online scoring;
  `is_session_level_scorer` does all the work.
- **Cost scales differently.** A turn judge reads one trace. A session judge
  reads every turn, so its prompt and its bill grow with conversation length —
  sample lower here.
- **Signal arrives later.** A session cannot be judged until it looks finished.
  Turn scoring tells you a reply was bad within minutes; session scoring tells
  you a customer was failed, once they stopped talking.
- Run both. They answer different questions and neither substitutes for the
  other.

## Next Steps

**[`../../../../M3_agent_optimization/`](../../../../M3_agent_optimization/)** —
now that you can measure an agent at both scopes, offline and online, improve it.
