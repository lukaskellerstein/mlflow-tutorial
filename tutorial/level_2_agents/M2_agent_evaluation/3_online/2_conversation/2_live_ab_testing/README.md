# L2-M2.3.2.2 — A/B Testing on Live Sessions: Sticky Assignment

**Level:** AI Agents
**Duration:** 45 min

## Overview

[`../../1_turn/2_live_ab_testing`](../../1_turn/2_live_ab_testing/) split live
**requests** between two versions. This lesson splits live **conversations**,
and one thing changes that makes it a different problem rather than the same one
at a larger size.

> At turn scope, a bad assignment costs you a noisy data point.
> At session scope, a bad assignment **fabricates** one.

## Prerequisites

- Completed: [`../../1_turn/2_live_ab_testing`](../../1_turn/2_live_ab_testing/),
  [`1_online_session_scoring`](../1_online_session_scoring/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Why assignment must be sticky

Assign per request and a customer's turn 1 is served by v1 while their turn 2 is
served by v2. **The conversation you then score was produced by neither
version.**

That is not noise. Noise averages out over enough traffic; a fabricated
conversation does not — it is a measurement of a system you never shipped,
counted against whichever arm you happened to tag it with.

So the arm is chosen **once**, when the session opens:

```python
session_id = f"{customer}-{uuid.uuid4().hex[:6]}"
variant = assign_variant(session_id)  # decided ONCE, here
for turn in turns:
    run_turn(agents[variant], history, session_id, variant)
```

Turn-scope A/B has no such failure mode, which is exactly why this is a separate
lesson.

### Same `hash()` trap, same fix

Bucketing on the built-in `hash()` reassigns sessions after every restart,
because Python salts string hashing per process. `hashlib.md5` is a stability
choice here, not a security one.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/3_online/2_conversation/2_live_ab_testing
uv sync
uv run python main.py
```

## Expected Output

```text
Step 2: register ONE session-level judge for both arms
  session-level : True

Step 3: serve 4 whole conversations
    cust-1  session d9e1b4  arm v1  2 turns (all on v1)
    cust-2  session 788d56  arm v1  2 turns (all on v1)
    cust-3  session aad8b7  arm v2  2 turns (all on v2)
    cust-4  session dbc027  arm v1  2 turns (all on v1)

  Every turn of a session went to one arm. That is the invariant.
```

Read the `(all on vN)` column — that is the lesson, and it is visible per
session rather than asserted.

Note the 3/1 split across four sessions. Stable hashing does not promise a
balanced split at small N; it promises that a given session never changes arm.

> [!note]
> Session scoring is server-side, asynchronous, and **slower than trace
> scoring** — a session cannot be judged until it looks finished. An empty score
> table means the sampler has not caught up, not that the run failed. Check the
> UI a few minutes later.

## Key Takeaways

- **Assign once per session, not per request.** A mid-conversation flip creates
  a conversation neither version produced.
- Fabricated data does not average out the way noise does.
- One session-level judge for both arms.
- **Cost per unit.** A turn judge reads one trace; a session judge reads every
  turn, so its prompt and its bill grow with conversation length. Sample lower.
- **Time to signal.** Turn A/B tells you a reply got worse within minutes.
  Session A/B tells you customers stopped being helped — and only after they
  have stopped talking.
- Run both. They fail at different times and neither substitutes for the other.

## Next Steps

**[`../../../M3_agent_optimization/`](../../../../M3_agent_optimization/)** —
you can now measure an agent at both scopes, offline and online. Time to improve
it.
