# L2-M2.3.1.2 — A/B Testing Two Versions on Live Traffic

**Level:** AI Agents
**Duration:** 45 min

## Overview

[`../../2_offline/1_turn/3_version_comparison`](../../../2_offline/1_turn/3_version_comparison/)
compared v1 and v2 on a dataset **you** chose. This lesson runs both versions at
the same time against traffic **nobody** chose.

## Prerequisites

- Completed: [`1_online_scoring`](../1_online_scoring/),
  [`../../2_offline/1_turn/3_version_comparison`](../../../2_offline/1_turn/3_version_comparison/)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Why you cannot pair

| | offline comparison | live A/B |
|:--|:--|:--|
| Input | your cases | real questions |
| Ground truth | known | none |
| Coverage | every case | a sample |
| Pairing | **yes** — same cases both arms | **no** — request 7 hits one arm only |

Compare distributions instead. What makes that legitimate is that assignment is
random *with respect to the question*, so each arm gets the same mix of easy and
hard traffic in expectation.

### The mechanism is three moves

1. **Assign** each request to an arm, deterministically per user.
2. **Tag** the trace with the arm that served it.
3. Let **one** registered judge score a sample of both, then split by tag.

Step 3 is the one people get wrong. Two judges — or one judge retuned between
arms — measures the judges rather than the agents.

Step 2 is the entire A/B. Without the tag the assessments arrive in one
undifferentiated pile and the experiment is unrecoverable.

### The `hash()` trap

```python
digest = hashlib.md5(user_id.encode(), usedforsecurity=False).hexdigest()
return "v1" if int(digest, 16) % 2 == 0 else "v2"
```

> [!warning]
> **Do not bucket on the built-in `hash()`.** Python salts string hashing per
> process (`PYTHONHASHSEED`), so `hash("user-a") % 2` gives a different answer
> after every restart. Users would silently flip arms between deploys, and your
> A/B would be measuring the flipping. `md5` here is a **stability** choice, not
> a security one.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/3_online/1_turn/2_live_ab_testing
uv sync
uv run python main.py
```

## Expected Output

```text
Step 3: assignment -- stable per user, before any traffic
    user-a -> v2
    user-b -> v1
    user-c -> v2
    user-d -> v1
```

Same user, same arm, every time and after every restart.

Then the sampled scores, split by tag:

```text
  arm     scored    mean
  ------ ------- -------
  v1           3    1.00
  v2           3    0.67
```

> [!note]
> **Scoring is server-side and asynchronous.** The lesson polls; the server
> samples on its own cadence. If the poll ends with no assessments, the run is
> not broken — check the experiment in the UI a few minutes later. The scorer is
> deliberately left `STARTED` so the scores do arrive.
>
> If assessments appear but carry **no value**, that is a different problem:
> look for an `error` on the feedback. See the three-cause checklist in
> [`1_online_scoring`](../1_online_scoring/).

## Key Takeaways

- Live A/B cannot pair. Compare distributions, and rely on random assignment.
- **One judge for both arms**, always.
- The variant **tag** is the whole experiment — without it, unrecoverable.
- Never bucket on `hash()`; it is salted per process.
- Six requests cannot separate two prompts. Real A/B tests run for days because
  traffic has to accumulate before the difference clears the noise.
- What online buys that offline cannot: the traffic **mix** is real, including
  the questions you would never have written.

## Next Steps

**[`../2_conversation/2_live_ab_testing`](../../2_conversation/2_live_ab_testing/)**
— the same idea at session scope, where a bad assignment does something worse
than add noise.
