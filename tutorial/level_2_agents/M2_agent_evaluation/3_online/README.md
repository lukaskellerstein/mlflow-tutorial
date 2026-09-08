# L2-M2.3 — Online Evaluation

> **"Is what shipped still good?"**

Sampled live traffic, no ground truth, partial coverage, and **the server** pulls
the trigger on its own schedule. Contrast with
[`../2_offline/`](../2_offline/), where you own the input and choose when to run.

Four lessons, split by the same scope axis used everywhere else in this module.

```text
1_turn/          one production trace     "was this reply good?"
2_conversation/  one production session   "did this customer get helped?"
```

## The four axes that separate online from offline

| | Offline | Online |
|:--|:--|:--|
| Input | curated, yours | sampled, whoever showed up |
| Ground truth | known | none |
| Coverage | every case | a sample rate |
| Trigger | you | the server, continuously |

## What is identical across both branches

Registration, `start()`, the sampling config, and the requirement that the judge
run on a **gateway** model. `Scorer.is_session_level_scorer` is the only thing
that changes the server's behaviour: it decides whether the judge is handed one
trace or a list of traces sharing a session id.

## Two traps that cost real time

1. **`.start()` demands a gateway model.** A scorer built with
   `model="openai:/gemma-judge"` registers happily and then refuses to start:
   *"does not use a gateway model"*. The judge runs **inside** the MLflow
   server, which has neither your base URL nor your key — an `openai:/` model is
   resolved client-side, and the server has no client. So both lessons name
   `gateway:/gemma-judge` instead. Nothing has to be built: the stack seeds every
   alias in `infra/mlflow/gateway/seed_gateway.py` into the gateway on `up -d`.
2. **`delete_scorer(name=...)` is not enough.** It raises asking for a version.
   Pass `version="all"`.

## Cost scales differently in each branch

Online judging costs money per **sampled unit**, so the sample rate is the whole
economic argument. But the unit is not the same size:

| Branch | The judge reads | Cost per sampled unit |
|:--|:--|:--|
| `1_turn/` | one trace | flat |
| `2_conversation/` | every turn of the conversation | **grows with conversation length** |

Sample lower at session level than you would at turn level.

## The two lessons

| # | Lesson | Unit |
|:--|:--|:--|
| 1 | [`1_turn/1_online_scoring`](1_turn/1_online_scoring/) | one trace |
| 2 | [`1_turn/2_live_ab_testing`](1_turn/2_live_ab_testing/) | one trace, two arms |
| 3 | [`2_conversation/1_online_session_scoring`](2_conversation/1_online_session_scoring/) | one session |
| 4 | [`2_conversation/2_live_ab_testing`](2_conversation/2_live_ab_testing/) | one session, two arms |
