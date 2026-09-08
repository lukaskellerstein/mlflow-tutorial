# L2-M2.2 — Offline Evaluation

> **"Is this version good enough to ship?"**

Curated input you own, known expectations, full coverage, and **you** pull the
trigger. That is the whole definition. Contrast it with
[`../3_online/`](../3_online/), which answers "is what shipped still good?"
against sampled live traffic.

Eleven lessons, split the same way the instruments are — by **scope**.

```text
1_turn/          one request, one answer      7 lessons
2_conversation/  many turns, one session      4 lessons
```

## Why the split is here too

Offline and online divide by **when** you run the check. Scope divides by
**what a scorer may see**. They are independent, so both scopes belong in both
groups.

The reason it matters is not symmetry. It is that **a turn gate can be all
green while the product fails**. A support agent can answer every individual
question correctly and still leave the customer unhelped, and "was the task
ever finished?" is a property of the session. If your only gate is turn-level,
that failure ships.

## The two branches

| Branch | The question it gates on | Lessons |
|:--|:--|--:|
| [`1_turn/`](1_turn/) | Was each answer correct? | 7 |
| [`2_conversation/`](2_conversation/) | Did the whole conversation succeed? | 4 |

## Whose bar are you measuring against?

That is the second axis, and it runs *inside* `1_turn/`:

| Lessons | Bar |
|:--|:--|
| `1_architecture_comparison`, `2_offline_gates`, `3_version_comparison`, `4_failure_analysis` | **Yours** — thresholds and questions you chose |
| `5_swe_bench`, `6_gaia`, `7_custom_benchmark` | **Everyone's** — a frozen, externally owned number |

A benchmark is just an offline evaluation whose dataset and metric are frozen
and owned outside your team. Nothing else separates it.

## Why the benchmarks are single-turn

This is the question the folder layout should provoke, so here is the answer.

A benchmark's value is that its number means the same thing to everyone.
Multi-turn breaks that: the conversation depends on the user simulator, which
is itself a model, so changing it changes the score.

It is solvable, and [`2_conversation/4_multi_turn_benchmark`](2_conversation/4_multi_turn_benchmark/)
shows the three moves that solve it — pin the user model, score state rather
than text, and report pass^k. Single-turn benchmarks avoid the problem;
multi-turn ones pay for it.

## Next

**[`../3_online/`](../3_online/)** — the same two scopes, against traffic you
did not choose.
