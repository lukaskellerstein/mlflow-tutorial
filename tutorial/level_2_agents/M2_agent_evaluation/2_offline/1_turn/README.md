# L2-M2.2.1 — Offline, Turn Scope

**One request in, one answer out, scored against something you already know.**

Every lesson here evaluates a single turn. That is a boundary in time, not
blindness to the steps — a benchmark that runs an agent for eight tool calls is
still scoring one task.

## The seven lessons

| # | Lesson | Unit | Bar |
|:--|:--|:--|:--|
| 1 | [`1_architecture_comparison`](1_architecture_comparison/) | one question | yours |
| 2 | [`2_offline_gates`](2_offline_gates/) | one question | yours |
| 3 | [`3_version_comparison`](3_version_comparison/) | one question, two versions | yours |
| 4 | [`4_failure_analysis`](4_failure_analysis/) | results you already have | yours |
| 5 | [`5_swe_bench`](5_swe_bench/) | one GitHub issue | everyone's |
| 6 | [`6_gaia`](6_gaia/) | one GAIA task | everyone's |
| 7 | [`7_custom_benchmark`](7_custom_benchmark/) | one support task | everyone's |

Lessons 1 to 4 work against a bar you chose. Lessons 5 to 7 measure
against a frozen number owned outside your team.

## `max_turns=8` is not a conversation

`6_gaia` and `7_custom_benchmark` both set `max_turns=8`. That sits inside
`ClaudeAgentOptions`, next to `allowed_tools` and `max_budget_usd`. It is the
agent's **internal tool-loop budget** — how many times it may think-and-call
before giving up on one task.

The agent loops eight times; the evaluation still sees one task and one answer.
That is the difference between a trajectory and a dialogue, and it is the most
common thing to misread in this group.

## What this scope cannot ask

- Was the task ever finished, across the whole conversation?
- Did turn 4 remember what the user said on turn 1?
- Is the customer getting more frustrated?

Those are in [`../2_conversation/`](../2_conversation/).
