# L2-M2.1.1 — Turn-Level Instruments

**One request in, one final answer out.** Everything in this group judges that
unit, and nothing in it can see the next turn.

## What a turn scorer is allowed to see

```text
   ┌──────────────────────────────────────────────┐
   │  question                                    │  <- scope A sees this
   │     |                                        │
   │  tool call: policy_lookup(topic="returns")   │
   │  tool call: policy_lookup(topic="returns")   │  <- scope B sees this too
   │     |                                        │
   │  final answer                                │  <- scope A sees this
   └──────────────────────────────────────────────┘
                    ONE TURN
```

Both scopes are turn-level. The difference is how the data reaches the scorer:

| Scope | How the steps arrive | Sees arguments? | Sees a repeated call? |
|:--|:--|:--|:--|
| **A** | you pass `inputs` and `outputs` | no | no |
| **B**, flattened | you put `tools_used` in `outputs` yourself | no | no |
| **B**, from the trace | the scorer takes `trace=` | **yes** | **yes** |

The third row is why `ToolCallEfficiency` can flag the duplicate call above and
`tool_selection_scorer` cannot — a flattened list of tool names has already
thrown the arguments away.

## The three lessons

| # | Lesson | Teaches |
|:--|:--|:--|
| 1 | [`1_hand_written_tests`](1_hand_written_tests/) | A case with two halves — the answer **and** the route. Pass or fail, a nested run per case, and a caught regression |
| 2 | [`2_judges`](2_judges/) | One rubric three ways: inline, registered, aligned. Only a registered judge can run online, and alignment is **turn-only** |
| 3 | [`3_quality_metrics`](3_quality_metrics/) | Partial credit, precision/recall/F1 of tool choice, composites with visible weights, and two built-in judges that read the trace |

## What you can measure here

- Did the answer contain what it should? — pass/fail, or partial credit
- Were the right tools called? — precision, recall, F1
- Were they called with sensible **arguments**? — `ToolCallCorrectness`
- Was the same call made twice inside this turn? — `ToolCallEfficiency`
- Is the reasoning coherent? — an LLM judge
- How long did the turn take, and how many tokens? — read off the trace

## What you can never ask here

- Was the task ever finished, across the whole conversation?
- Did turn 4 remember what the user said on turn 1?
- Is the user getting more frustrated?

None of those is a function of one turn. They live in
[`../2_conversation/`](../2_conversation/).

## A note on the word "judge"

A **judge** is a scorer that decides with an LLM — `class Judge(Scorer)`. The
test is whether it takes `model=`. In lesson 3, `tool_selection_scorer` computes
and is not a judge; `ToolCallCorrectness` decides and is.
