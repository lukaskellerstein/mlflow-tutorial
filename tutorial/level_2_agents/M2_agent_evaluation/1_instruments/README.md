# L2-M2.1 — Instruments

The materials you evaluate an agent *with*. Nine lessons in three groups.

Before any of them, one picture. **Scope** is what a scorer is allowed to see,
and it decides what a scorer can ask.

```text
              TURN 1                      TURN 2   TURN 3   TURN 4
   ┌────────────────────────────┐         ┌────┐   ┌────┐   ┌────┐
   │ question                   │         │ .. │   │ .. │   │ .. │
   │   |                        │         │    │   │    │   │    │
   │ tool call: policy_lookup   │         │    │   │    │   │    │
   │ tool call: policy_lookup   │   <-- B │    │   │    │   │    │
   │   |                        │         │    │   │    │   │    │
   │ final answer               │         │    │   │    │   │    │
   └────────────────────────────┘         └────┘   └────┘   └────┘
     ^
     A = the question and the final answer only
     B = the whole inside of ONE turn (the trace)
     C = all four turns together  ─────────────────────────────>
```

**"One turn" is a boundary in time, not blindness to the steps.** A turn-level
scorer can see every tool call and every argument inside that turn. What it
cannot see is the next turn.

## The three groups

| Group | Scope | The question it answers |
|:--|:--|:--|
| [`1_turn/`](1_turn/) | A and B | How good was this one answer? |
| [`2_conversation/`](2_conversation/) | C | How good was the whole discussion? |
| [`3_dataset_store/`](3_dataset_store/) | — | Where do the cases live after the script ends? |

## What decides pass/fail versus a score

Not scope. **Ground truth.**

| What you give the evaluator | What comes back |
|:--|:--|
| A question **and its right answer** | pass or fail |
| A question and a **rubric** | a score, 0.0 to 1.0 |
| A whole conversation and a **rubric** | a word (`yes` / `no` / `none`), mapped to a number |
| A whole conversation and **nothing else** | a list of issues, with severity and root causes |

A turn can give you a score, and a conversation could give you pass/fail if you
wrote the correct ending down. This tutorial has no ground truth for whole
conversations, because writing the "correct conversation" by hand is not
practical.

## Scorer versus judge

**A judge is a scorer that decides with an LLM.** In code it is literally
`class Judge(Scorer)`, so every judge is a scorer and not every scorer is a
judge. The practical test is whether it takes a `model=` parameter.

| | Scorer | Judge |
|:--|:--|:--|
| Example | `tool_selection_scorer` — precision and recall in plain Python | `ToolCallCorrectness(model=...)` |
| Needs a model? | No | Yes |

MLflow mixes the two words in its API — `list_scorers()` returns judges, and
`judge.register()` writes into the scorer registry. One registry, two words.

## Two things that cost the most confusion

> MLflow calls them `test_cases`, but a **simulator** test case is a *scenario*,
> not an assertion. It holds a goal, not a right answer, so it cannot pass or
> fail. `3_dataset_store/2_generated_records` proves this at runtime: the
> simulator rejects an evaluation dataset and accepts a scenario dataset.

> `persona` describes **one** simulated user. `ConversationSimulator` has exactly
> two sides — that user and your agent. There is no multi-party mode.

## The nine lessons

| Lesson | You give it | You get back |
|:--|:--|:--|
| `1_turn/1_hand_written_tests` | a question, its answer, the expected tools | pass or fail |
| `1_turn/2_judges` | a rubric | a score for the subjective half |
| `1_turn/3_quality_metrics` | a question and a rubric | scores on four axes, plus what the trace shows |
| `2_conversation/1_conversation_simulation` | a goal and one persona | conversations |
| `2_conversation/2_goals_from_real_sessions` | existing conversations | goals and personas |
| `2_conversation/3_judging_conversations` | conversations | issues in words, scores in numbers |
| `2_conversation/4_test_agent` | only the agent | conversations and issues |
| `3_dataset_store/1_hand_written_records` | records with full expectations | a versioned, server-side dataset |
| `3_dataset_store/2_generated_records` | thinner and multi-turn records | one store holding every shape |

Read the middle column down each group and you get the same ladder: how much a
human has to write, falling toward none.

## Next

**[`../2_offline/`](../2_offline/)** — is this version good enough to ship?
