# L2-M2.1.3 — The Dataset Store

The two groups before this one **produce** test material. This one **keeps** it.

Hand-written cases live in `main.py`. Distilled goals live in a variable. Both
disappear when the process exits — nothing else can load them, nobody can tell
which version produced last month's numbers, and CI has nothing to run.

An `EvaluationDataset` is where they live instead: named, versioned,
server-side, loadable by name.

## Who wrote a record decides what you can assert, and nothing else

```text
   hand-written  ─┐
   distilled     ─┼──▶  merge_records()  ──▶  ONE dataset
   generated     ─┤                             (versioned by tag)
   multi-turn    ─┘
```

`merge_records` never asks where a record came from. It is an **upsert**, keyed
on the inputs, so running it twice adds nothing — which is what makes it safe to
call from CI on every commit.

| Record came from | Its `expectations` holds | So it is |
|:--|:--|:--|
| Hand-written tests | the answer **and** the tools | compared against ground truth |
| A distilled goal | the tools only | route compared, answer judged |
| `test_agent` | nothing at all | judged only, never compared |

## The two lessons

| # | Lesson | Teaches |
|:--|:--|:--|
| 1 | [`1_hand_written_records`](1_hand_written_records/) | The store API — `create_dataset`, `merge_records`, `to_df`, `search_datasets`, tag versioning — on the richest record shape |
| 2 | [`2_generated_records`](2_generated_records/) | The thinner shapes, the **multi-turn** record, and the two kinds of "test case" the store holds |

Neither lesson makes an LLM call. They are about the store, not the agent.

## The multi-turn record

`inputs` is a message list, not a prompt string, precisely so a stored case can
begin partway through a conversation:

```python
{
    "inputs": {
        "messages": [
            {"role": "user", "content": "What is the status of order A1001?"},
            {"role": "assistant", "content": "Order A1001 has shipped, arriving Thursday."},
            {"role": "user", "content": "And if it turns up damaged, who pays to send it back?"},
        ]
    },
    "expectations": {"contains": "free", "expected_tools": ["policy_lookup"]},
}
```

The expectation is about **this** turn. The two before it are context.

## Two kinds of test case, one store

This is the collision that costs the most confusion, and lesson 2 proves it by
running:

| Kind | `inputs` holds | Feeds |
|:--|:--|:--|
| **Evaluation record** | `messages`, plus `expectations` | `mlflow.genai.evaluate()` |
| **Simulator scenario** | `goal`, `persona` | `ConversationSimulator` |

They are not interchangeable. `ConversationSimulator` reads
`to_df()["inputs"]` and demands a `goal` key — hand it an evaluation dataset and
it raises:

```text
EvaluationDataset passed to ConversationSimulator must contain
conversational test cases with a 'goal' field in the 'inputs' column
```

MLflow calls both of them `test_cases`. The store holds both the same way.

## Next

**[`../../2_offline/`](../../2_offline/)** — running these datasets as a
ship gate.
