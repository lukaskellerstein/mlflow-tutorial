"""L2-M2.1.3.2 -- The Dataset Store: Generated and Multi-Turn Records.

1_hand_written_records stored the richest shape: a person knew the answer AND
the route, so every record could be compared against ground truth.

Records that nobody wrote are thinner, and the store holds them all the same
way. That is the point of this lesson -- who produced a record changes what you
can ASSERT about it, and changes nothing about how it is stored.

Three shapes, and what each one buys you:

    hand-written    {"contains": ..., "expected_tools": [...]}   compared
    distilled goal  {"expected_tools": [...]}                    route compared,
                                                                 answer judged
    generated       {}                                           judged only

And one shape no other lesson in the module shows: a MULTI-TURN record, whose
`inputs` carries a conversation already in progress rather than a single
question. That is what lets a stored case start at turn 3.

This lesson makes no LLM calls. It is about the store, not the agent.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.genai.datasets import (
    create_dataset,
    delete_dataset,
    get_dataset,
    search_datasets,
    set_dataset_tags,
)

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/3_dataset_store/2_generated_records"
DATASET_NAME = "support_agent_regression"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id


def user_turn(text: str) -> dict[str, Any]:
    """One record's `inputs`: a conversation of exactly one user message."""
    return {"messages": [{"role": "user", "content": text}]}


def conversation(*messages: tuple[str, str]) -> dict[str, Any]:
    """One record's `inputs`: a conversation ALREADY IN PROGRESS.

    `inputs` is a message list, not a prompt string, precisely so a stored case
    can begin partway through. Nothing else in this module shows this shape.
    """
    return {"messages": [{"role": role, "content": text} for role, text in messages]}


# -- The shape from lesson 1, so the merge has something to merge INTO -------- #
HAND_WRITTEN: list[dict[str, Any]] = [
    {
        "inputs": user_turn("What is the status of order A1001?"),
        "expectations": {"contains": "shipped", "expected_tools": ["order_status"]},
    },
    {
        "inputs": user_turn("How long do I have to return something?"),
        "expectations": {"contains": "30 days", "expected_tools": ["policy_lookup"]},
    },
]

# -- Produced by 2_conversation/2_goals_from_real_sessions -------------------- #
# A goal, not a question. There is no single right answer to compare against, so
# the expectation is about the ROUTE only. A judge grades the rest.
DISTILLED_GOALS: list[dict[str, Any]] = [
    {
        "inputs": user_turn("Find out whether order A1002 will arrive this week, and why it is delayed"),
        "expectations": {"expected_tools": ["order_status"]},
    },
    {
        "inputs": user_turn("Return a faulty item bought two months ago and find out who pays return shipping"),
        "expectations": {"expected_tools": ["policy_lookup"]},
    },
]

# -- Produced by 2_conversation/4_test_agent ---------------------------------- #
# No expectations at all. Nobody knows the right answer -- these probe the edges,
# and what you assert about them is "did not hallucinate", which is a judge's job.
GENERATED: list[dict[str, Any]] = [
    {"inputs": user_turn("What is the status of order Z9999?"), "expectations": {}},
    {"inputs": user_turn("Can you tell me the delivery address on order A1001?"), "expectations": {}},
    {"inputs": user_turn("What is your policy on price matching a competitor?"), "expectations": {}},
]

# -- The shape that only exists once conversations do ------------------------- #
# Turn 3 of a real conversation, stored so it can be replayed from there. The
# expectation is about this turn only -- the two before it are context.
MULTI_TURN: list[dict[str, Any]] = [
    {
        "inputs": conversation(
            ("user", "What is the status of order A1001?"),
            ("assistant", "Order A1001 has shipped and is arriving Thursday."),
            ("user", "And if it turns up damaged, who pays to send it back?"),
        ),
        "expectations": {"contains": "free", "expected_tools": ["policy_lookup"]},
    },
    {
        "inputs": conversation(
            ("user", "How long is the warranty?"),
            ("assistant", "The warranty is 12 months against manufacturing defects (P-204)."),
            ("user", "Remind me which policy code that was?"),
        ),
        "expectations": {"contains": "P-204"},
    },
]


# -- A DIFFERENT kind of record entirely -------------------------------------- #
# Everything above is an EVALUATION record: a question plus what to assert about
# the answer. This is a SIMULATOR SCENARIO: a goal, with no right answer at all.
# MLflow calls both of them "test cases" and the store holds both, but they are
# not interchangeable -- step 4 proves it by trying.
SCENARIOS: list[dict[str, Any]] = [
    {
        "inputs": {
            "goal": "Find out whether order A1002 will arrive this week, and why it is delayed",
            "persona": "An impatient customer who asks short, blunt follow-up questions",
        }
    },
    {
        "inputs": {
            "goal": "Return a faulty item and find out who pays the return shipping",
            "persona": "A polite first-time customer who does not know the policy names",
        }
    },
]
SCENARIO_DATASET_NAME = "support_agent_scenarios"


def show(dataset: Any, label: str) -> int:
    """Print the dataset's record count."""
    df = dataset.to_df()
    print(f"  {label:<32} {len(df):>2} records")
    return len(df)


def turns_of(inputs: Any) -> list[Any]:
    """The message list inside a record's `inputs`, or empty if it has none."""
    return inputs.get("messages", []) if isinstance(inputs, dict) else []


def describe_shape(expectations: Any) -> str:
    """Name the record shape from what its expectations carry."""
    if not expectations:
        return "generated     -> judged only"
    keys = set(expectations)
    if "contains" in keys and "expected_tools" in keys:
        return "hand-written  -> compared"
    if "expected_tools" in keys:
        return "distilled     -> route compared"
    return "partial       -> judged in part"


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.3.2  The Dataset Store -- Generated and Multi-Turn Records")
    print("=" * 70)

    # -- 0. Start clean, so re-running the lesson teaches the same thing ------ #
    removed = 0
    for name in (DATASET_NAME, SCENARIO_DATASET_NAME):
        for old in search_datasets(experiment_ids=[EXPERIMENT_ID], filter_string=f"name = '{name}'"):
            delete_dataset(dataset_id=old.dataset_id)
            removed += 1
    if removed:
        print(f"\n  removed {removed} dataset(s) from an earlier run")

    # -- 1. Seed with the hand-written shape --------------------------------- #
    print("\nStep 1: start from the hand-written records")
    dataset = create_dataset(
        name=DATASET_NAME,
        experiment_id=EXPERIMENT_ID,
        tags={"version": "1", "owner": "support-quality", "lesson": "L2-M2.1.3.2"},
    )
    dataset.merge_records(HAND_WRITTEN)
    show(dataset, "hand-written:")

    # -- 2. Add what nobody wrote -------------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 2: merge the records nobody wrote")
    print("=" * 70)
    dataset.merge_records(DISTILLED_GOALS)
    show(dataset, "+ distilled goals:")
    dataset.merge_records(GENERATED)
    show(dataset, "+ generated cases:")
    dataset.merge_records(MULTI_TURN)
    total = show(dataset, "+ multi-turn:")

    print("\n  One store, one call, four producers. `merge_records` did not ask")
    print("  where any of them came from, and it did not need to.")

    # -- 3. The three shapes, side by side ----------------------------------- #
    print("\n" + "=" * 70)
    print("Step 3: what each shape lets you assert")
    print("=" * 70)
    df = dataset.to_df()
    print(f"\n  {'turns':>5}  {'shape -> what you can do':<32} first message")
    print(f"  {'-' * 5}  {'-' * 32} {'-' * 24}")
    for _, row in df.iterrows():
        inputs = row["inputs"]
        msgs = turns_of(inputs)
        first = str(msgs[0].get("content", "")) if msgs else str(inputs)
        print(f"  {len(msgs):>5}  {describe_shape(row.get('expectations')):<32} {first[:24]}")

    with_full = sum(1 for _, r in df.iterrows() if r.get("expectations") and "contains" in dict(r["expectations"]))
    multi = sum(1 for _, r in df.iterrows() if len(turns_of(r["inputs"])) > 1)
    print(f"\n  {with_full}/{len(df)} records can be compared against a right answer.")
    print(f"  {len(df) - with_full}/{len(df)} must be judged instead.")
    print(f"  {multi}/{len(df)} start partway through a conversation.")

    # -- 4. Two kinds of "test case", one store ------------------------------- #
    print("\n" + "=" * 70)
    print("Step 4: the store holds TWO kinds of test case")
    print("=" * 70)
    from mlflow.genai.simulators import ConversationSimulator

    scenario_dataset = create_dataset(
        name=SCENARIO_DATASET_NAME,
        experiment_id=EXPERIMENT_ID,
        tags={"kind": "simulator_scenarios", "lesson": "L2-M2.1.3.2"},
    )
    scenario_dataset.merge_records(SCENARIOS)
    show(scenario_dataset, "simulator scenarios:")

    # Constructing the simulator VALIDATES its input, so both checks below are
    # free -- no LLM call, no .simulate(). It reads to_df()["inputs"] and
    # demands a "goal" key, which is the whole difference between the two kinds.
    for label, candidate in (("evaluation records", dataset), ("simulator scenarios", scenario_dataset)):
        try:
            ConversationSimulator(test_cases=candidate, max_turns=2)
            print(f"\n  ConversationSimulator({label:<20}) -> ACCEPTED")
        except ValueError as exc:
            print(f"\n  ConversationSimulator({label:<20}) -> REJECTED")
            print(f"    {' '.join(str(exc).split())[:66]}")

    print("\n  Both are stored the same way. Neither can do the other's job:")
    print("    inputs = {messages, ...} + expectations -> mlflow.genai.evaluate()")
    print("    inputs = {goal, persona}                -> ConversationSimulator")
    print("\n  MLflow calls both 'test_cases'. A simulator case holds a GOAL, not a")
    print("  right answer, so it can never pass or fail -- it produces a")
    print("  conversation that something else then judges.")

    print("\n  scenarios -> ConversationSimulator -> conversations -> scorers -> numbers")
    print("       ^                                                              |")
    print("       +----------- merge_records() puts the good goals back ---------+")

    # -- 5. Version and record ------------------------------------------------ #
    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={"version": "2", "sources": "hand_written,distilled,generated,multi_turn"},
    )
    reloaded = get_dataset(dataset_id=dataset.dataset_id)
    print(f"\n  tags now : {reloaded.tags}")

    with mlflow.start_run(run_name="build_mixed_dataset"):
        mlflow.log_params({"dataset": DATASET_NAME, "dataset_id": dataset.dataset_id})
        mlflow.log_metrics(
            {
                "records_total": total,
                "records_hand_written": len(HAND_WRITTEN),
                "records_distilled": len(DISTILLED_GOALS),
                "records_generated": len(GENERATED),
                "records_multi_turn": len(MULTI_TURN),
                "records_comparable": with_full,
            }
        )

    print("\n" + "=" * 70)
    print(f"  '{DATASET_NAME}' holds {total} records from 4 producers")
    print("\n  Who wrote a record decides what you can assert about it.")
    print("  It decides nothing about how the record is stored.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
