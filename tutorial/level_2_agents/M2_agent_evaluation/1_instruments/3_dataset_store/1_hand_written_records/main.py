"""L2-M2.1.3.1 -- The Dataset Store: Hand-Written Records.

Every lesson before this one leaves its test cases in a Python list, which dies
with the script. Nothing else can load them, nobody can tell which version
produced last month's numbers, and CI has nothing to run.

An `EvaluationDataset` is where they live instead: named, versioned, server-side.

This lesson owns the store API and the RICHEST record shape -- the one written
by a person who knew both the answer and the route:

    inputs       = one user turn
    expectations = {"contains": ..., "expected_tools": [...]}

That shape is the only one you can compare against ground truth, which is why
it is the one that yields pass or fail. The thinner shapes, and the multi-turn
record, are 2_generated_records.

Why an agent dataset is not a model dataset (L1-M4.2.3):

  - `inputs` is a MESSAGE LIST, not a prompt string, because an agent case can
    start mid-conversation.
  - `expectations` covers WHICH TOOLS should be called, not only what the answer
    should contain. Half of agent failure is right answer, wrong route.

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

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/3_dataset_store/1_hand_written_records"
DATASET_NAME = "support_agent_regression"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id


def user_turn(text: str) -> dict[str, Any]:
    """One record's `inputs`: the conversation the agent is handed."""
    return {"messages": [{"role": "user", "content": text}]}


# -- Source 1: the hand-written cases from L2-M2.1.1.1 ------------------------- #
# Full expectations -- a person knew the answer AND the route.
HAND_WRITTEN: list[dict[str, Any]] = [
    {
        "inputs": user_turn("What is the status of order A1001?"),
        "expectations": {"contains": "shipped", "expected_tools": ["order_status"]},
    },
    {
        "inputs": user_turn("Has order A1003 arrived yet?"),
        "expectations": {"contains": "delivered", "expected_tools": ["order_status"]},
    },
    {
        "inputs": user_turn("Is order A1002 on its way?"),
        "expectations": {"contains": "payment", "expected_tools": ["order_status"]},
    },
    {
        "inputs": user_turn("How long do I have to return something?"),
        "expectations": {"contains": "30 days", "expected_tools": ["policy_lookup"]},
    },
    {
        "inputs": user_turn("What does the warranty cover and for how long?"),
        "expectations": {"contains": "12 months", "expected_tools": ["policy_lookup"]},
    },
    {
        "inputs": user_turn("My item is faulty. Who pays for the return shipping?"),
        "expectations": {"contains": "free", "expected_tools": ["policy_lookup"]},
    },
]


def show(dataset: Any, label: str) -> int:
    """Print the dataset's record count and its inferred schema."""
    df = dataset.to_df()
    print(f"  {label:<28} {len(df):>2} records")
    return len(df)


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.3.1  The Dataset Store -- Hand-Written Records")
    print("=" * 70)

    # -- 0. Start clean, so re-running the lesson teaches the same thing ----- #
    existing = search_datasets(experiment_ids=[EXPERIMENT_ID], filter_string=f"name = '{DATASET_NAME}'")
    for old in existing:
        delete_dataset(dataset_id=old.dataset_id)
    if existing:
        print(f"\n  removed {len(existing)} dataset(s) from an earlier run")

    # -- 1. Create ----------------------------------------------------------- #
    print("\nStep 1: create the dataset")
    dataset = create_dataset(
        name=DATASET_NAME,
        experiment_id=EXPERIMENT_ID,
        tags={"version": "1", "owner": "support-quality", "lesson": "L2-M2.1.3.1"},
    )
    print(f"  name       : {dataset.name}")
    print(f"  dataset_id : {dataset.dataset_id}")
    print(f"  tags       : {dataset.tags}")

    # -- 2. Merge the three sources ------------------------------------------ #
    print("\n" + "=" * 70)
    print("Step 2: merge the hand-written records")
    print("=" * 70)
    dataset.merge_records(HAND_WRITTEN)
    total = show(dataset, "after hand-written:")

    # -- 3. Merge is an upsert, not an append -------------------------------- #
    print("\n" + "=" * 70)
    print("Step 3: merge the SAME records again")
    print("=" * 70)
    dataset.merge_records(HAND_WRITTEN)
    after = show(dataset, "after re-merging 6:")
    print(f"\n  {total} -> {after}. `merge_records` keys on the inputs, so a record")
    print("  that is already there is updated, not duplicated. That is what makes")
    print("  it safe to run in CI on every commit.")

    # -- 4. What a record looks like ----------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 4: the record shape")
    print("=" * 70)
    df = dataset.to_df()
    print(f"  columns: {list(df.columns)}")
    for _, row in df.head(3).iterrows():
        inputs = row["inputs"]
        expectations = row.get("expectations")
        text = inputs["messages"][0]["content"] if isinstance(inputs, dict) else str(inputs)
        print(f"\n  inputs       : {text[:64]}")
        print(f"  expectations : {expectations}")

    with_expectations = sum(1 for _, r in df.iterrows() if r.get("expectations"))
    print(f"\n  {with_expectations}/{len(df)} records carry FULL expectations --")
    print("  the answer and the route. That is what makes them comparable against")
    print("  ground truth, and comparable is what yields pass or fail.")
    print("\n  A record with thinner expectations, or none at all, is not broken.")
    print("  It is judged instead of compared. 2_generated_records has both.")

    # -- 5. Versioning is tags, not copies ----------------------------------- #
    print("\n" + "=" * 70)
    print("Step 5: version it")
    print("=" * 70)
    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={"version": "2", "sources": "hand_written"},
    )
    reloaded = get_dataset(dataset_id=dataset.dataset_id)
    print(f"  tags now : {reloaded.tags}")
    print("\n  Tags are merged, not replaced -- 'owner' survived. Set a tag to None")
    print("  to remove it, or use delete_dataset_tag().")

    # -- 6. Find it again ----------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Step 6: find it without knowing its id")
    print("=" * 70)
    found = search_datasets(
        experiment_ids=[EXPERIMENT_ID],
        filter_string="tags.owner = 'support-quality'",
        max_results=5,
    )
    for d in found:
        print(f"  {d.name:<28} {d.dataset_id}  v{d.tags.get('version') if d.tags else '?'}")
    print("\n  Always pass experiment_ids or a filter. A bare search_datasets()")
    print("  returns every dataset on the tracking server.")

    # -- 7. Record the run ---------------------------------------------------- #
    with mlflow.start_run(run_name="build_regression_dataset"):
        mlflow.log_params({"dataset": DATASET_NAME, "dataset_id": dataset.dataset_id})
        mlflow.log_metrics(
            {
                "records_total": len(df),
                "records_hand_written": len(HAND_WRITTEN),
                "records_with_expectations": with_expectations,
            }
        )

    print("\n" + "=" * 70)
    print(f"  '{DATASET_NAME}' holds {len(df)} hand-written records")
    print("\n  Any later lesson loads it by name instead of rewriting it:")
    print("    get_dataset(dataset_id=...) or search_datasets(filter_string=...)")
    print("  It also drops straight into ConversationSimulator(test_cases=dataset)")
    print("  and mlflow.genai.evaluate(data=dataset, ...).")
    print("\n  Next: 2_generated_records adds the records nobody wrote -- thinner")
    print("  expectations, and the multi-turn shape this lesson never showed.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
