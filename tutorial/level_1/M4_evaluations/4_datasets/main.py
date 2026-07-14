"""L1-M4.4 — Datasets: create evaluation datasets, inspect schemas,
log with lineage, add human labels, and run LLM inference on datasets."""

import pandas as pd

import mlflow
import mlflow.data
from openai import OpenAI

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M4_evaluations/4_datasets"

QA_ROWS = [
    ("What is MLflow?", "An open-source platform for managing the ML lifecycle.", "overview", "easy"),
    ("What is an MLflow experiment?", "A named collection of runs for organizing work.", "tracking", "easy"),
    ("How do you log a metric?", "Use mlflow.log_metric(key, value) in an active run.", "tracking API", "medium"),
    ("What is the Model Registry?", "A centralized store for versioning models.", "registry", "medium"),
    ("What model format does MLflow use?", "The MLmodel format with per-framework flavors.", "models", "hard"),
    ("What is MLflow Tracing?", "Captures execution flow of LLM and agent calls.", "tracing", "hard"),
]

LABEL_ROWS = [
    ("MLflow is an open-source ML lifecycle platform.", "correct", "Accurate."),
    ("An experiment groups related runs together.", "correct", "Good."),
    ("Call mlflow.log_metric('accuracy', 0.95).", "correct", "Concrete example."),
    ("The Registry stores and versions models.", "partial", "Missing lifecycle detail."),
    ("Uses MLmodel format with flavors.", "correct", "Solid."),
    ("Records LLM call chains for debugging.", "partial", "Missing agent workflows."),
]


def ask_llm(client: OpenAI, question: str) -> str:
    """Send a question to the LLM and return the response text."""
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # Part 1: Creating an Evaluation Dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 1: Creating an Evaluation Dataset")
    print("=" * 60)

    qa_data = pd.DataFrame(QA_ROWS, columns=["question", "ground_truth_answer", "context", "difficulty"])

    print(f"  Created Q&A dataset with {len(qa_data)} entries")
    print(f"  Columns: {list(qa_data.columns)}")
    print(f"  Difficulty distribution:")
    for level, count in qa_data["difficulty"].value_counts().items():
        print(f"    {level}: {count}")
    print()

    with mlflow.start_run(run_name="datasets_demo") as run:
        run_id = run.info.run_id

        dataset = mlflow.data.from_pandas(
            qa_data, source="tutorial_qa_pairs",
            name="qa_evaluation_dataset", targets="ground_truth_answer",
        )
        mlflow.log_input(dataset, context="evaluation")
        print(f"  Logged dataset — name: {dataset.name}, digest: {dataset.digest}")
        print(f"  Schema: {dataset.schema}")
        print()

        hard_data = qa_data[qa_data["difficulty"] == "hard"].copy()
        hard_dataset = mlflow.data.from_pandas(
            hard_data, source="tutorial_qa_pairs",
            name="qa_hard_subset", targets="ground_truth_answer",
        )
        mlflow.log_input(hard_dataset, context="evaluation_subset")
        print(f"  Logged subset — name: {hard_dataset.name}, digest: {hard_dataset.digest}")
        print()

        # --------------------------------------------------------------
        # Part 2: LLM Inference on the Dataset
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 2: Running LLM Inference on the Dataset")
        print("=" * 60)

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        llm_answers = []
        for _, row in qa_data.iterrows():
            answer = ask_llm(client, row["question"])
            llm_answers.append(answer)
            print(f"  Q: {row['question']}")
            print(f"  A: {answer[:80]}...")
            print()

        results_df = qa_data.copy()
        results_df["llm_answer"] = llm_answers
        mlflow.log_table(results_df, artifact_file="inference_results.json")
        mlflow.log_param("model", "google/gemma-4-e4b")
        mlflow.log_param("temperature", 0.3)
        print(f"  Logged inference results as 'inference_results.json'")
        print()

        # --------------------------------------------------------------
        # Part 3: Adding Labels / Assessments
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 3: Adding Human Labels / Assessments")
        print("=" * 60)

        labels_data = pd.DataFrame({
            "question": qa_data["question"].tolist(),
            "model_answer": [r[0] for r in LABEL_ROWS],
            "human_label": [r[1] for r in LABEL_ROWS],
            "notes": [r[2] for r in LABEL_ROWS],
        })
        mlflow.log_table(labels_data, artifact_file="labels.json")

        print(f"  Logged {len(labels_data)} labels as 'labels.json'")
        for label, count in labels_data["human_label"].value_counts().items():
            print(f"    {label}: {count}")
        print()

    # ------------------------------------------------------------------
    # Part 4: Querying Dataset Lineage
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 4: Querying Dataset Lineage")
    print("=" * 60)

    logged_run = mlflow.get_run(run_id)
    for ds_input in logged_run.inputs.dataset_inputs:
        ds = ds_input.dataset
        ctx = {t.key: t.value for t in ds_input.tags}.get("mlflow.data.context", "N/A")
        print(f"  Dataset: {ds.name}")
        print(f"    Digest:  {ds.digest}")
        print(f"    Source:  {ds.source}")
        print(f"    Context: {ctx}")
        print()

    # ------------------------------------------------------------------
    # Part 5: Loading Labels Back for Analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 5: Loading Labels Back for Analysis")
    print("=" * 60)

    loaded_labels = mlflow.load_table("labels.json", run_ids=[run_id])
    total = len(loaded_labels)
    print("  Label summary:")
    for label in ["correct", "partial", "incorrect"]:
        count = int((loaded_labels["human_label"] == label).sum())
        print(f"    {label:12s}: {count}/{total} ({count / total:.0%})")

    accuracy = (loaded_labels["human_label"] == "correct").sum() / total
    print(f"\n  Overall accuracy: {accuracy:.1%}")
    print()

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Open MLflow UI at {TRACKING_URI}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print("  Check the Datasets tab, inference_results.json, and labels.json.")
    print("  In Level 2, we'll build full human-in-the-loop labeling pipelines.")


if __name__ == "__main__":
    main()
