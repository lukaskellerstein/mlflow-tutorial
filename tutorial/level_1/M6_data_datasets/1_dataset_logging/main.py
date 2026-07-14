"""
L1-M7.1 -- Dataset Logging and Lineage

Log an LLM evaluation dataset, run inference on the questions,
and link the dataset to the evaluation run for data lineage.
"""

import mlflow
import pandas as pd
from openai import OpenAI

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M6_data_datasets/1_dataset_logging")


def ask_llm(client: OpenAI, question: str) -> str:
    """Send a question to the LLM and return the response text."""
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 -- Create a Q&A evaluation dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Creating a Q&A evaluation dataset")
    print("=" * 60)

    qa_data = pd.DataFrame({
        "question": [
            "What is the capital of France?",
            "What programming language is MLflow written in?",
            "How many planets are in our solar system?",
            "What does API stand for?",
            "What is the boiling point of water in Celsius?",
        ],
        "expected_answer": [
            "Paris",
            "Python",
            "8",
            "Application Programming Interface",
            "100 degrees Celsius",
        ],
        "category": [
            "geography",
            "technology",
            "science",
            "technology",
            "science",
        ],
    })

    print(f"  Dataset size: {len(qa_data)} Q&A pairs")
    print(f"  Columns: {list(qa_data.columns)}")
    print(f"  Categories: {qa_data['category'].unique().tolist()}")
    print()

    # ------------------------------------------------------------------
    # Step 2 -- Create MLflow datasets from the DataFrame
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Creating MLflow datasets")
    print("=" * 60)

    eval_dataset = mlflow.data.from_pandas(
        qa_data,
        source="manual_qa_pairs",
        targets="expected_answer",
        name="llm_eval_qa",
    )
    print(f"  Name:   {eval_dataset.name}")
    print(f"  Digest: {eval_dataset.digest}")
    print(f"  Schema: {eval_dataset.schema}")
    print()

    # Also create a subset for a second context demonstration
    tech_data = qa_data[qa_data["category"] == "technology"].copy()
    tech_dataset = mlflow.data.from_pandas(
        tech_data,
        source="manual_qa_pairs",
        targets="expected_answer",
        name="llm_eval_qa_tech",
    )
    print(f"  Subset: name={tech_dataset.name}, "
          f"digest={tech_dataset.digest}")
    print()

    # ------------------------------------------------------------------
    # Step 3 -- Run LLM inference and log with dataset lineage
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Running LLM inference and logging results")
    print("=" * 60)

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    with mlflow.start_run(run_name="llm_eval_with_dataset") as run:
        # Link datasets to this run
        mlflow.log_input(eval_dataset, context="evaluation")
        mlflow.log_input(tech_dataset, context="evaluation_subset")
        print("  Logged evaluation dataset (context='evaluation')")
        print("  Logged tech subset (context='evaluation_subset')")

        # Run inference on each question
        llm_answers = []
        correct_count = 0
        for i, row in qa_data.iterrows():
            answer = ask_llm(client, row["question"])
            llm_answers.append(answer)

            # Simple exact-match check (case-insensitive)
            expected = row["expected_answer"].lower()
            is_correct = expected in answer.lower()
            if is_correct:
                correct_count += 1

            print(f"  Q: {row['question']}")
            print(f"  A: {answer[:80]}...")
            print(f"  Expected: {row['expected_answer']}  "
                  f"Match: {'yes' if is_correct else 'no'}")
            print()

        # Log metrics
        accuracy = correct_count / len(qa_data)
        mlflow.log_metric("num_questions", len(qa_data))
        mlflow.log_metric("correct_count", correct_count)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("model", "google/gemma-4-e4b")
        mlflow.log_param("temperature", 0.3)
        print(f"  Accuracy: {correct_count}/{len(qa_data)} = {accuracy:.1%}")

        # Log the results table as an artifact
        results_df = qa_data.copy()
        results_df["llm_answer"] = llm_answers
        results_df["correct"] = [
            exp.lower() in ans.lower()
            for exp, ans in zip(qa_data["expected_answer"], llm_answers)
        ]
        mlflow.log_table(results_df, artifact_file="eval_results.json")
        print(f"  Results table logged as artifact")
        print(f"  Run ID: {run.info.run_id}")
    print()

    # ------------------------------------------------------------------
    # Step 4 -- Query dataset lineage from the completed run
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Querying dataset lineage from the completed run")
    print("=" * 60)

    run_data = mlflow.get_run(run.info.run_id)
    for ds_input in run_data.inputs.dataset_inputs:
        ds = ds_input.dataset
        tags = {t.key: t.value for t in ds_input.tags}
        ctx = tags.get("mlflow.data.context", "N/A")
        print(f"  Dataset: {ds.name}")
        print(f"    Digest:  {ds.digest}")
        print(f"    Source:  {ds.source}")
        print(f"    Context: {ctx}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done! View dataset lineage in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("Open the run and look for the Datasets section.")
    print("=" * 60)


if __name__ == "__main__":
    main()
