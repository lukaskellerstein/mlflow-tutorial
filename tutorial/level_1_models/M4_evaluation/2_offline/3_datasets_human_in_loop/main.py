"""L1-M4.2.3 — Datasets and Human-in-the-Loop Evaluation

Combines dataset creation, logging, and lineage tracking with
human-in-the-loop assessment workflows:
- Create and log evaluation datasets with mlflow.data
- Run LLM inference and log results
- Attach human assessments (Feedback + Expectation) to traces
- Combine auto-judge pre-screening with human review triage
- Query dataset lineage and load labels for analysis
"""

import json
import random
import time
from typing import cast

import mlflow
import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import AssessmentSource, AssessmentSourceType
from openai import OpenAI

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L1/M4_evaluation/2_offline/3_datasets_human_in_loop")

QA_ROWS = [
    ("What is MLflow?", "An open-source platform for managing the ML lifecycle."),
    ("What is an MLflow experiment?", "A named collection of runs for organizing work."),
    ("How do you log a metric?", "Use mlflow.log_metric(key, value) in an active run."),
    ("What is the Model Registry?", "A centralized store for versioning models."),
    ("What is MLflow Tracing?", "Captures execution flow of LLM and agent calls."),
]


def ask_llm(client: OpenAI, question: str) -> str:
    """Send a question to the LLM and return the response text."""
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


def auto_judge_score(client: OpenAI, question: str, expected: str, answer: str) -> dict:
    """Use LLM-as-judge for automated pre-screening. Returns score 1-5."""
    prompt = (
        "You are a strict evaluation judge. Compare the actual answer to the expected answer.\n"
        "Score 1-5 where 1=completely wrong, 3=partially correct, 5=fully correct.\n\n"
        f"Question: {question}\nExpected: {expected}\nActual: {answer}\n\n"
        'Reply ONLY with valid JSON: {"score": <number 1-5>, "reasoning": "<why>"}'
    )
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-e4b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        text = (response.choices[0].message.content or "").strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return {
                "score": max(1, min(5, int(parsed.get("score", 3)))),
                "reasoning": str(parsed.get("reasoning", "No reasoning.")),
            }
    except Exception as e:
        print(f"    Judge error: {e}")
    return {"score": 3, "reasoning": "Judge could not produce a valid score."}


# ── Part 1: Create and Log an Evaluation Dataset ──────────────────────────


def part1_create_and_log_dataset(client: OpenAI) -> tuple[str, list[dict]]:
    """Create a dataset, log it with lineage, run inference, return results."""
    print("=" * 60)
    print("Part 1: Create, Log, and Run Inference on a Dataset")
    print("=" * 60)

    qa_data = pd.DataFrame(QA_ROWS, columns=pd.Index(["question", "ground_truth_answer"]))
    print(f"  Created Q&A dataset with {len(qa_data)} entries")

    with mlflow.start_run(run_name="dataset_and_inference") as run:
        run_id = run.info.run_id

        # Log the dataset with lineage
        dataset = from_pandas(
            qa_data,
            source="tutorial_qa_pairs",
            name="qa_evaluation_dataset",
            targets="ground_truth_answer",
        )
        mlflow.log_input(dataset, context="evaluation")
        print(f"  Logged dataset -- name: {dataset.name}, digest: {dataset.digest}")
        print(f"  Schema: {dataset.schema}")

        # Run LLM inference on each question
        results = []
        for _, row in qa_data.iterrows():

            @mlflow.trace(name=f"qa_{row['question'][:30]}")
            def traced_qa(question: str, expected: str) -> dict:
                answer = ask_llm(client, question)
                return {"question": question, "expected": expected, "answer": answer}

            result = traced_qa(str(row["question"]), str(row["ground_truth_answer"]))
            result["trace_id"] = mlflow.get_last_active_trace_id()
            results.append(result)
            print(f"  Q: {row['question']}")
            print(f"  A: {result['answer'][:80]}...")
            print()

        # Log inference results as a table artifact
        results_df = cast(pd.DataFrame, pd.DataFrame(results)[["question", "expected", "answer"]])
        mlflow.log_table(results_df, artifact_file="inference_results.json")
        mlflow.log_param("model", "google/gemma-4-e4b")
        print("  Logged inference results as 'inference_results.json'")

    mlflow.flush_trace_async_logging()
    time.sleep(2)
    return run_id, results


# ── Part 2: Human Assessments via MLflow Assessment API ───────────────────


SIMULATED_REVIEWS = [
    {"label": "correct", "confidence": 0.95, "notes": "Accurate definition."},
    {"label": "correct", "confidence": 0.90, "notes": "Good explanation."},
    {"label": "partial", "confidence": 0.60, "notes": "Missing concrete example."},
    {"label": "partial", "confidence": 0.65, "notes": "Missing lifecycle detail."},
    {"label": "correct", "confidence": 0.85, "notes": "Captures the concept well."},
]


def part2_human_assessments(results: list[dict]) -> None:
    """Attach simulated human assessments (Feedback + Expectation) to traces."""
    print("=" * 60)
    print("Part 2: Attach Human Assessments to Traces")
    print("=" * 60)

    human_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="reviewer@example.com",
    )

    with mlflow.start_run(run_name="human_assessments"):
        for i, (result, review) in enumerate(zip(results, SIMULATED_REVIEWS)):
            trace_id = result.get("trace_id")
            if not trace_id:
                continue

            mlflow.log_expectation(
                trace_id=trace_id,
                name="expected_answer",
                value=result["expected"],
                source=human_source,
            )
            mlflow.log_feedback(
                trace_id=trace_id,
                name="human_correctness",
                value=review["label"],
                source=human_source,
                rationale=review["notes"],
                metadata={"confidence": str(review["confidence"])},
            )
            print(f"  Q{i + 1}: label={review['label']}, confidence={review['confidence']}")

        summary_df = pd.DataFrame(SIMULATED_REVIEWS)
        mlflow.log_table(summary_df, artifact_file="assessments.json")
        label_counts = summary_df["label"].value_counts().to_dict()
        for label, count in label_counts.items():
            mlflow.log_metric(f"count_{label}", count)
        print(f"\n  Assessment summary: {label_counts}")


# ── Part 3: Combined Auto-Judge + Human Triage ────────────────────────────


def part3_combined_evaluation(client: OpenAI, results: list[dict]) -> None:
    """Auto-judge pre-screens; borderline cases go to human review."""
    print("\n" + "=" * 60)
    print("Part 3: Combined Auto-Judge + Human Triage")
    print("=" * 60)
    print("  Workflow: auto-judge -> triage -> human review borderline cases\n")

    with mlflow.start_run(run_name="combined_evaluation"):
        counts = {"auto_approved": 0, "auto_rejected": 0, "borderline_human_review": 0}

        for i, result in enumerate(results):
            trace_id = result.get("trace_id")
            judge = auto_judge_score(client, result["question"], result["expected"], result["answer"])
            score = judge["score"]

            auto_feedback = None
            if trace_id:
                auto_feedback = mlflow.log_feedback(
                    trace_id=trace_id,
                    name="auto_judge_score",
                    value=score,
                    source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="google/gemma-4-e4b"),
                    rationale=judge["reasoning"],
                )

            if score >= 4:
                verdict = "AUTO_APPROVED"
                counts["auto_approved"] += 1
            elif score <= 2:
                verdict = "AUTO_REJECTED"
                counts["auto_rejected"] += 1
            else:
                counts["borderline_human_review"] += 1
                human_verdict = random.choice(["correct", "incorrect"])
                if trace_id and auto_feedback and auto_feedback.assessment_id:
                    mlflow.override_feedback(
                        trace_id=trace_id,
                        assessment_id=auto_feedback.assessment_id,
                        value=4 if human_verdict == "correct" else 2,
                        rationale=f"Human override: {human_verdict}",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN,
                            source_id="senior_reviewer@example.com",
                        ),
                    )
                verdict = f"HUMAN_OVERRIDE -> {human_verdict}"

            print(f"  Q{i + 1}: auto_score={score}, verdict={verdict}")
            mlflow.log_metric(f"q{i + 1}_auto_score", score)

        mlflow.log_metrics({k: float(v) for k, v in counts.items()})
        total = len(results)
        print(f"\n  Triage summary ({total} items):")
        for k, v in counts.items():
            print(f"    {k:25s} {v}")


# ── Part 4: Query Dataset Lineage ─────────────────────────────────────────


def part4_query_lineage(run_id: str) -> None:
    """Query dataset lineage and load labels for analysis."""
    print("\n" + "=" * 60)
    print("Part 4: Query Dataset Lineage and Labels")
    print("=" * 60)

    logged_run = mlflow.get_run(run_id)
    for ds_input in logged_run.inputs.dataset_inputs:
        ds = ds_input.dataset
        ctx = {t.key: t.value for t in ds_input.tags}.get("mlflow.data.context", "N/A")
        print(f"  Dataset: {ds.name}")
        print(f"    Digest:  {ds.digest}")
        print(f"    Context: {ctx}")

    loaded_results = mlflow.load_table("inference_results.json", run_ids=[run_id])
    print(f"\n  Loaded {len(loaded_results)} inference results from run {run_id[:8]}...")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    run_id, results = part1_create_and_log_dataset(client)
    part2_human_assessments(results)
    part3_combined_evaluation(client, results)
    part4_query_lineage(run_id)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("  Open MLflow UI at http://127.0.0.1:5555")
    print("  Experiment: L1/M4_evaluation/2_offline/3_datasets_human_in_loop")
    print("  Check: Datasets tab, inference_results.json, trace assessments")


if __name__ == "__main__":
    main()
