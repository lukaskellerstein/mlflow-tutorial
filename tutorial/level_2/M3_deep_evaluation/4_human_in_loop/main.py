"""
L2-3.4 — Human-in-the-Loop Evaluation

Demonstrates human-in-the-loop evaluation workflows with MLflow:
- Generate model outputs and create traces for review
- Attach human assessments (Feedback + Expectation) to traces
- Combine automated LLM-judge pre-screening with human review
- Build a feedback loop that improves evaluation datasets over rounds
"""

import json
import random
import time

import mlflow
import ollama
import pandas as pd
from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "L2/M3_deep_evaluation/4_human_in_loop"

QA_PAIRS = [
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "Explain photosynthesis in one sentence.",
     "expected": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
    {"question": "What is 15 * 17?", "expected": "255"},
    {"question": "Name three programming languages used for data science.",
     "expected": "Python, R, and Julia are commonly used for data science."},
    {"question": "What year did the first moon landing occur?", "expected": "1969"},
]


def generate_answer(question: str) -> str:
    """Call Ollama to generate an answer."""
    response = ollama.chat(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.7, "num_predict": 100},
        think=False,
    )
    return response["message"]["content"].strip()


def auto_judge_score(question: str, expected: str, answer: str) -> dict:
    """Use LLM-as-judge for automated pre-screening. Returns score 1-5 + reasoning."""
    prompt = (
        "You are a strict evaluation judge. Compare the actual answer to the expected answer.\n"
        "Score 1-5 where 1=completely wrong, 3=partially correct, 5=fully correct.\n\n"
        f"Question: {question}\nExpected: {expected}\nActual: {answer}\n\n"
        'Reply ONLY with valid JSON: {"score": <number 1-5>, "reasoning": "<why>"}'
    )
    try:
        response = ollama.chat(
            model="gemma4:e2b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 80},
            format="json", think=False,
        )
        text = response["message"]["content"].strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return {
                "score": max(1, min(5, int(parsed.get("score", 3)))),
                "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            }
    except Exception as e:
        print(f"    Judge error: {e}")
    return {"score": 3, "reasoning": "Judge could not produce a valid score."}


# ===================================================================
# Part 1: Generate model outputs and create traces for review
# ===================================================================
def part1_generate_outputs() -> list[dict]:
    """Generate LLM answers for each Q&A pair, creating traced runs."""
    print("=" * 60)
    print("PART 1: Generate Model Outputs for Review")
    print("=" * 60)

    results = []
    with mlflow.start_run(run_name="part1_generate_outputs"):
        for i, qa in enumerate(QA_PAIRS):
            @mlflow.trace(name=f"qa_pair_{i}")
            def traced_qa(question: str, expected: str) -> dict:
                answer = generate_answer(question)
                return {"question": question, "expected": expected, "answer": answer}

            result = traced_qa(qa["question"], qa["expected"])
            result["trace_id"] = mlflow.get_last_active_trace_id()
            results.append(result)
            print(f"  Q{i+1}: {qa['question']}")
            print(f"      Answer: {result['answer'][:80]}...")
            print(f"      Trace:  {result['trace_id']}")

        mlflow.log_table(
            pd.DataFrame(results)[["question", "expected", "answer"]],
            artifact_file="outputs.json",
        )
        print(f"\n  Logged {len(results)} Q&A outputs as table artifact")

    # Flush async trace logging so traces are available for assessments
    mlflow.flush_trace_async_logging()
    time.sleep(2)
    return results


# ===================================================================
# Part 2: Simulate human assessments using MLflow Assessment API
# ===================================================================
SIMULATED_REVIEWS = [
    {"label": "correct",  "confidence": 0.95, "notes": "Exact match with expected answer."},
    {"label": "partial",  "confidence": 0.60, "notes": "Captures the concept but misses detail."},
    {"label": "correct",  "confidence": 0.90, "notes": "Numerically correct answer provided."},
    {"label": "correct",  "confidence": 0.85, "notes": "Lists valid languages for data science."},
    {"label": "correct",  "confidence": 0.95, "notes": "Correct year stated."},
]


def part2_human_assessments(results: list[dict]) -> list[dict]:
    """Attach simulated human assessments to each trace."""
    print("\n" + "=" * 60)
    print("PART 2: Simulate Human Assessments")
    print("=" * 60)

    assessed = []
    human_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="reviewer@example.com",
    )

    with mlflow.start_run(run_name="part2_human_assessments"):
        for i, (result, review) in enumerate(zip(results, SIMULATED_REVIEWS)):
            trace_id = result["trace_id"]
            if not trace_id:
                continue

            # Log ground truth as an Expectation
            mlflow.log_expectation(
                trace_id=trace_id, name="expected_answer",
                value=result["expected"], source=human_source,
            )
            # Log human correctness judgment as Feedback
            mlflow.log_feedback(
                trace_id=trace_id, name="human_correctness",
                value=review["label"], source=human_source,
                rationale=review["notes"],
                metadata={"confidence": str(review["confidence"])},
            )
            # Log confidence as a separate numeric feedback
            mlflow.log_feedback(
                trace_id=trace_id, name="human_confidence",
                value=review["confidence"], source=human_source,
            )
            print(f"  Q{i+1}: label={review['label']}, "
                  f"confidence={review['confidence']}, notes={review['notes'][:50]}")
            assessed.append({**result, **review})

        # Log summary metrics
        summary_df = pd.DataFrame(assessed)
        mlflow.log_table(summary_df[["question", "label", "confidence", "notes"]],
                         artifact_file="assessments.json")
        label_counts = summary_df["label"].value_counts().to_dict()
        for label, count in label_counts.items():
            mlflow.log_metric(f"count_{label}", count)
        mlflow.log_metric("avg_confidence", summary_df["confidence"].mean())
        print(f"\n  Assessment summary: {label_counts}")
        print(f"  Average confidence: {summary_df['confidence'].mean():.2f}")

    return assessed


# ===================================================================
# Part 3: Combine automated LLM-judge + human review
# ===================================================================
def part3_combined_evaluation(results: list[dict]) -> None:
    """Auto-judge pre-screens; borderline cases (score 3) go to human review."""
    print("\n" + "=" * 60)
    print("PART 3: Combined Automated + Human Evaluation")
    print("=" * 60)
    print("  Workflow: auto-judge -> flag borderline -> human review -> final verdict\n")

    with mlflow.start_run(run_name="part3_combined_evaluation"):
        counts = {"auto_approved": 0, "auto_rejected": 0, "borderline_human_review": 0}

        for i, result in enumerate(results):
            trace_id = result["trace_id"]
            judge = auto_judge_score(result["question"], result["expected"], result["answer"])
            score = judge["score"]

            # Log automated feedback on the trace
            auto_feedback = None
            if trace_id:
                auto_feedback = mlflow.log_feedback(
                    trace_id=trace_id, name="auto_judge_score", value=score,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.LLM_JUDGE, source_id="gemma4:e2b"),
                    rationale=judge["reasoning"],
                )

            # Triage: auto-approve high, auto-reject low, human-review borderline
            if score >= 4:
                verdict = "AUTO_APPROVED"
                counts["auto_approved"] += 1
            elif score <= 2:
                verdict = "AUTO_REJECTED"
                counts["auto_rejected"] += 1
            else:
                counts["borderline_human_review"] += 1
                human_verdict = random.choice(["correct", "incorrect"])
                if trace_id and auto_feedback:
                    mlflow.override_feedback(
                        trace_id=trace_id,
                        assessment_id=auto_feedback.assessment_id,
                        value=4 if human_verdict == "correct" else 2,
                        rationale=f"Human override: marked as {human_verdict}",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN,
                            source_id="senior_reviewer@example.com"),
                    )
                verdict = f"HUMAN_OVERRIDE -> {human_verdict}"

            print(f"  Q{i+1}: auto_score={score}, verdict={verdict}")
            print(f"        reasoning: {judge['reasoning'][:60]}")
            mlflow.log_metric(f"q{i+1}_auto_score", score)

        mlflow.log_metrics(counts)
        total = len(results)
        print(f"\n  Triage summary ({total} items):")
        for k, v in counts.items():
            print(f"    {k:25s} {v}")
        print(f"    {'human_review_rate':25s} {counts['borderline_human_review']/total*100:.0f}%")


# ===================================================================
# Part 4: Build feedback loop -- grow the evaluation dataset
# ===================================================================
def part4_feedback_loop(results: list[dict]) -> None:
    """Show how reviewed examples improve the evaluation dataset over rounds."""
    print("\n" + "=" * 60)
    print("PART 4: Feedback Loop -- Growing the Evaluation Dataset")
    print("=" * 60)

    eval_dataset = [{"question": r["question"], "expected": r["expected"]} for r in results]
    new_examples = [
        {"question": "What is the boiling point of water in Celsius?", "expected": "100"},
        {"question": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare"},
        {"question": "What is the square root of 144?", "expected": "12"},
    ]

    with mlflow.start_run(run_name="part4_feedback_loop"):
        for round_num in range(1, 4):
            print(f"\n  --- Evaluation Round {round_num} ---")
            if round_num > 1 and new_examples:
                new_ex = new_examples.pop(0)
                eval_dataset.append(new_ex)
                print(f"  Added new example: '{new_ex['question']}'")

            with mlflow.start_run(run_name=f"round_{round_num}", nested=True):
                mlflow.log_param("round", round_num)
                mlflow.log_param("dataset_size", len(eval_dataset))

                correct = sum(
                    1 for item in eval_dataset
                    if auto_judge_score(item["question"], item["expected"],
                                        generate_answer(item["question"]))["score"] >= 4
                )
                total = len(eval_dataset)
                accuracy = correct / total if total > 0 else 0
                mlflow.log_metrics({"dataset_size": total, "correct_count": correct,
                                    "accuracy": accuracy})
                mlflow.log_table(pd.DataFrame(eval_dataset),
                                 artifact_file=f"eval_dataset_round_{round_num}.json")
                print(f"  Dataset size: {total}, Accuracy: {accuracy:.1%} ({correct}/{total})")

        print(f"\n  Final evaluation dataset: {len(eval_dataset)} examples")
        print("  In production, human-reviewed corrections feed back into the dataset")
        print("  to continuously improve evaluation coverage.")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}\n")

    results = part1_generate_outputs()
    part2_human_assessments(results)
    part3_combined_evaluation(results)
    part4_feedback_loop(results)

    print("\n" + "=" * 60)
    print("LESSON COMPLETE")
    print("=" * 60)
    print("Check the MLflow UI at http://127.0.0.1:5000")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("\nKey takeaways:")
    print("  - mlflow.log_feedback() attaches human/LLM judgments to traces")
    print("  - mlflow.log_expectation() records ground truth on traces")
    print("  - mlflow.override_feedback() lets humans correct automated scores")
    print("  - Automated judges handle clear cases; humans review borderline ones")
    print("  - Feedback loops grow evaluation datasets over time")


if __name__ == "__main__":
    main()
