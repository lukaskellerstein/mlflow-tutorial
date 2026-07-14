"""
L1-M4.2 — LLM-as-Judge Evaluation

Uses one LLM (the "judge") to evaluate the quality of answers
produced by another LLM (the "student"). Both roles use the same
small model here (google/gemma-4-e4b) to keep things fast; in practice
the judge would be a larger, more capable model.
"""

import json
import re

import mlflow
import pandas as pd
from openai import OpenAI

# -- Configuration --
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M4_evaluations/2_llm_as_judge")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ---------------------------------------------------------------------------
# Evaluation dataset: question, ground_truth pairs
# ---------------------------------------------------------------------------
EVAL_DATA = [
    {
        "question": "What is the capital of France?",
        "ground_truth": "The capital of France is Paris.",
    },
    {
        "question": "What is photosynthesis?",
        "ground_truth": (
            "Photosynthesis is the process by which green plants convert "
            "sunlight, water, and carbon dioxide into glucose and oxygen."
        ),
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "ground_truth": "William Shakespeare wrote Romeo and Juliet.",
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "ground_truth": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    },
]

JUDGE_PROMPT = """\
You are an impartial evaluation judge. Compare the model's answer to the
ground-truth answer and rate correctness on a scale of 1-5:
  1 = completely wrong
  2 = mostly wrong with a minor correct element
  3 = partially correct but missing key information
  4 = mostly correct with minor omissions
  5 = fully correct

Question: {question}
Ground-truth answer: {ground_truth}
Model answer: {model_answer}

Output ONLY valid JSON with two keys:
  "score": an integer from 1 to 5
  "justification": a one-sentence explanation

Do NOT include any text outside the JSON object.
"""


def generate_answer(question: str) -> str:
    """Ask the student model a question and return its answer."""
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": f"Answer concisely: {question}"}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def judge_answer(question: str, ground_truth: str, model_answer: str) -> dict:
    """Ask the judge model to score the student's answer."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    text = response.choices[0].message.content.strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "score": int(parsed.get("score", 3)),
                "justification": str(parsed.get("justification", "No justification.")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {"score": 3, "justification": f"Could not parse judge response: {text[:100]}"}


def main() -> None:
    """Run LLM-as-Judge evaluation and log results to MLflow."""
    results = []

    print()
    print("=" * 60)
    print("LLM-as-Judge Evaluation")
    print("=" * 60)

    for i, item in enumerate(EVAL_DATA, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n--- Q{i}: {question}")

        model_answer = generate_answer(question)
        print(f"  Student answer : {model_answer[:120]}")

        verdict = judge_answer(question, ground_truth, model_answer)
        print(f"  Judge score    : {verdict['score']}/5")
        print(f"  Justification  : {verdict['justification']}")

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "score": verdict["score"],
                "justification": verdict["justification"],
            }
        )

    # ---- Log everything to MLflow ----
    df = pd.DataFrame(results)
    avg_score = df["score"].mean()

    print("\n" + "=" * 60)
    print(f"Average judge score: {avg_score:.2f} / 5")
    print("=" * 60)

    with mlflow.start_run(run_name="llm_as_judge_eval"):
        mlflow.log_param("student_model", "google/gemma-4-e4b")
        mlflow.log_param("judge_model", "google/gemma-4-e4b")
        mlflow.log_param("num_questions", len(EVAL_DATA))

        for i, row in df.iterrows():
            mlflow.log_metric(f"q{i + 1}_score", row["score"])

        mlflow.log_metric("avg_score", avg_score)

        mlflow.log_table(df, artifact_file="evaluation_results.json")

        print("\nResults logged to MLflow.")
        print("  Metrics: per-question scores + average")
        print("  Artifact: evaluation_results.json")

    print()
    print("Open the MLflow UI to review:")
    print("  http://127.0.0.1:5000")
    print("  Experiment: L1/M4_evaluations/2_llm_as_judge")
    print("=" * 60)


if __name__ == "__main__":
    main()
