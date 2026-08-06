"""
L1-M4.1.1 -- Evaluation Fundamentals

Combines three evaluation approaches using one shared dataset:
- Part 1: mlflow.genai.evaluate() with built-in + custom @scorer
- Part 2: Manual LLM-as-Judge pattern (structured prompt, JSON output)
- Part 3: Custom deterministic scorer functions (keyword overlap, detail)
- Part 4: Combined view -- all scores side by side, logged to MLflow
"""

import json
import re

import mlflow
import pandas as pd
from mlflow.genai.scorers import ResponseLength, scorer
from openai import OpenAI

# -- Configuration --
# The LiteLLM gateway from infra/, not a provider directly. The aliases below are
# defined in infra/litellm/config.yaml, which also owns the fallback order and
# each model's context window. Swapping model or provider is a change there,
# never here.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin

MODEL_NAME = "gemma-chat"

client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)

# -- Shared evaluation dataset --------------------------------------------- #

QA_PAIRS = [
    {
        "question": "What is the capital of France?",
        "expected_response": "Paris",
        "ground_truth": "The capital of France is Paris.",
    },
    {
        "question": "What is photosynthesis?",
        "expected_response": "plants convert sunlight",
        "ground_truth": (
            "Photosynthesis is the process by which green plants convert "
            "sunlight, water, and carbon dioxide into glucose and oxygen."
        ),
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "expected_response": "William Shakespeare",
        "ground_truth": "William Shakespeare wrote Romeo and Juliet.",
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "expected_response": "100",
        "ground_truth": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    },
    {
        "question": "What causes tides?",
        "expected_response": "gravitational pull of the Moon",
        "ground_truth": "Tides are caused primarily by the gravitational pull of the Moon and the Sun on Earth's oceans.",
    },
]


def answer_question(question: str) -> str:
    """Ask the LLM a factual question and return its answer."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


@scorer
def contains_expected(inputs, outputs, expectations) -> bool:
    """Check whether the LLM output contains the expected answer text."""
    expected = expectations.get("expected_response", "")
    return expected.lower() in outputs.lower()


def part1_genai_evaluate() -> None:
    """Run mlflow.genai.evaluate() with built-in and custom scorers."""
    print("=" * 60)
    print("Part 1: mlflow.genai.evaluate() with scorers")
    print("=" * 60)

    eval_data = pd.DataFrame(
        [
            {
                "inputs": {"question": qa["question"]},
                "expectations": {"expected_response": qa["expected_response"]},
            }
            for qa in QA_PAIRS
        ]
    )

    print(f"  Dataset: {len(eval_data)} questions")
    print("  Scorers: ResponseLength(1-500 words), contains_expected")
    print("  Running evaluation...\n")

    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=answer_question,
        scorers=[
            ResponseLength(min_length=1, max_length=500, unit="words"),  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
            contains_expected,
        ],
    )

    print("  --- Aggregate Metrics ---")
    for name, value in results.metrics.items():
        print(f"    {name}: {value}")

    print("\n  --- Per-Row Results ---")
    table = results.result_df
    if table is not None:
        for i, (_, row) in enumerate(table.iterrows()):
            q = (row.get("request") or {}).get("question", "N/A")
            a = str(row.get("response", ""))
            if len(a) > 80:
                a = a[:77] + "..."
            print(f"\n    Q{i + 1}: {q}")
            print(f"       Answer: {a}")
            for c in table.columns:
                if c.endswith(("/value", "/rationale")):
                    print(f"       {c}: {row[c]}")
    print()


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


def judge_answer(question: str, ground_truth: str, model_answer: str) -> dict:
    """Ask the judge model to score the student's answer."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    text = (response.choices[0].message.content or "").strip()

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


def part2_llm_as_judge() -> list[dict]:
    """Run LLM-as-Judge evaluation manually."""
    print("=" * 60)
    print("Part 2: LLM-as-Judge (manual pattern)")
    print("=" * 60)

    results = []
    for i, qa in enumerate(QA_PAIRS, 1):
        model_answer = answer_question(qa["question"])
        verdict = judge_answer(qa["question"], qa["ground_truth"], model_answer)

        print(f"\n  Q{i}: {qa['question']}")
        print(f"    Student answer : {model_answer[:120]}")
        print(f"    Judge score    : {verdict['score']}/5")
        print(f"    Justification  : {verdict['justification']}")

        results.append(
            {
                "question": qa["question"],
                "model_answer": model_answer,
                "judge_score": verdict["score"],
                "justification": verdict["justification"],
            }
        )

    avg_score = sum(r["judge_score"] for r in results) / len(results)
    print(f"\n  Average judge score: {avg_score:.2f} / 5")
    print()
    return results


def custom_scorer(expected: str, response: str) -> dict[str, float]:
    """Deterministic scorer: length ratio, keyword overlap, detail depth."""
    len_ratio = min(len(response) / max(len(expected), 1), 1.5) / 1.5

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "by",
        "it",
        "that",
        "this",
        "for",
    }
    expected_kw = set(expected.lower().split()) - stop_words
    response_kw = set(response.lower().split()) - stop_words
    overlap = len(expected_kw & response_kw) / max(len(expected_kw), 1)

    sentence_count = len(re.split(r"[.!?]+", response.strip())) - 1
    detail = min(sentence_count / 3.0, 1.0)

    composite = round(0.4 * len_ratio + 0.4 * overlap + 0.2 * detail, 3)
    return {
        "keyword_overlap": round(overlap, 3),
        "detail_depth": round(detail, 3),
        "custom_composite": composite,
    }


def part3_custom_scorers(judge_results: list[dict]) -> None:
    """Run deterministic custom scorers on the same dataset."""
    print("=" * 60)
    print("Part 3: Custom deterministic scorers")
    print("=" * 60)

    for i, (qa, jr) in enumerate(zip(QA_PAIRS, judge_results), 1):
        scores = custom_scorer(qa["ground_truth"], jr["model_answer"])
        print(f"\n  Q{i}: {qa['question']}")
        for k, v in scores.items():
            print(f"    {k:<20s}: {v:.3f}")
    print()


def part4_combined(judge_results: list[dict]) -> None:
    """Combine all scoring approaches and log to MLflow."""
    print("=" * 60)
    print("Part 4: Combined evaluation -- logged to MLflow")
    print("=" * 60)

    all_rows = []
    with mlflow.start_run(run_name="combined_evaluation"):
        mlflow.log_param("student_model", MODEL_NAME)
        mlflow.log_param("judge_model", MODEL_NAME)
        mlflow.log_param("num_questions", len(QA_PAIRS))

        for i, (qa, jr) in enumerate(zip(QA_PAIRS, judge_results)):
            cs = custom_scorer(qa["ground_truth"], jr["model_answer"])
            row = {
                "question": qa["question"],
                "judge_score": jr["judge_score"] / 5.0,
                "keyword_overlap": cs["keyword_overlap"],
                "detail_depth": cs["detail_depth"],
                "custom_composite": cs["custom_composite"],
            }
            all_rows.append(row)

            mlflow.log_metric(f"q{i + 1}_judge", row["judge_score"])
            mlflow.log_metric(f"q{i + 1}_custom", row["custom_composite"])

        # Log averages
        for key in ["judge_score", "keyword_overlap", "custom_composite"]:
            avg = sum(r[key] for r in all_rows) / len(all_rows)
            mlflow.log_metric(f"avg_{key}", round(avg, 3))

    # Print comparison table
    print(f"\n  {'Question':<35s} {'Judge':>7s} {'Keywords':>9s} {'Custom':>8s}")
    print(f"  {'-' * 35} {'-' * 7} {'-' * 9} {'-' * 8}")
    for r in all_rows:
        q = r["question"][:33] + (".." if len(r["question"]) > 33 else "")
        print(f"  {q:<35s} {r['judge_score']:>7.2f} {r['keyword_overlap']:>9.3f} {r['custom_composite']:>8.3f}")

    avgs = {
        k: sum(r[k] for r in all_rows) / len(all_rows) for k in ["judge_score", "keyword_overlap", "custom_composite"]
    }
    print(
        f"  {'AVERAGE':<35s} {avgs['judge_score']:>7.2f} "
        f"{avgs['keyword_overlap']:>9.3f} {avgs['custom_composite']:>8.3f}"
    )
    print()


def main() -> None:
    part1_genai_evaluate()
    judge_results = part2_llm_as_judge()
    part3_custom_scorers(judge_results)
    part4_combined(judge_results)

    print("=" * 60)
    print("Done! See results in MLflow UI at http://127.0.0.1:5555")
    print("Experiment: 'L1/M4_evaluation/1_fundamentals/1_evaluation_fundamentals'")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L1/M4_evaluation/1_fundamentals/1_evaluation_fundamentals")

    main()
