"""
L1-M5.2 — GenAI Scorers and Judges

Demonstrates two approaches to evaluating LLM outputs:
- Custom scorers: deterministic Python functions (fast, reproducible)
- LLM judges: an LLM grades another LLM's output (nuanced, flexible)
- Combining both for a comprehensive evaluation view
"""

import json
import re

import mlflow
from langchain_openai import ChatOpenAI


# -- Sample Q&A data -------------------------------------------------------- #

QA_PAIRS = [
    {
        "question": "What is photosynthesis?",
        "expected": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "response": "Photosynthesis is how plants use sunlight to make food. They take in carbon dioxide and water, then produce glucose and oxygen using light energy.",
    },
    {
        "question": "What causes tides?",
        "expected": "Tides are caused primarily by the gravitational pull of the Moon and the Sun on Earth's oceans.",
        "response": "Tides happen because of the Moon.",
    },
    {
        "question": "Explain the water cycle.",
        "expected": "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection.",
        "response": "Water evaporates from oceans and lakes, rises to form clouds through condensation, falls back as rain or snow (precipitation), and collects in bodies of water. This cycle repeats continuously and is driven by solar energy.",
    },
    {
        "question": "What is gravity?",
        "expected": "Gravity is a fundamental force that attracts objects with mass toward each other.",
        "response": "Gravity is a force.",
    },
]


# -- Part 1: Custom scorer -------------------------------------------------- #

def custom_scorer(question: str, expected: str, response: str) -> dict[str, float]:
    """Deterministic scorer: length ratio, keyword overlap, detail depth."""
    # Length ratio: how much of the expected detail is covered
    len_ratio = min(len(response) / max(len(expected), 1), 1.5) / 1.5

    # Keyword overlap: fraction of expected keywords found in response
    expected_words = set(expected.lower().split())
    response_words = set(response.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                  "of", "to", "in", "on", "by", "it", "that", "this", "for"}
    expected_keywords = expected_words - stop_words
    overlap = len(expected_keywords & response_words) / max(len(expected_keywords), 1)

    # Detail depth: does the response have multiple clauses / sentences?
    sentence_count = len(re.split(r'[.!?]+', response.strip())) - 1
    detail = min(sentence_count / 3.0, 1.0)

    composite = round(0.4 * len_ratio + 0.4 * overlap + 0.2 * detail, 3)
    return {
        "length_ratio": round(len_ratio, 3),
        "keyword_overlap": round(overlap, 3),
        "detail_depth": round(detail, 3),
        "custom_composite": composite,
    }


# -- Part 2: LLM Judge ------------------------------------------------------ #

JUDGE_PROMPT = """\
You are an evaluation judge. Score the RESPONSE to the QUESTION on three criteria.
Use the EXPECTED answer as a reference for what a good answer looks like.

QUESTION: {question}
EXPECTED: {expected}
RESPONSE: {response}

Score each criterion from 0.0 to 1.0 and provide a brief justification.
Return ONLY valid JSON (no markdown, no code fences) in this exact format:
{{"relevance": <float>, "completeness": <float>, "clarity": <float>, "justification": "<short text>"}}
"""


def llm_judge(
    llm: ChatOpenAI, question: str, expected: str, response: str
) -> dict[str, float]:
    """Use an LLM to judge response quality on relevance, completeness, clarity."""
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, response=response
    )
    raw = llm.invoke(prompt).content.strip()

    # Try to extract JSON from the response
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
        else:
            print(f"    WARNING: Could not parse judge output, using defaults")
            scores = {"relevance": 0.5, "completeness": 0.5, "clarity": 0.5,
                      "justification": "parse error"}

    return {
        "judge_relevance": float(scores.get("relevance", 0.5)),
        "judge_completeness": float(scores.get("completeness", 0.5)),
        "judge_clarity": float(scores.get("clarity", 0.5)),
        "judge_justification": scores.get("justification", ""),
    }


# -- Part 3: Combine and evaluate ------------------------------------------- #

def main() -> None:
    llm = ChatOpenAI(
        model="google/gemma-4-e4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.0,
    )

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Part 1: Custom Scorer (deterministic)")
    print("=" * 60)

    for i, qa in enumerate(QA_PAIRS):
        scores = custom_scorer(qa["question"], qa["expected"], qa["response"])
        print(f"\n  Q{i+1}: {qa['question']}")
        for k, v in scores.items():
            print(f"    {k:<20s}: {v:.3f}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Part 2: LLM Judge (google/gemma-4-e4b)")
    print("=" * 60)

    for i, qa in enumerate(QA_PAIRS):
        judge_scores = llm_judge(llm, qa["question"], qa["expected"], qa["response"])
        print(f"\n  Q{i+1}: {qa['question']}")
        for k, v in judge_scores.items():
            if k != "judge_justification":
                print(f"    {k:<20s}: {v:.3f}")
            else:
                print(f"    {'justification':<20s}: {v}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Part 3: Combined evaluation — logged to MLflow")
    print("=" * 60)

    all_results = []
    with mlflow.start_run(run_name="combined_scorer_judge"):
        for i, qa in enumerate(QA_PAIRS):
            cs = custom_scorer(qa["question"], qa["expected"], qa["response"])
            js = llm_judge(llm, qa["question"], qa["expected"], qa["response"])

            row = {"question": qa["question"], **cs, **{k: v for k, v in js.items() if k != "judge_justification"}}
            all_results.append(row)

            # Log per-question metrics
            for key in ["custom_composite", "judge_relevance", "judge_completeness", "judge_clarity"]:
                mlflow.log_metric(f"q{i+1}_{key}", row.get(key, 0.0))

        # Log averages
        for key in ["custom_composite", "judge_relevance", "judge_completeness", "judge_clarity"]:
            avg = sum(r[key] for r in all_results) / len(all_results)
            mlflow.log_metric(f"avg_{key}", round(avg, 3))

    # Print comparison table
    print(f"\n  {'Question':<30s} {'Custom':>8s} {'Relev':>8s} {'Compl':>8s} {'Clarity':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        q = r["question"][:28] + (".." if len(r["question"]) > 28 else "")
        print(f"  {q:<30s} {r['custom_composite']:>8.3f} {r['judge_relevance']:>8.3f} "
              f"{r['judge_completeness']:>8.3f} {r['judge_clarity']:>8.3f}")

    avgs = {key: sum(r[key] for r in all_results) / len(all_results)
            for key in ["custom_composite", "judge_relevance", "judge_completeness", "judge_clarity"]}
    print(f"  {'AVERAGE':<30s} {avgs['custom_composite']:>8.3f} {avgs['judge_relevance']:>8.3f} "
          f"{avgs['judge_completeness']:>8.3f} {avgs['judge_clarity']:>8.3f}")

    print("\n" + "=" * 60)
    print("Done! View results in MLflow UI: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M5_genai_features/2_scorers_judges")
    main()
