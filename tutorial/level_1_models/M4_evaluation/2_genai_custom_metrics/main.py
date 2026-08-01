"""
L1-M4.2 -- GenAI Custom Metrics

Combines custom metric building with batch evaluation across configs:
- Part 1: Deterministic scorer returning Feedback (rich metadata)
- Part 2: LLM-based scorer returning Feedback (judge quality)
- Part 3: Additional custom scorers (keyword_coverage, conciseness, has_example)
- Part 4: Batch evaluation across two LLM configurations
- Part 5: Cross-config comparison and threshold gates
"""

import json
import math
import re

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.scorers import ResponseLength, scorer
from openai import OpenAI

# -- Configuration --
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
MODEL_NAME = "google/gemma-4-e4b"

# -- Shared evaluation dataset -------------------------------------------- #

EVAL_DATA = pd.DataFrame(
    [
        {
            "inputs": {"question": "What are Python decorators?"},
            "expectations": {
                "expected_response": "Decorators are functions that modify the behavior of other functions or classes. They use the @syntax and are commonly used for logging, access control, and caching."
            },
        },
        {
            "inputs": {"question": "Explain list comprehensions in Python."},
            "expectations": {
                "expected_response": "List comprehensions provide a concise way to create lists. The syntax is [expression for item in iterable if condition]. They are more readable and often faster than equivalent for loops."
            },
        },
        {
            "inputs": {"question": "What is the GIL in Python?"},
            "expectations": {
                "expected_response": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. It limits true parallelism in CPU-bound multi-threaded programs."
            },
        },
        {
            "inputs": {"question": "How does garbage collection work in Python?"},
            "expectations": {
                "expected_response": "Python uses reference counting as its primary garbage collection mechanism. When an object's reference count drops to zero, it is deallocated. A cyclic garbage collector handles reference cycles that reference counting alone cannot resolve."
            },
        },
        {
            "inputs": {"question": "What is a generator in Python?"},
            "expectations": {
                "expected_response": "Generators are functions that use yield to produce a sequence of values lazily. They maintain state between calls and are memory-efficient for large datasets because they generate values on demand rather than storing them all in memory."
            },
        },
    ]
)


@scorer
def formatting_quality(outputs, expectations) -> Feedback:
    """Check response quality: sentence count, keyword overlap, length."""
    text = str(outputs)
    expected = str(expectations.get("expected_response", ""))

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_score = min(len(sentences) / 3.0, 1.0)

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
        "with",
        "as",
        "at",
        "from",
        "they",
        "be",
        "its",
    }
    expected_kw = set(expected.lower().split()) - stop_words
    response_kw = set(text.lower().split()) - stop_words
    overlap = len(expected_kw & response_kw) / max(len(expected_kw), 1)

    word_count = len(text.split())
    if word_count < 10:
        length_score = 0.2
    elif word_count < 20:
        length_score = 0.6
    elif word_count <= 150:
        length_score = 1.0
    else:
        length_score = 0.7

    composite = round(0.3 * sentence_score + 0.4 * overlap + 0.3 * length_score, 3)
    return Feedback(
        value=composite,
        rationale=(
            f"sentences={len(sentences)}, keyword_overlap={overlap:.2f}, word_count={word_count}"
        ),
        source=AssessmentSource(source_type="CODE", source_id="formatting_quality"),
    )


JUDGE_PROMPT = """\
You are an expert evaluator. Score the RESPONSE to the QUESTION for technical \
accuracy and completeness. Use the EXPECTED answer as reference.

QUESTION: {question}
EXPECTED: {expected}
RESPONSE: {response}

Score from 0.0 to 1.0 on each criterion:
- accuracy: Are the facts correct?
- completeness: Does it cover the key points from the expected answer?
- clarity: Is it clearly written and easy to understand?

Return ONLY valid JSON (no markdown fences):
{{"accuracy": <float>, "completeness": <float>, "clarity": <float>, "rationale": "<one sentence>"}}
"""


@scorer
def llm_technical_quality(inputs, outputs, expectations) -> Feedback:
    """Use LLM judge to assess technical accuracy and completeness."""
    llm_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    prompt = JUDGE_PROMPT.format(
        question=inputs.get("question", ""),
        expected=expectations.get("expected_response", ""),
        response=str(outputs),
    )
    resp = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = (resp.choices[0].message.content or "").strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        scores = (
            json.loads(match.group())
            if match
            else {
                "accuracy": 0.5,
                "completeness": 0.5,
                "clarity": 0.5,
                "rationale": "Could not parse judge output",
            }
        )

    avg = round(
        (
            float(scores.get("accuracy", 0.5))
            + float(scores.get("completeness", 0.5))
            + float(scores.get("clarity", 0.5))
        )
        / 3.0,
        3,
    )
    return Feedback(
        value=avg,
        rationale=(
            f"accuracy={scores.get('accuracy')}, "
            f"completeness={scores.get('completeness')}, "
            f"clarity={scores.get('clarity')} | "
            f"{scores.get('rationale', '')}"
        ),
        source=AssessmentSource(source_type="LLM_JUDGE", source_id=MODEL_NAME),
    )


@scorer(name="keyword_coverage")
def keyword_coverage(inputs: dict, outputs: str) -> Feedback:
    """Check whether the answer references key terms from the question."""
    question = inputs.get("question", "").lower()
    answer = str(outputs).lower()
    stopwords = {
        "what",
        "does",
        "how",
        "when",
        "this",
        "that",
        "with",
        "from",
        "have",
        "they",
        "their",
        "about",
        "which",
        "will",
        "been",
        "explain",
        "describe",
        "python",
    }
    keywords = [w for w in question.split() if len(w) > 3 and w.strip("'\"?,.") not in stopwords]
    if not keywords:
        return Feedback(value=1.0, rationale="No keywords to check.")
    hits = sum(1 for kw in keywords if kw.strip("'\"?,.") in answer)
    score = hits / len(keywords)
    return Feedback(value=round(score, 2), rationale=f"{hits}/{len(keywords)} keywords found.")


@scorer(name="answer_conciseness")
def answer_conciseness(outputs: str) -> Feedback:
    """Score how concise the answer is (prefer 30-150 words)."""
    wc = len(str(outputs).split())
    if wc < 10:
        return Feedback(value=0.2, rationale=f"Too short ({wc} words).")
    if wc <= 30:
        return Feedback(value=0.6, rationale=f"Brief ({wc} words).")
    if wc <= 150:
        return Feedback(value=1.0, rationale=f"Good length ({wc} words).")
    if wc <= 250:
        return Feedback(value=0.7, rationale=f"Somewhat verbose ({wc} words).")
    return Feedback(value=0.4, rationale=f"Too verbose ({wc} words).")


@scorer(name="has_example")
def has_example(outputs: str) -> bool:
    """Check whether the answer includes a concrete example or code snippet."""
    text = str(outputs).lower()
    indicators = [
        "for example",
        "e.g.",
        "such as",
        "```",
        ">>>",
        "= [",
        "= {",
        "def ",
        "class ",
        "import ",
    ]
    return any(ind in text for ind in indicators)


ALL_SCORERS = [
    ResponseLength(min_length=20, max_length=500, unit="words"),  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
    formatting_quality,
    llm_technical_quality,
    keyword_coverage,
    answer_conciseness,
    has_example,
]


CONFIGS = [
    {"name": "temp_0.3", "temperature": 0.3},
    {"name": "temp_0.9", "temperature": 0.9},
]


def build_predict_fn(temperature: float):
    """Return a predict function with the given temperature."""

    def predict_fn(question: str) -> str:
        llm_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable Python tutor. "
                    "Answer clearly and concisely. Include a short example when appropriate.",
                },
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    return predict_fn


def evaluate_config(config: dict) -> dict:
    """Evaluate a single LLM configuration and return its metrics."""
    print(f"\n  Evaluating: {config['name']} (temperature={config['temperature']})")

    predict_fn = build_predict_fn(config["temperature"])

    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "temperature": config["temperature"],
                "num_questions": len(EVAL_DATA),
            }
        )
        result = mlflow.genai.evaluate(
            data=EVAL_DATA,
            predict_fn=predict_fn,
            scorers=ALL_SCORERS,
        )
        for name, value in result.metrics.items():
            mlflow.log_metric(name, value)

    print("    Metrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"      {name}: {value:.4f}")
    return result.metrics


def compare_and_check(all_metrics: dict[str, dict]) -> None:
    """Print comparison table and check quality thresholds."""
    print("\n" + "=" * 60)
    print("Part 5: Results Comparison and Threshold Gates")
    print("=" * 60)

    names = list(all_metrics.keys())
    metrics = sorted(set().union(*(m.keys() for m in all_metrics.values())))

    hdr = f"  {'Metric':<35}" + "".join(f" {n:>12}" for n in names) + f" {'Best':>12}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    wins: dict[str, int] = {n: 0 for n in names}
    for m in metrics:
        vals = {n: all_metrics[n].get(m, float("nan")) for n in names}
        valid = {k: v for k, v in vals.items() if not math.isnan(v)}
        best = max(valid, key=lambda k: valid[k]) if valid else "n/a"
        if best in wins:
            wins[best] += 1
        row = f"  {m:<35}" + "".join(f" {vals[n]:>12.4f}" for n in names)
        print(f"{row} {best:>12}")

    overall = max(wins, key=lambda k: wins[k])
    print(f"\n  Overall best: {overall}")

    # Threshold gates
    print("\n  --- Threshold Gates ---")
    thresholds = {"formatting_quality/mean": 0.4, "llm_technical_quality/mean": 0.5}
    for config_name, metrics_dict in all_metrics.items():
        print(f"\n  Config: {config_name}")
        for metric_name, min_value in thresholds.items():
            actual = metrics_dict.get(metric_name, 0.0)
            passed = actual >= min_value
            print(
                f"    [{'PASS' if passed else 'FAIL'}] {metric_name}: "
                f"{actual:.3f} (threshold: {min_value})"
            )
    print()


def main() -> None:
    print("=" * 60)
    print("L1-M4.2 -- GenAI Custom Metrics")
    print("=" * 60)

    # Show dataset
    print(f"\n  Evaluation dataset: {len(EVAL_DATA)} questions")
    for i, (_, row) in enumerate(EVAL_DATA.iterrows()):
        print(f"    Q{i + 1}: {row['inputs']['question']}")

    # Show scorers
    print(f"\n  Scorers ({len(ALL_SCORERS)}):")
    for s in ALL_SCORERS:
        print(f"    - {s.name}")

    # Part 4: Batch evaluation
    print("\n" + "=" * 60)
    print("Batch evaluation across configurations")
    print("=" * 60)

    all_metrics: dict[str, dict] = {}
    for config in CONFIGS:
        all_metrics[config["name"]] = evaluate_config(config)

    # Part 5: Comparison and thresholds
    compare_and_check(all_metrics)

    print("=" * 60)
    print("Done! View results in MLflow UI: http://127.0.0.1:5555")
    print("Experiment: L1/M4_evaluation/2_genai_custom_metrics")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L1/M4_evaluation/2_genai_custom_metrics")
    main()
