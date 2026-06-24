"""
L2-M3.1 — Custom Metrics and Evaluators

Builds on L1-M4.2 (LLM Eval Basics) and L1-M6.2 (Scorers & Judges).
Custom deterministic scorers, LLM-based scorers returning structured
Feedback, combining scorers in mlflow.genai.evaluate(), and threshold gates.
"""

import json
import re

import mlflow
import pandas as pd
from langchain_ollama import ChatOllama
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.scorers import scorer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L2/M3_deep_evaluation/1_custom_metrics")

# -- Evaluation dataset ---------------------------------------------------- #
EVAL_DATA = pd.DataFrame([
    {"inputs": {"question": "What are Python decorators?"},
     "expectations": {"expected_response": "Decorators are functions that modify the behavior of other functions or classes. They use the @syntax and are commonly used for logging, access control, and caching."}},
    {"inputs": {"question": "Explain list comprehensions in Python."},
     "expectations": {"expected_response": "List comprehensions provide a concise way to create lists. The syntax is [expression for item in iterable if condition]. They are more readable and often faster than equivalent for loops."}},
    {"inputs": {"question": "What is the GIL in Python?"},
     "expectations": {"expected_response": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. It limits true parallelism in CPU-bound multi-threaded programs."}},
    {"inputs": {"question": "How does garbage collection work in Python?"},
     "expectations": {"expected_response": "Python uses reference counting as its primary garbage collection mechanism. When an object's reference count drops to zero, it is deallocated. A cyclic garbage collector handles reference cycles that reference counting alone cannot resolve."}},
    {"inputs": {"question": "What is a generator in Python?"},
     "expectations": {"expected_response": "Generators are functions that use yield to produce a sequence of values lazily. They maintain state between calls and are memory-efficient for large datasets because they generate values on demand rather than storing them all in memory."}},
    {"inputs": {"question": "Explain Python's MRO."},
     "expectations": {"expected_response": "Method Resolution Order (MRO) determines the order in which base classes are searched when looking for a method. Python uses the C3 linearization algorithm to compute MRO, ensuring a consistent and predictable method lookup order in multiple inheritance."}},
])


# -- Part 1: Deterministic scorer ------------------------------------------ #
@scorer
def formatting_quality(outputs, expectations) -> Feedback:
    """Check response quality: sentence count, keyword overlap, length."""
    text = str(outputs)
    expected = str(expectations.get("expected_response", ""))

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_score = min(len(sentences) / 3.0, 1.0)

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                  "of", "to", "in", "on", "by", "it", "that", "this", "for",
                  "with", "as", "at", "from", "they", "be", "its"}
    expected_kw = set(expected.lower().split()) - stop_words
    response_kw = set(text.lower().split()) - stop_words
    overlap = len(expected_kw & response_kw) / max(len(expected_kw), 1)

    word_count = len(text.split())
    length_score = 0.2 if word_count < 10 else (0.6 if word_count < 20 else (1.0 if word_count <= 150 else 0.7))

    composite = round(0.3 * sentence_score + 0.4 * overlap + 0.3 * length_score, 3)
    return Feedback(
        value=composite,
        rationale=f"sentences={len(sentences)}, keyword_overlap={overlap:.2f}, word_count={word_count}",
        source=AssessmentSource(source_type="CODE", source_id="formatting_quality"),
    )


# -- Part 2: LLM-based scorer --------------------------------------------- #
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
    """Use gemma4:e2b to judge technical accuracy and completeness."""
    llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
    prompt = JUDGE_PROMPT.format(
        question=inputs.get("question", ""),
        expected=expectations.get("expected_response", ""),
        response=str(outputs),
    )
    raw = llm.invoke(prompt).content.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        scores = json.loads(match.group()) if match else {
            "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5,
            "rationale": "Could not parse judge output"}

    avg = round((float(scores.get("accuracy", 0.5))
                 + float(scores.get("completeness", 0.5))
                 + float(scores.get("clarity", 0.5))) / 3.0, 3)
    return Feedback(
        value=avg,
        rationale=(f"accuracy={scores.get('accuracy')}, completeness={scores.get('completeness')}, "
                   f"clarity={scores.get('clarity')} | {scores.get('rationale', '')}"),
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gemma4:e2b"),
    )


# -- Predict function ----------------------------------------------------- #
def answer_question(question: str) -> str:
    """Generate an answer using the local LLM."""
    llm = ChatOllama(model="gemma4:e2b", temperature=0.7)
    return llm.invoke(f"Answer concisely in 2-3 sentences: {question}").content


# -- Main ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 60)
    print("Part 1: Deterministic Scorer (formatting_quality)")
    print("  Checks sentence structure, keyword overlap, length.")
    print("Part 2: LLM-Based Scorer (llm_technical_quality)")
    print("  Uses gemma4:e2b to judge accuracy/completeness/clarity.")
    print("=" * 60)

    # Part 3: Combined evaluation
    print(f"\nPart 3: Running evaluation on {len(EVAL_DATA)} questions...")
    print("  Scorers: formatting_quality + llm_technical_quality\n")

    results = mlflow.genai.evaluate(
        data=EVAL_DATA,
        predict_fn=answer_question,
        scorers=[formatting_quality, llm_technical_quality],
    )

    # Aggregate metrics
    print("\n--- Aggregate Metrics ---")
    for name, value in results.metrics.items():
        print(f"  {name}: {value:.3f}")

    # Per-row results
    print("\n--- Per-Row Results ---")
    df = results.result_df
    if df is not None:
        questions = [r["inputs"]["question"] for _, r in EVAL_DATA.iterrows()]
        for i, row in df.iterrows():
            q = questions[i] if i < len(questions) else "N/A"
            print(f"\n  Q{i + 1}: {q}")
            for col in sorted(df.columns):
                if col.endswith("/value") and "expected" not in col:
                    print(f"     {col}: {row[col]}")
                elif col.endswith("/rationale") and row[col] and "expected" not in col:
                    rationale = str(row[col])[:90]
                    print(f"     {col}: {rationale}")

    # Part 4: Threshold checking
    print("\n" + "=" * 60)
    print("Part 4: Programmatic Threshold Checking")
    print("=" * 60)

    thresholds = {
        "formatting_quality/mean": 0.4,
        "llm_technical_quality/mean": 0.5,
    }
    all_passed = True
    for metric_name, min_value in thresholds.items():
        actual = results.metrics.get(metric_name)
        if actual is None:
            print(f"  WARNING: metric '{metric_name}' not found")
            all_passed = False
            continue
        passed = actual >= min_value
        print(f"  [{'PASS' if passed else 'FAIL'}] {metric_name}: {actual:.3f} (threshold: {min_value})")
        if not passed:
            all_passed = False

    print(f"\n  {'All quality thresholds met.' if all_passed else 'Some thresholds not met.'}")
    print("\n" + "=" * 60)
    print("Done! View results in MLflow UI: http://127.0.0.1:5000")
    print("Experiment: L2/M3_deep_evaluation/1_custom_metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
