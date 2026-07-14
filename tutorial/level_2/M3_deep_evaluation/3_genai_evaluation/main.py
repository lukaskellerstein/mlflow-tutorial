"""
L2-3.3 — GenAI Evaluation Framework

Demonstrates the full mlflow.genai evaluation framework: built-in scorers,
custom scorers via the @scorer decorator, and batch evaluation across
multiple LLM configurations to find the best setup.
"""

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mlflow.entities import Feedback
from mlflow.genai.scorers import ResponseLength, scorer

# -- Part 1: Evaluation dataset ------------------------------------------------

EVAL_DATA = [
    {"inputs": {"question": "What is Python's GIL?"},
     "expectations": {"expected_response":
         "The GIL is a mutex in CPython that allows only one thread to "
         "execute Python bytecodes at a time, limiting true parallelism."}},
    {"inputs": {"question": "Explain list comprehensions in Python."},
     "expectations": {"expected_response":
         "List comprehensions are a concise syntax [expr for item in iter "
         "if cond] for creating lists, generally faster than for-loops."}},
    {"inputs": {"question": "What is a Python decorator?"},
     "expectations": {"expected_response":
         "A decorator is a function that wraps another function to extend "
         "its behavior, applied with the @decorator syntax."}},
    {"inputs": {"question": "How does garbage collection work in Python?"},
     "expectations": {"expected_response":
         "Python uses reference counting; when the count drops to zero the "
         "object is freed. A cyclic GC handles reference cycles."}},
    {"inputs": {"question": "What are Python generators?"},
     "expectations": {"expected_response":
         "Generators yield values lazily one at a time via the yield "
         "keyword, making them memory-efficient for large sequences."}},
    {"inputs": {"question": "Describe Python's dataclass decorator."},
     "expectations": {"expected_response":
         "The @dataclass decorator auto-generates __init__, __repr__, and "
         "__eq__ from annotated fields, reducing boilerplate."}},
]

# -- Part 2: Built-in scorers -------------------------------------------------

BUILTIN_SCORERS = [
    ResponseLength(min_length=20, max_length=500, unit="words"),
]

# -- Part 3: Custom scorers via @scorer decorator ------------------------------


@scorer(name="keyword_coverage")
def keyword_coverage(inputs: dict, outputs: str) -> Feedback:
    """Check whether the answer references key terms from the question."""
    question = inputs.get("question", "").lower()
    answer = str(outputs).lower()
    stopwords = {"what", "does", "how", "when", "this", "that", "with", "from",
                 "have", "they", "their", "about", "which", "will", "been",
                 "explain", "describe", "python"}
    keywords = [w for w in question.split()
                if len(w) > 3 and w.strip("'\"?,.") not in stopwords]
    if not keywords:
        return Feedback(value=1.0, rationale="No keywords to check.")
    hits = sum(1 for kw in keywords if kw.strip("'\"?,.") in answer)
    score = hits / len(keywords)
    return Feedback(value=round(score, 2),
                    rationale=f"{hits}/{len(keywords)} keywords found in answer.")


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
    indicators = ["for example", "e.g.", "such as", "```", ">>>",
                  "= [", "= {", "def ", "class ", "import "]
    return any(ind in text for ind in indicators)


ALL_SCORERS = BUILTIN_SCORERS + [keyword_coverage, answer_conciseness, has_example]

# -- Part 4: Batch evaluation across configurations ----------------------------

CONFIGS = [
    {"name": "temp_0.3", "model": "google/gemma-4-26b-a4b", "temperature": 0.3},
    {"name": "temp_0.9", "model": "google/gemma-4-26b-a4b", "temperature": 0.9},
]


def build_predict_fn(model: str, temperature: float):
    """Return a predict function that answers questions with the given config."""
    llm = ChatOpenAI(
        model=model,
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=temperature,
    )
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable Python tutor. Answer clearly "
                       "and concisely. Include a short example when appropriate."),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )

    def predict_fn(question: str) -> str:
        return chain.invoke({"question": question})

    return predict_fn


def evaluate_config(config: dict) -> dict:
    """Evaluate a single LLM configuration and return its metrics."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {config['name']}  "
          f"(model={config['model']}, temp={config['temperature']})")
    print("=" * 60)

    predict_fn = build_predict_fn(config["model"], config["temperature"])

    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params({
            "model": config["model"],
            "temperature": config["temperature"],
            "num_questions": len(EVAL_DATA),
        })
        result = mlflow.genai.evaluate(
            data=EVAL_DATA, predict_fn=predict_fn, scorers=ALL_SCORERS,
        )
        for name, value in result.metrics.items():
            mlflow.log_metric(name, value)

    print(f"\n  Metrics for '{config['name']}':")
    for name, value in sorted(result.metrics.items()):
        print(f"    {name}: {value:.4f}")
    return result.metrics


# -- Part 5: Results analysis --------------------------------------------------


def compare_results(all_metrics: dict[str, dict]) -> None:
    """Print a side-by-side comparison and identify best config per metric."""
    print(f"\n{'=' * 60}")
    print("Results Comparison")
    print("=" * 60)

    names = list(all_metrics.keys())
    metrics = sorted(set().union(*(m.keys() for m in all_metrics.values())))

    # Header + rows
    hdr = f"  {'Metric':<35}" + "".join(f" {n:>12}" for n in names) + f" {'Best':>12}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    wins: dict[str, int] = {n: 0 for n in names}
    for m in metrics:
        vals = {n: all_metrics[n].get(m, float("nan")) for n in names}
        valid = {k: v for k, v in vals.items() if v == v}
        best = max(valid, key=lambda k: valid[k]) if valid else "n/a"
        if best in wins:
            wins[best] += 1
        row = f"  {m:<35}" + "".join(f" {vals[n]:>12.4f}" for n in names)
        print(f"{row} {best:>12}")

    print(f"\n  Summary:")
    for n in names:
        print(f"    {n}: won {wins[n]}/{len(metrics)} metrics")
    overall = max(wins, key=lambda k: wins[k])
    print(f"\n  Overall best configuration: {overall}")


# -- Main ----------------------------------------------------------------------


def main() -> None:
    """Run the full GenAI evaluation workflow."""
    mlflow.langchain.autolog()

    print("=" * 60)
    print("L2-3.3 — GenAI Evaluation Framework")
    print("=" * 60)

    # Show dataset
    print(f"\nEvaluation dataset: {len(EVAL_DATA)} questions")
    for i, item in enumerate(EVAL_DATA, 1):
        print(f"  Q{i}: {item['inputs']['question']}")

    # Show scorers
    print(f"\nScorers ({len(ALL_SCORERS)}):")
    for s in ALL_SCORERS:
        print(f"  - {s.name}")

    # Evaluate each configuration
    all_metrics: dict[str, dict] = {}
    for config in CONFIGS:
        all_metrics[config["name"]] = evaluate_config(config)

    # Compare results
    compare_results(all_metrics)

    print(f"\n{'=' * 60}")
    print("Done! Open MLflow UI to explore results:")
    print("  http://127.0.0.1:5000")
    print("  Experiment: L2/M3_deep_evaluation/3_genai_evaluation")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M3_deep_evaluation/3_genai_evaluation")
    main()
