"""
L1-M4.1 — LLM Evaluation Basics

Demonstrates how to evaluate LLM outputs using mlflow.genai.evaluate().
Uses built-in deterministic scorers and a custom scorer to assess
a simple Q&A function powered by a local LMStudio model.
"""

import mlflow
import pandas as pd
from openai import OpenAI
from mlflow.genai.scorers import ResponseLength, scorer

# -- Configuration --
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M4_evaluations/1_llm_eval_basics")

# -- LLM client --
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def answer_question(question: str) -> str:
    """Ask the LLM a factual question and return its answer."""
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# -- Custom scorer --
@scorer
def contains_expected(inputs, outputs, expectations) -> bool:
    """Check whether the LLM output contains the expected answer text."""
    expected = expectations.get("expected_response", "")
    return expected.lower() in outputs.lower()


def main() -> None:
    # Step 1 — Build the evaluation dataset
    print("=" * 60)
    print("Step 1: Creating evaluation dataset")
    print("=" * 60)
    eval_data = pd.DataFrame([
        {"inputs": {"question": "What is the capital of France?"},
         "expectations": {"expected_response": "Paris"}},
        {"inputs": {"question": "What is the chemical symbol for water?"},
         "expectations": {"expected_response": "H2O"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"},
         "expectations": {"expected_response": "William Shakespeare"}},
        {"inputs": {"question": "How many continents are there?"},
         "expectations": {"expected_response": "7"}},
        {"inputs": {"question": "What planet is closest to the Sun?"},
         "expectations": {"expected_response": "Mercury"}},
    ])
    for i, row in eval_data.iterrows():
        print(f"  Q{i + 1}: {row['inputs']['question']}")

    # Step 2 — Configure scorers
    print("\n" + "=" * 60)
    print("Step 2: Scorers")
    print("=" * 60)
    print("  - ResponseLength(1-500 words): deterministic length check")
    print("  - contains_expected: custom scorer via @scorer decorator")

    # Step 3 — Run evaluation
    print("\n" + "=" * 60)
    print("Step 3: Running mlflow.genai.evaluate()")
    print("=" * 60)
    print("  Calling LMStudio for each question...\n")
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=answer_question,
        scorers=[
            ResponseLength(min_length=1, max_length=500, unit="words"),
            contains_expected,
        ],
    )

    # Step 4 — Display results
    print("\n" + "=" * 60)
    print("Step 4: Evaluation Results")
    print("=" * 60)
    print("\n--- Aggregate Metrics ---")
    for name, value in results.metrics.items():
        print(f"  {name}: {value}")

    print("\n--- Per-Row Results ---")
    table = results.result_df
    if table is not None:
        for i, row in table.iterrows():
            q = row.get("request", {}).get("question", "N/A")
            a = str(row.get("response", ""))
            if len(a) > 80:
                a = a[:77] + "..."
            print(f"\n  Q{i + 1}: {q}")
            print(f"     Answer: {a}")
            for c in table.columns:
                if c.endswith("/value") or c.endswith("/rationale"):
                    print(f"     {c}: {row[c]}")

    print("\n" + "=" * 60)
    print("Done! See results in MLflow UI at http://127.0.0.1:5000")
    print("Experiment: 'L1/M4_evaluations/1_llm_eval_basics'")
    print("=" * 60)


if __name__ == "__main__":
    main()
