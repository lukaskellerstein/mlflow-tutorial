"""
L1-M5.3 — Prompt Optimization

Demonstrates systematic prompt optimization tracked with MLflow:
- Define an evaluation dataset and custom scoring function
- Iterate through prompt variations (instruction style, constraints, few-shot)
- Log every attempt as an MLflow run for reproducible comparison
- Identify the best-performing prompt and analyze the optimization trajectory
"""

import json
import time

import mlflow
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Evaluation dataset — Q&A pairs for a geography knowledge task
# ---------------------------------------------------------------------------
EVAL_DATA = [
    {
        "question": "What is the capital of France?",
        "expected": "Paris",
    },
    {
        "question": "What is the largest ocean on Earth?",
        "expected": "Pacific Ocean",
    },
    {
        "question": "What is the longest river in Africa?",
        "expected": "Nile",
    },
    {
        "question": "What is the smallest continent by land area?",
        "expected": "Australia",
    },
    {
        "question": "What is the highest mountain in the world?",
        "expected": "Mount Everest",
    },
]

# ---------------------------------------------------------------------------
# Few-shot examples (not part of the eval set)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the largest desert in the world?", "answer": "Sahara Desert"},
    {"question": "What is the deepest lake in the world?", "answer": "Lake Baikal"},
]


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------
def score_answer(predicted: str, expected: str) -> dict[str, float]:
    """Score a predicted answer against the expected answer.

    Returns three sub-scores that together give a rounded picture:
      - exact_match:  1.0 if the expected string appears in the prediction
      - brevity:      penalizes overly long answers (ideal <= 5 words)
      - confidence:   1.0 if the answer is a direct statement (no hedging)
    """
    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact match — does the expected answer appear in the prediction?
    exact_match = 1.0 if exp_lower in pred_lower else 0.0

    # Brevity — short, direct answers are better for factual questions
    word_count = len(predicted.split())
    if word_count <= 5:
        brevity = 1.0
    elif word_count <= 15:
        brevity = 0.7
    elif word_count <= 30:
        brevity = 0.4
    else:
        brevity = 0.2

    # Confidence — penalize hedging language
    hedging = ["i think", "probably", "maybe", "i'm not sure", "it might be"]
    confidence = 0.0 if any(h in pred_lower for h in hedging) else 1.0

    composite = (exact_match * 0.5) + (brevity * 0.3) + (confidence * 0.2)
    return {
        "exact_match": exact_match,
        "brevity": brevity,
        "confidence": confidence,
        "composite": composite,
    }


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-26b-a4b"

llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def ask_llm(prompt: str, question: str) -> str:
    """Send a question to the LLM with the given system prompt."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Evaluate a prompt variant across the full dataset
# ---------------------------------------------------------------------------
def evaluate_prompt(prompt_text: str, dataset: list[dict]) -> list[dict]:
    """Run *prompt_text* against every item in *dataset*, return per-row results."""
    results = []
    for item in dataset:
        predicted = ask_llm(prompt_text, item["question"])
        scores = score_answer(predicted, item["expected"])
        results.append({
            "question": item["question"],
            "expected": item["expected"],
            "predicted": predicted,
            **scores,
        })
    return results


# ---------------------------------------------------------------------------
# Run one optimization iteration as an MLflow run
# ---------------------------------------------------------------------------
def run_iteration(
    variant_name: str,
    prompt_text: str,
    dataset: list[dict],
    iteration: int,
    parent_run_id: str,
) -> dict:
    """Evaluate a prompt variant and log everything to MLflow."""
    with mlflow.start_run(run_name=variant_name, nested=True) as run:
        mlflow.log_params({
            "variant": variant_name,
            "iteration": iteration,
            "prompt_length": len(prompt_text),
            "prompt_word_count": len(prompt_text.split()),
            "model": MODEL_NAME,
        })
        mlflow.set_tag("prompt_text", prompt_text[:250])

        start = time.time()
        results = evaluate_prompt(prompt_text, dataset)
        elapsed = time.time() - start

        # Aggregate metrics
        df = pd.DataFrame(results)
        avg_metrics = {
            "avg_exact_match": df["exact_match"].mean(),
            "avg_brevity": df["brevity"].mean(),
            "avg_confidence": df["confidence"].mean(),
            "avg_composite": df["composite"].mean(),
            "eval_time_s": round(elapsed, 2),
        }
        mlflow.log_metrics(avg_metrics)

        # Log per-question results as a table artifact
        mlflow.log_table(df, artifact_file="eval_results.json")

        # Log the full prompt as an artifact
        prompt_path = "/tmp/prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        mlflow.log_artifact(prompt_path, artifact_path="prompts")

        print(f"  [{iteration}] {variant_name:30s}  "
              f"match={avg_metrics['avg_exact_match']:.2f}  "
              f"brevity={avg_metrics['avg_brevity']:.2f}  "
              f"composite={avg_metrics['avg_composite']:.2f}  "
              f"({elapsed:.1f}s)")

        return {
            "variant": variant_name,
            "iteration": iteration,
            "run_id": run.info.run_id,
            **avg_metrics,
        }


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
def build_few_shot_block(n: int) -> str:
    """Build a few-shot examples block with *n* examples."""
    if n == 0:
        return ""
    lines = ["\nExamples:"]
    for ex in FEW_SHOT_EXAMPLES[:n]:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"A: {ex['answer']}")
        lines.append("")
    return "\n".join(lines)


PROMPT_VARIANTS = [
    (
        "baseline",
        "Answer the following geography question.",
    ),
    (
        "concise_instruction",
        "Answer the following geography question in as few words as possible. "
        "Give only the answer, no explanation.",
    ),
    (
        "role_assignment",
        "You are a geography expert. Answer the following question with a short, "
        "definitive answer. Do not hedge or qualify your response.",
    ),
    (
        "structured_constraints",
        "You are a geography expert. Rules:\n"
        "1. Answer in 1-3 words only.\n"
        "2. Do not include any explanation.\n"
        "3. Do not start with 'The answer is'.\n"
        "4. Be direct and confident.",
    ),
]


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    # ---- Part 1: Define the problem ----------------------------------------
    print("=" * 60)
    print("Part 1: Optimization Problem")
    print("=" * 60)
    print(f"  Dataset:  {len(EVAL_DATA)} Q&A pairs (geography)")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Scoring:  exact_match (50%) + brevity (30%) + confidence (20%)")
    print()

    all_results: list[dict] = []

    with mlflow.start_run(run_name="prompt_optimization") as parent_run:
        mlflow.set_tags({
            "task": "prompt_optimization",
            "dataset_size": str(len(EVAL_DATA)),
            "model": MODEL_NAME,
        })

        # ---- Part 2: Manual optimization loop ------------------------------
        print("=" * 60)
        print("Part 2: Manual Prompt Optimization (instruction variants)")
        print("=" * 60)

        for i, (name, prompt) in enumerate(PROMPT_VARIANTS):
            result = run_iteration(
                variant_name=name,
                prompt_text=prompt,
                dataset=EVAL_DATA,
                iteration=i,
                parent_run_id=parent_run.info.run_id,
            )
            all_results.append(result)

        print()

        # ---- Part 3: Few-shot example optimization -------------------------
        print("=" * 60)
        print("Part 3: Few-Shot Example Optimization")
        print("=" * 60)

        # Use the best instruction variant so far as the base
        best_so_far = max(all_results, key=lambda r: r["avg_composite"])
        base_prompt = dict(PROMPT_VARIANTS)[best_so_far["variant"]]
        print(f"  Base prompt: '{best_so_far['variant']}' "
              f"(composite={best_so_far['avg_composite']:.2f})")
        print()

        for n_examples in range(4):  # 0, 1, 2, 3
            fs_block = build_few_shot_block(n_examples)
            prompt_text = base_prompt + fs_block
            name = f"few_shot_{n_examples}_examples"
            iteration = len(all_results)

            result = run_iteration(
                variant_name=name,
                prompt_text=prompt_text,
                dataset=EVAL_DATA,
                iteration=iteration,
                parent_run_id=parent_run.info.run_id,
            )
            all_results.append(result)

        print()

        # ---- Part 4: Systematic comparison ---------------------------------
        print("=" * 60)
        print("Part 4: Systematic Comparison")
        print("=" * 60)

        summary_df = pd.DataFrame(all_results)
        display_cols = ["iteration", "variant", "avg_exact_match",
                        "avg_brevity", "avg_confidence", "avg_composite"]
        print()
        print(summary_df[display_cols].to_string(index=False))
        print()

        # Best overall
        best = summary_df.loc[summary_df["avg_composite"].idxmax()]
        print(f"  Best variant: {best['variant']}")
        print(f"  Best composite score: {best['avg_composite']:.2f}")
        print()

        # Log optimization trajectory to parent run
        for r in all_results:
            mlflow.log_metric(
                "optimization_trajectory",
                r["avg_composite"],
                step=r["iteration"],
            )

        mlflow.log_params({
            "best_variant": best["variant"],
            "best_composite": round(float(best["avg_composite"]), 4),
            "total_iterations": len(all_results),
        })

        # Log the full summary table
        mlflow.log_table(summary_df, artifact_file="optimization_summary.json")

    print("=" * 60)
    print("Done! View the optimization runs in the MLflow UI:")
    print("  http://127.0.0.1:5000")
    print("  Experiment: L1/M5_prompt_engineering/3_prompt_optimization")
    print("  Expand 'prompt_optimization' to see all variants as nested runs.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M5_prompt_engineering/3_prompt_optimization")
    main()
