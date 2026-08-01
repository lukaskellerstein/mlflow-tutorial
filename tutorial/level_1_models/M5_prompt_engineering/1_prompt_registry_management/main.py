"""L1-M5.1 — Prompt Registry and Management

Combines prompt registry fundamentals with management at scale:
- Register versioned prompts with mlflow.genai.register_prompt()
- Set aliases (production, staging) for version-independent loading
- Load and format prompts by version or alias
- A/B test multiple prompt variants with the same questions
- Compare results and select the best variant
"""

from typing import cast

import mlflow
import pandas as pd
from openai import OpenAI

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L1/M5_prompt_engineering/1_prompt_registry_management")

PROMPT_NAME = "L1_M5_qa_prompt"

VARIANTS: list[dict] = [
    {
        "label": "concise",
        "template": "Answer the question in 1-2 sentences. Be direct and factual.\n\nQuestion: {{question}}",
        "commit_message": "Concise variant -- short, factual answers",
        "tags": {"style": "concise", "target_length": "short"},
    },
    {
        "label": "detailed",
        "template": (
            "You are a thorough assistant. Answer the question with a detailed "
            "explanation. Include relevant context and examples. Aim for 3-5 sentences.\n\n"
            "Question: {{question}}"
        ),
        "commit_message": "Detailed variant -- thorough explanations with examples",
        "tags": {"style": "detailed", "target_length": "long"},
    },
    {
        "label": "creative",
        "template": (
            "You are a creative and engaging assistant. Answer the question using "
            "vivid language, analogies, or metaphors. Make it memorable and fun "
            "while staying accurate.\n\nQuestion: {{question}}"
        ),
        "commit_message": "Creative variant -- vivid language and analogies",
        "tags": {"style": "creative", "target_length": "medium"},
    },
]

TEST_QUESTIONS = [
    "What is a hash table?",
    "Why do leaves change color in autumn?",
    "How does a blockchain work?",
]


# ── Part 1: Register Prompt Versions ─────────────────────────────────────


def part1_register_prompts() -> list[int]:
    """Register three prompt versions and set aliases."""
    print("=" * 60)
    print("Part 1: Register Prompt Versions and Set Aliases")
    print("=" * 60)

    versions: list[int] = []
    for variant in VARIANTS:
        pv = mlflow.genai.register_prompt(
            name=PROMPT_NAME,
            template=variant["template"],
            commit_message=variant["commit_message"],
            tags=variant["tags"],
        )
        versions.append(pv.version)
        preview = variant["template"][:70].replace("\n", " ")
        print(f"  v{pv.version} [{variant['label']}] -- {preview}...")

    # Set alias on v2 (detailed) as production
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=versions[1])
    print(f"\n  Alias 'production' -> v{versions[1]} (detailed)")

    # Demonstrate loading by version and alias
    print()
    print("=" * 60)
    print("Part 1b: Load Prompts by Version and Alias")
    print("=" * 60)

    loaded_v1 = mlflow.genai.load_prompt(PROMPT_NAME, version=versions[0])
    print(f"  Loaded v{loaded_v1.version}: {loaded_v1.template[:60]}...")

    loaded_prod = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
    print(f"  Loaded @production (v{loaded_prod.version}): {loaded_prod.template[:60]}...")

    # Format a prompt with variables
    formatted = str(loaded_prod.format(question="What is recursion?"))
    print(f"  Formatted: {formatted[:80]}...")

    # Search registered prompts
    prompts = mlflow.genai.search_prompts()
    print(f"\n  Total registered prompts: {len(prompts)}")

    return versions


# ── Part 2: A/B Test Prompt Variants ──────────────────────────────────────


def part2_ab_test(versions: list[int]) -> pd.DataFrame:
    """Run A/B test: each prompt version answers the same questions."""
    print()
    print("=" * 60)
    print("Part 2: A/B Test Prompt Variants")
    print("=" * 60)

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    rows: list[dict] = []

    for version, variant in zip(versions, VARIANTS):
        label = variant["label"]
        prompt_version = mlflow.genai.load_prompt(PROMPT_NAME, version=version)

        run_name = f"ab_test_{label}_v{version}"
        print(f"\n  --- Variant: {label} (v{version}) ---")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("variant", label)
            mlflow.log_param("prompt_version", version)
            mlflow.log_param("prompt_name", PROMPT_NAME)
            mlflow.log_param("model", "google/gemma-4-e4b")

            total_words = 0

            for i, question in enumerate(TEST_QUESTIONS, 1):
                formatted = str(prompt_version.format(question=question))
                response = client.chat.completions.create(
                    model="google/gemma-4-e4b",
                    messages=[{"role": "user", "content": formatted}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                answer = (response.choices[0].message.content or "").strip()

                word_count = len(answer.split())
                total_words += word_count
                mlflow.log_metric(f"q{i}_word_count", word_count)

                preview = answer[:80].replace("\n", " ")
                print(f"    Q{i}: {question}")
                print(f"       A: {preview}...")
                print(f"       [{word_count} words]")

                rows.append(
                    {
                        "variant": label,
                        "version": version,
                        "question": question,
                        "answer": answer,
                        "word_count": word_count,
                    }
                )

            avg_words = total_words / len(TEST_QUESTIONS)
            mlflow.log_metric("avg_word_count", avg_words)
            print(f"    Average: {avg_words:.0f} words")

    return pd.DataFrame(rows)


# ── Part 3: Compare Results and Select Best ───────────────────────────────


def part3_compare_results(results_df: pd.DataFrame) -> None:
    """Build comparison table, find best variant, log as artifact."""
    print()
    print("=" * 60)
    print("Part 3: Compare Results Across Variants")
    print("=" * 60)

    summary = (
        results_df.groupby("variant")
        .agg(
            avg_words=("word_count", "mean"),
            min_words=("word_count", "min"),
            max_words=("word_count", "max"),
            total_responses=("answer", "count"),
        )
        .round(1)
    )

    print("\n  Per-variant summary:")
    print(summary.to_string(index=True))

    # Determine best variant (closest to medium length: 40-80 words)
    target_words = 60
    summary["distance_from_target"] = abs(summary["avg_words"] - target_words)
    best_variant = cast(pd.Series, summary["distance_from_target"]).idxmin()
    print(f"\n  Best balanced variant (closest to ~{target_words} words): {best_variant}")

    with mlflow.start_run(run_name="ab_test_comparison"):
        mlflow.log_param("best_variant", best_variant)
        mlflow.log_param("num_variants", len(VARIANTS))
        mlflow.log_param("num_questions", len(TEST_QUESTIONS))

        mlflow.log_table(results_df, artifact_file="ab_test_full_results.json")
        mlflow.log_table(summary.reset_index(), artifact_file="ab_test_summary.json")

        for variant_name in summary.index:
            mlflow.log_metric(
                f"{variant_name}_avg_words",
                summary.loc[variant_name, "avg_words"],
            )
        print("  Comparison artifacts logged to MLflow.")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    versions = part1_register_prompts()
    results_df = part2_ab_test(versions)
    part3_compare_results(results_df)

    # Cleanup alias
    mlflow.genai.delete_prompt_alias(PROMPT_NAME, alias="production")

    print()
    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5555")
    print("  Experiment: L1/M5_prompt_engineering/1_prompt_registry_management")
    print("  Prompt Registry: look for the registered prompt versions")
    print("  Runs: 3 A/B test runs + 1 comparison run")
    print("=" * 60)


if __name__ == "__main__":
    main()
