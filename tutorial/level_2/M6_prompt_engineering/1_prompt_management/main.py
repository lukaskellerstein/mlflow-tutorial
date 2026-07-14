"""L2-M6.1 — Prompt Management at Scale: versioning, A/B testing, comparison."""

import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── MLflow setup ──────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L2/M6_prompt_engineering/1_prompt_management")

# ── Constants ─────────────────────────────────────────────────
PROMPT_NAME = "L2_M6_qa_prompt"

VARIANTS: list[dict] = [
    {
        "label": "concise",
        "template": [
            {"role": "system", "content": "You are a concise assistant. Answer the question in 1-2 sentences. Be direct and factual."},
            {"role": "user", "content": "{{question}}"},
        ],
        "commit_message": "Concise variant — short, factual answers",
        "tags": {"style": "concise", "target_length": "short"},
    },
    {
        "label": "detailed",
        "template": [
            {"role": "system", "content": "You are a thorough assistant. Answer the question with a detailed explanation. Include relevant context, examples, and reasoning. Aim for 3-5 sentences."},
            {"role": "user", "content": "{{question}}"},
        ],
        "commit_message": "Detailed variant — thorough explanations with examples",
        "tags": {"style": "detailed", "target_length": "long"},
    },
    {
        "label": "creative",
        "template": [
            {"role": "system", "content": "You are a creative and engaging assistant. Answer the question using vivid language, analogies, or metaphors. Make the answer memorable and fun while staying accurate."},
            {"role": "user", "content": "{{question}}"},
        ],
        "commit_message": "Creative variant — vivid language and analogies",
        "tags": {"style": "creative", "target_length": "medium"},
    },
]

TEST_QUESTIONS = [
    "What is a hash table?",
    "Why do leaves change color in autumn?",
    "How does a blockchain work?",
]


def register_prompt_variants() -> list[int]:
    """Register three prompt versions and return their version numbers."""
    print("=" * 60)
    print("Part 1: Register multiple prompt versions")
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
        print(f"  v{pv.version} [{variant['label']}] — registered")
        print(f"    System: {variant['template'][0]['content'][:70]}...")

    print(f"\n  Total versions registered: {len(versions)}")
    return versions


def ab_test_prompts(versions: list[int]) -> pd.DataFrame:
    """Run A/B test: each prompt version answers the same questions."""
    print()
    print("=" * 60)
    print("Part 2: A/B test prompts with ChatOpenAI")
    print("=" * 60)

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )
    rows: list[dict] = []

    for version, variant in zip(versions, VARIANTS):
        label = variant["label"]
        prompt_version = mlflow.genai.load_prompt(PROMPT_NAME, version=version)

        # Build LangChain prompt from the chat template
        lc_messages = prompt_version.to_single_brace_format()
        lc_prompt = ChatPromptTemplate.from_messages(
            [(msg["role"], msg["content"]) for msg in lc_messages]
        )
        chain = lc_prompt | llm

        run_name = f"ab_test_{label}_v{version}"
        print(f"\n  --- Variant: {label} (v{version}) ---")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("variant", label)
            mlflow.log_param("prompt_version", version)
            mlflow.log_param("prompt_name", PROMPT_NAME)
            mlflow.log_param("model", "google/gemma-4-26b-a4b")

            total_length = 0
            total_words = 0

            for i, question in enumerate(TEST_QUESTIONS, 1):
                response = chain.invoke({"question": question})
                answer = response.content

                resp_len = len(answer)
                word_count = len(answer.split())
                total_length += resp_len
                total_words += word_count

                mlflow.log_metric(f"q{i}_char_length", resp_len)
                mlflow.log_metric(f"q{i}_word_count", word_count)

                preview = answer[:80].replace("\n", " ")
                print(f"    Q{i}: {question}")
                print(f"       A: {preview}...")
                print(f"       [{resp_len} chars, {word_count} words]")

                rows.append({
                    "variant": label,
                    "version": version,
                    "question": question,
                    "answer": answer,
                    "char_length": resp_len,
                    "word_count": word_count,
                })

            avg_length = total_length / len(TEST_QUESTIONS)
            avg_words = total_words / len(TEST_QUESTIONS)
            mlflow.log_metric("avg_char_length", avg_length)
            mlflow.log_metric("avg_word_count", avg_words)
            print(f"    Averages: {avg_length:.0f} chars, {avg_words:.0f} words")

    return pd.DataFrame(rows)


def compare_results(results_df: pd.DataFrame) -> None:
    """Build comparison table, find best variant, log as artifact."""
    print()
    print("=" * 60)
    print("Part 3: Compare results across variants")
    print("=" * 60)

    # Summary table per variant
    summary = (
        results_df.groupby("variant")
        .agg(
            avg_chars=("char_length", "mean"),
            avg_words=("word_count", "mean"),
            min_chars=("char_length", "min"),
            max_chars=("char_length", "max"),
            total_responses=("answer", "count"),
        )
        .round(1)
    )

    print("\n  Per-variant summary:")
    print(summary.to_string(index=True))

    # Per-question breakdown
    pivot = results_df.pivot_table(
        index="question",
        columns="variant",
        values="word_count",
        aggfunc="first",
    )
    print("\n  Word count per question:")
    print(pivot.to_string())

    # Determine best variant (closest to medium length: 40-80 words)
    target_words = 60
    summary["distance_from_target"] = abs(summary["avg_words"] - target_words)
    best_variant = summary["distance_from_target"].idxmin()
    print(f"\n  Best balanced variant (closest to ~{target_words} words): {best_variant}")

    # Log comparison as artifact in a parent run
    with mlflow.start_run(run_name="ab_test_comparison"):
        mlflow.log_param("best_variant", best_variant)
        mlflow.log_param("num_variants", len(VARIANTS))
        mlflow.log_param("num_questions", len(TEST_QUESTIONS))

        # Save full results and summary as CSV artifacts
        results_df.to_csv("/tmp/ab_test_full_results.csv", index=False)
        summary.to_csv("/tmp/ab_test_summary.csv")

        mlflow.log_artifact("/tmp/ab_test_full_results.csv")
        mlflow.log_artifact("/tmp/ab_test_summary.csv")

        # Log summary metrics
        for variant_name in summary.index:
            mlflow.log_metric(
                f"{variant_name}_avg_words",
                summary.loc[variant_name, "avg_words"],
            )

        print("\n  Comparison artifacts logged to MLflow.")
        print(f"  Run: ab_test_comparison")


def main() -> None:
    versions = register_prompt_variants()
    results_df = ab_test_prompts(versions)
    compare_results(results_df)

    print()
    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("  Experiment: L2/M6_prompt_engineering/1_prompt_management")
    print("  Look for: 3 A/B test runs + 1 comparison run")
    print("=" * 60)


if __name__ == "__main__":
    main()
