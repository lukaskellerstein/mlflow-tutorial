"""
L1-M2.2 — Manual Tracing

Demonstrates MLflow's manual tracing APIs:
- Part 1: @mlflow.trace decorator for automatic span creation
- Part 2: mlflow.start_span() context manager for fine-grained control
- Part 3: Combining manual tracing with OpenAI autolog
"""

import mlflow
from openai import OpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
MODEL_NAME = "google/gemma-4-e4b"


# --------------------------------------------------------------------- #
# Part 1: @mlflow.trace decorator
# --------------------------------------------------------------------- #


@mlflow.trace(name="validate_text")
def validate_text(text: str) -> str:
    """Validate and clean input text."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    return text.strip()


@mlflow.trace(name="transform_text")
def transform_text(text: str) -> dict:
    """Compute basic text statistics."""
    words = text.split()
    return {
        "original": text,
        "word_count": len(words),
        "char_count": len(text),
        "uppercase": text.upper(),
    }


@mlflow.trace(name="process_pipeline")
def process_pipeline(text: str) -> dict:
    """Run a two-step pipeline: validate then transform."""
    validated = validate_text(text)
    return transform_text(validated)


# --------------------------------------------------------------------- #
# Part 2: mlflow.start_span() context manager
# --------------------------------------------------------------------- #


def analyze_texts(texts: list[str]) -> dict:
    """Analyze a batch of texts with explicit span control."""
    with mlflow.start_span(name="batch_analysis") as root_span:
        root_span.set_inputs({"texts": texts, "count": len(texts)})
        all_results = []
        total_words = 0

        for i, text in enumerate(texts):
            with mlflow.start_span(name=f"analyze_item_{i}") as child:
                child.set_inputs({"text": text, "index": i})
                child.set_attributes({"position": i, "text_length": len(text)})

                words = text.split()
                result = {
                    "text": text,
                    "word_count": len(words),
                    "avg_word_length": round(
                        sum(len(w) for w in words) / max(len(words), 1), 2
                    ),
                }
                total_words += len(words)
                all_results.append(result)
                child.set_outputs(result)

        summary = {
            "total_texts": len(texts),
            "total_words": total_words,
            "results": all_results,
        }
        root_span.set_outputs(summary)
        return summary


# --------------------------------------------------------------------- #
# Part 3: Combining manual tracing with autolog
# --------------------------------------------------------------------- #


@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    """Summarize text using an LLM, with manual + auto tracing combined."""
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

    with mlflow.start_span(name="prepare_prompt") as span:
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": f"Summarize this in one sentence:\n\n{text}"},
        ]
        span.set_inputs({"text_length": len(text)})
        span.set_outputs({"message_count": len(messages)})

    # This call is auto-traced by mlflow.openai.autolog()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main() -> None:
    print("=" * 60)
    print("Part 1: @mlflow.trace decorator")
    print("=" * 60)

    result = process_pipeline("  Hello world from MLflow tracing  ")
    print(f"  Pipeline result: {result}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}")
    print()
    print("  The trace shows process_pipeline as the root span,")
    print("  with validate_text and transform_text as nested children.")
    print()

    print("=" * 60)
    print("Part 2: mlflow.start_span() context manager")
    print("=" * 60)

    texts = [
        "MLflow makes tracking easy",
        "Tracing shows execution flow",
        "Spans capture details",
    ]
    summary = analyze_texts(texts)
    print(f"  Analyzed {summary['total_texts']} texts, {summary['total_words']} total words")
    for r in summary["results"]:
        print(f"    - \"{r['text']}\" -> {r['word_count']} words, avg len {r['avg_word_length']}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}")
    print()
    print("  The trace shows batch_analysis as the root span,")
    print("  with analyze_item_0/1/2 as children — each with custom attributes.")
    print()

    print("=" * 60)
    print("Part 3: Combining manual tracing with autolog")
    print("=" * 60)

    mlflow.openai.autolog()

    sample_text = (
        "MLflow is an open-source platform for managing the end-to-end "
        "machine learning lifecycle. It provides tools for experiment tracking, "
        "model registry, deployment, and evaluation."
    )
    result = summarize_with_llm(sample_text)
    print(f"  LLM summary: {result}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}")
    print()
    print("  The trace shows summarize_with_llm as the root span,")
    print("  with prepare_prompt (manual) and chat.completions (auto)")
    print("  as nested children — both in one unified trace tree.")
    print()

    print("=" * 60)
    print("Done! View traces in the MLflow UI:")
    print("  http://127.0.0.1:5000")
    print("  Look for experiment: L1/M2_tracing/2_manual_tracing")
    print("  Click any trace to see the span tree.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M2_tracing/2_manual_tracing")

    main()
