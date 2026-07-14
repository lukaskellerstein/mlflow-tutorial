"""
L1-5.2 — Manual Tracing

Demonstrates MLflow's manual tracing APIs:
- @mlflow.trace decorator for automatic span creation
- mlflow.start_span() context manager for fine-grained control
- Combining manual tracing with LangChain auto-tracing
"""

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------ #
# Part 1: @mlflow.trace decorator
# ------------------------------------------------------------------ #

@mlflow.trace(name="validate_text")
def validate_text(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    return text.strip()

@mlflow.trace(name="transform_text")
def transform_text(text: str) -> dict:
    words = text.split()
    return {"original": text, "word_count": len(words),
            "char_count": len(text), "uppercase": text.upper()}

@mlflow.trace(name="process_pipeline")
def process_pipeline(text: str) -> dict:
    validated = validate_text(text)
    return transform_text(validated)

# ------------------------------------------------------------------ #
# Part 2: mlflow.start_span() context manager
# ------------------------------------------------------------------ #

def analyze_texts(texts: list[str]) -> dict:
    with mlflow.start_span(name="batch_analysis") as root_span:
        root_span.set_inputs({"texts": texts, "count": len(texts)})
        all_results, total_words = [], 0

        for i, text in enumerate(texts):
            with mlflow.start_span(name=f"analyze_item_{i}") as child:
                child.set_inputs({"text": text, "index": i})
                child.set_attributes({"position": i, "text_length": len(text)})
                words = text.split()
                result = {"text": text, "word_count": len(words),
                          "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2)}
                total_words += len(words)
                all_results.append(result)
                child.set_outputs(result)

        summary = {"total_texts": len(texts), "total_words": total_words, "results": all_results}
        root_span.set_outputs(summary)
        return summary

# ------------------------------------------------------------------ #
# Part 3: Combining auto + manual tracing
# ------------------------------------------------------------------ #

@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    llm = ChatOpenAI(
        model="google/gemma-4-e4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("user", "Summarize this in one sentence:\n\n{text}")]
    )
    chain = prompt | llm
    return chain.invoke({"text": text}).content

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print("Part 1: @mlflow.trace decorator")
    print("=" * 60)
    result = process_pipeline("  Hello world from MLflow tracing  ")
    print(f"  Pipeline result: {result}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}\n")

    print("=" * 60)
    print("Part 2: mlflow.start_span() context manager")
    print("=" * 60)
    texts = ["MLflow makes tracking easy", "Tracing shows execution flow", "Spans capture details"]
    summary = analyze_texts(texts)
    print(f"  Analyzed {summary['total_texts']} texts, {summary['total_words']} total words")
    for r in summary["results"]:
        print(f"    - \"{r['text']}\" -> {r['word_count']} words, avg len {r['avg_word_length']}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}\n")

    print("=" * 60)
    print("Part 3: Combining auto + manual tracing")
    print("=" * 60)
    mlflow.langchain.autolog()
    sample_text = (
        "MLflow is an open-source platform for managing the end-to-end "
        "machine learning lifecycle. It provides tools for experiment tracking, "
        "model registry, deployment, and evaluation."
    )
    print(f"  LLM summary: {summarize_with_llm(sample_text)}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}\n")

    print("=" * 60)
    print("Done! View traces in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("  Look for experiment: L1/M5_tracing/2_manual_tracing")
    print("  Click any run, then open the 'Traces' tab to see spans.")
    print("=" * 60)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M5_tracing/2_manual_tracing")
    main()
