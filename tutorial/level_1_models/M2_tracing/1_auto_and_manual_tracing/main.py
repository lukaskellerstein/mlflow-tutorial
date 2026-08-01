"""
L1-M2.1 -- Auto-Tracing and Manual Tracing

Demonstrates both sides of MLflow tracing in one lesson:
- Part 1: mlflow.openai.autolog() -- automatic tracing for OpenAI SDK calls
- Part 2: mlflow.langchain.autolog() -- automatic tracing for LangChain agents
- Part 3: @mlflow.trace decorator -- manual tracing for custom functions
- Part 4: mlflow.start_span() -- fine-grained manual span control
- Part 5: Combining auto + manual tracing in a single trace tree
"""

import time

import mlflow
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
MODEL_NAME = "google/gemma-4-e4b"


# ── Part 1: mlflow.openai.autolog() -- Direct OpenAI SDK ──────────────


def part1_openai_autolog() -> None:
    """Trace a direct OpenAI SDK call to LMStudio automatically."""
    print("=" * 60)
    print("Part 1: mlflow.openai.autolog() -- Direct OpenAI SDK")
    print("=" * 60)

    mlflow.openai.autolog()

    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is MLflow? Answer in one sentence."}],
        temperature=0.7,
        max_tokens=1024,
    )

    print(f'  Response: {(response.choices[0].message.content or "")[:200]}')
    print()
    print("  [Autolog captured]")
    print("    - Input messages, output content, token usage, latency")
    print("    - Model name and parameters")
    print("    -> Check the Traces tab in MLflow UI")
    print()

    mlflow.openai.autolog(disable=True)


# ── Part 2: mlflow.langchain.autolog() -- LangChain Agent ─────────────


def get_current_time() -> str:
    """Return the current time as a formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def part2_langchain_autolog() -> None:
    """Trace a LangChain agent built with create_agent."""
    print("=" * 60)
    print("Part 2: mlflow.langchain.autolog() -- LangChain Agent")
    print("=" * 60)

    mlflow.langchain.autolog()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        temperature=0.7,
        max_tokens=1024,  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
    )

    agent = create_agent(
        model=llm,
        tools=[get_current_time],
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What time is it right now?"}]}
    )

    final_message = result["messages"][-1].content
    print(f"  Response: {final_message[:200]}")
    print()
    print("  [Autolog captured]")
    print("    - Full agent execution flow (model calls, tool calls)")
    print("    - Each step as a child span in the trace")
    print("    -> Expand the trace to see the agent reasoning loop")
    print()

    mlflow.langchain.autolog(disable=True)


# ── Part 3: @mlflow.trace decorator ───────────────────────────────────


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


def part3_trace_decorator() -> None:
    """Demonstrate @mlflow.trace for automatic span creation."""
    print("=" * 60)
    print("Part 3: @mlflow.trace decorator")
    print("=" * 60)

    result = process_pipeline("  Hello world from MLflow tracing  ")
    print(f"  Pipeline result: {result}")
    print(f"  Trace ID: {mlflow.get_last_active_trace_id()}")
    print()
    print("  The trace shows process_pipeline as the root span,")
    print("  with validate_text and transform_text as nested children.")
    print()


# ── Part 4: mlflow.start_span() context manager ──────────────────────


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


def part4_start_span() -> None:
    """Demonstrate mlflow.start_span() for fine-grained control."""
    print("=" * 60)
    print("Part 4: mlflow.start_span() context manager")
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
    print("  with analyze_item_0/1/2 as children -- each with custom attributes.")
    print()


# ── Part 5: Combining auto + manual tracing ───────────────────────────


@mlflow.trace(name="summarize_with_llm")
def summarize_with_llm(text: str) -> str:
    """Summarize text using an LLM, with manual + auto tracing combined."""
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

    with mlflow.start_span(name="prepare_prompt") as span:
        messages: list[ChatCompletionMessageParam] = [
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

    return response.choices[0].message.content or ""


def part5_combined() -> None:
    """Demonstrate combining manual tracing with autolog."""
    print("=" * 60)
    print("Part 5: Combining manual tracing with autolog")
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
    print("  as nested children -- both in one unified trace tree.")
    print()

    mlflow.openai.autolog(disable=True)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    part1_openai_autolog()
    part2_langchain_autolog()
    part3_trace_decorator()
    part4_start_span()
    part5_combined()

    print("=" * 60)
    print("Done! Open the MLflow UI to explore the traces:")
    print("  http://127.0.0.1:5555")
    print()
    print("Navigate to experiment 'L1/M2_tracing/1_auto_and_manual_tracing'")
    print("and click the Traces tab to see all captured traces.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L1/M2_tracing/1_auto_and_manual_tracing")

    main()
