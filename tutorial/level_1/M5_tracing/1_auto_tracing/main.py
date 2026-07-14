"""
L1-5.1 — Automatic Tracing

Demonstrates MLflow's automatic tracing for LangChain:
- Enable auto-tracing with mlflow.langchain.autolog()
- Simple chain tracing (prompt -> LLM -> output parser)
- Multi-step chain tracing (parent-child span relationships)
- Searching and inspecting traces programmatically
"""

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def part1_simple_chain() -> None:
    """Part 1: Trace a simple prompt -> LLM -> parser chain."""
    print("=" * 60)
    print("Part 1: Simple Chain Tracing")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Keep answers to one sentence."),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(
        model="google/gemma-4-e4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )
    chain = prompt | llm | StrOutputParser()

    print("  Chain: ChatPromptTemplate -> ChatOpenAI -> StrOutputParser")
    print("  Invoking chain...")

    result = chain.invoke({"question": "What is MLflow?"})

    print(f"  Response: {result[:120]}...")
    print("\n  A trace was automatically created with one span per chain step.")
    print("  View it in MLflow UI -> Traces tab")


def part2_multi_step_chain() -> None:
    """Part 2: Trace a multi-step chain showing parent-child spans."""
    print("\n" + "=" * 60)
    print("Part 2: Multi-Step Chain Tracing")
    print("=" * 60)

    llm = ChatOpenAI(
        model="google/gemma-4-e4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )

    # Step 1: Summarize a topic
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the topic in exactly two sentences."),
        ("human", "Topic: {topic}"),
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # Step 2: Generate questions from the summary
    questions_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given this summary, generate exactly 3 short quiz questions. "
                   "Number them 1-3."),
        ("human", "Summary: {summary}"),
    ])
    questions_chain = questions_prompt | llm | StrOutputParser()

    print("  Chain 1: Summarize a topic")
    print("  Chain 2: Generate questions from the summary")
    print("  Invoking multi-step pipeline...\n")

    summary = summarize_chain.invoke({"topic": "machine learning experiment tracking"})
    print(f"  Summary: {summary[:120]}...")

    questions = questions_chain.invoke({"summary": summary})
    print(f"  Questions:\n{questions[:200]}...")

    print("\n  Two traces were created — one per chain invocation.")
    print("  Each trace has parent-child spans:")
    print("    Root span (chain) -> Prompt -> LLM -> OutputParser")


def part3_search_traces() -> None:
    """Part 3: Search and inspect traces programmatically."""
    print("\n" + "=" * 60)
    print("Part 3: Searching Traces")
    print("=" * 60)

    # Get experiment ID for scoping the search
    experiment = mlflow.get_experiment_by_name("L1/M5_tracing/1_auto_tracing")
    if experiment is None:
        print("  No experiment found — run Parts 1-2 first.")
        return

    exp_id = experiment.experiment_id

    # Search traces as a list of Trace objects (flush=True ensures
    # all async-logged traces are persisted before we query)
    traces = mlflow.search_traces(
        locations=[exp_id],
        max_results=5,
        return_type="list",
        flush=True,
    )

    print(f"  Found {len(traces)} trace(s) in the experiment\n")

    for i, trace in enumerate(traces):
        info = trace.info
        duration = info.execution_duration
        print(f"  --- Trace {i + 1} ---")
        print(f"  Trace ID:    {info.trace_id}")
        print(f"  State:       {info.state}")
        print(f"  Duration:    {duration} ms" if duration else "  Duration:    (pending)")

        # Show spans (the building blocks of a trace)
        spans = trace.data.spans
        print(f"  Spans ({len(spans)}):")
        for span in spans:
            parent_tag = "(root)" if span.parent_id is None else f"parent={span.parent_id[:8]}..."
            print(f"    - {span.name}  [{span.span_type}]  {parent_tag}")

        print()

    # Also show as a DataFrame for a tabular overview
    df = mlflow.search_traces(
        locations=[exp_id],
        max_results=5,
        return_type="pandas",
        flush=True,
    )
    print("  Trace DataFrame columns:", list(df.columns))
    print()

    print("  Key trace concepts:")
    print("    - Trace:  An end-to-end record of one chain invocation")
    print("    - Span:   A single step within a trace (prompt, LLM call, parser)")
    print("    - Parent: Spans nest — the chain span is parent to its steps")
    print("    - Each span records: inputs, outputs, start/end time, status")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M5_tracing/1_auto_tracing")

    # Enable automatic tracing for LangChain
    print("=" * 60)
    print("Enabling LangChain auto-tracing: mlflow.langchain.autolog()")
    print("=" * 60)
    mlflow.langchain.autolog()

    part1_simple_chain()
    part2_multi_step_chain()
    part3_search_traces()

    print("=" * 60)
    print("Done! View your traces in the MLflow UI:")
    print("  http://127.0.0.1:5000 -> Traces tab")
    print("=" * 60)
