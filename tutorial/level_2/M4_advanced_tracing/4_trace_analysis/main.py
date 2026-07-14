"""
L2-4.4 — Trace-based Debugging and Analysis

Demonstrates how to use mlflow.search_traces() to retrieve traces,
analyze span durations to find latency bottlenecks, extract token
usage, and build a summary report logged as an MLflow artifact.
"""

import time

import mlflow
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Part 1: Generate traces by running several LangChain chains
# ---------------------------------------------------------------------------

def generate_traces(llm: ChatOpenAI) -> None:
    """Run 4 different chains to produce varied traces for analysis."""
    print("=" * 60)
    print("Part 1: Generating traces from LangChain chains")
    print("=" * 60)

    # Chain A — simple question
    simple_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in one sentence."),
        ("human", "{question}"),
    ])
    simple_chain = simple_prompt | llm | StrOutputParser()

    # Chain B — translation
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Translate the text to {language}."),
        ("human", "{text}"),
    ])
    translate_chain = translate_prompt | llm | StrOutputParser()

    # Chain C — summarization (longer output expected)
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical writer."),
        ("human", "Write a brief summary about {topic} in 3-4 sentences."),
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # Chain D — multi-step: generate then title
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "You create short, catchy titles."),
        ("human", "Create a title for this text:\n\n{text}"),
    ])
    title_chain = title_prompt | llm | StrOutputParser()

    # Execute chains
    chains = [
        ("Simple Q&A", lambda: simple_chain.invoke(
            {"question": "What is the speed of light?"})),
        ("Translation", lambda: translate_chain.invoke(
            {"text": "Hello, how are you today?", "language": "French"})),
        ("Summarization", lambda: summarize_chain.invoke(
            {"topic": "containerization with Docker and Podman"})),
        ("Multi-step (summarize + title)", lambda: (
            title_chain.invoke({"text": summarize_chain.invoke(
                {"topic": "machine learning model deployment"})})
        )),
    ]

    for name, run_fn in chains:
        print(f"\n  Running: {name}")
        result = run_fn()
        print(f"  Result:  {result[:80]}...")

    # Give async logging a moment to flush
    time.sleep(2)
    print(f"\n  Generated {len(chains)} chain executions (5 total LLM calls)")
    print()


# ---------------------------------------------------------------------------
# Part 2: Latency analysis — find the slowest spans
# ---------------------------------------------------------------------------

def analyze_latency(traces: list) -> pd.DataFrame:
    """Extract span durations and identify the slowest operations."""
    print("=" * 60)
    print("Part 2: Latency Analysis")
    print("=" * 60)

    span_records = []
    for trace in traces:
        for span in trace.data.spans:
            start_ns = span.start_time_ns
            end_ns = span.end_time_ns
            if start_ns and end_ns:
                duration_ms = (end_ns - start_ns) / 1_000_000
            else:
                duration_ms = 0.0

            span_records.append({
                "trace_id": trace.info.trace_id,
                "span_name": span.name,
                "span_type": span.span_type or "UNKNOWN",
                "duration_ms": round(duration_ms, 2),
            })

    span_df = pd.DataFrame(span_records)

    if span_df.empty:
        print("  No span data found.")
        return span_df

    # Slowest individual spans
    slowest = span_df.nlargest(5, "duration_ms")
    print("\n  Top 5 slowest spans:")
    print("  " + "-" * 56)
    print(f"  {'Span Name':<25} {'Type':<12} {'Duration (ms)':>12}")
    print("  " + "-" * 56)
    for _, row in slowest.iterrows():
        print(f"  {row['span_name']:<25} {row['span_type']:<12} {row['duration_ms']:>12.1f}")

    # Average duration by span type
    print("\n  Average duration by span type:")
    print("  " + "-" * 40)
    type_avg = span_df.groupby("span_type")["duration_ms"].agg(["mean", "count"])
    for stype, row in type_avg.iterrows():
        print(f"  {stype:<20} avg={row['mean']:>8.1f} ms  (n={int(row['count'])})")

    print()
    return span_df


# ---------------------------------------------------------------------------
# Part 3: Token usage analysis
# ---------------------------------------------------------------------------

def analyze_token_usage(traces: list) -> pd.DataFrame:
    """Extract token counts from trace metadata and span attributes."""
    print("=" * 60)
    print("Part 3: Token Usage Analysis")
    print("=" * 60)

    token_records = []
    for trace in traces:
        # Check trace-level aggregated token usage
        trace_usage = trace.info.token_usage
        trace_id = trace.info.trace_id

        if trace_usage:
            token_records.append({
                "trace_id": trace_id,
                "source": "trace_metadata",
                "input_tokens": trace_usage.get("input_tokens", 0),
                "output_tokens": trace_usage.get("output_tokens", 0),
                "total_tokens": trace_usage.get("total_tokens", 0),
            })
        else:
            # Fall back to span-level token usage
            total_input = 0
            total_output = 0
            for span in trace.data.spans:
                usage = span.get_attribute("mlflow.chat.tokenUsage")
                if usage:
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)

            total = total_input + total_output
            token_records.append({
                "trace_id": trace_id,
                "source": "span_attributes" if total > 0 else "unavailable",
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total,
            })

    token_df = pd.DataFrame(token_records)

    if token_df.empty:
        print("  No token data found.")
        return token_df

    has_tokens = token_df["total_tokens"].sum() > 0

    if has_tokens:
        print("\n  Per-trace token usage:")
        print("  " + "-" * 60)
        print(f"  {'Trace ID':<30} {'Input':>8} {'Output':>8} {'Total':>8}")
        print("  " + "-" * 60)
        for _, row in token_df.iterrows():
            tid = row["trace_id"][:28] + ".."
            print(f"  {tid:<30} {row['input_tokens']:>8} "
                  f"{row['output_tokens']:>8} {row['total_tokens']:>8}")

        total_all = token_df["total_tokens"].sum()
        print(f"\n  Total tokens across all traces: {total_all}")
        print(f"  Note: LMStudio local models do not have per-token pricing.")
        print(f"  For cloud APIs, cost = input_tokens * rate + output_tokens * rate")
    else:
        print("\n  Token usage data not available in traces.")
        print("  (LMStudio may not report token counts via LangChain autolog.)")
        print("  For cloud-hosted LLMs (OpenAI, Anthropic), token counts")
        print("  are automatically captured in span attributes.")

    print()
    return token_df


# ---------------------------------------------------------------------------
# Part 4: Build and log a summary analysis report
# ---------------------------------------------------------------------------

def build_analysis_report(
    traces: list, span_df: pd.DataFrame, token_df: pd.DataFrame
) -> pd.DataFrame:
    """Create a per-trace summary and log it as an MLflow artifact."""
    print("=" * 60)
    print("Part 4: Analysis Report")
    print("=" * 60)

    report_rows = []
    for trace in traces:
        trace_id = trace.info.trace_id
        duration_ms = trace.info.execution_duration  # in milliseconds
        spans = trace.data.spans
        num_spans = len(spans)

        # Find the slowest span within this trace
        slowest_span_name = "N/A"
        slowest_span_ms = 0.0
        for span in spans:
            start_ns = span.start_time_ns
            end_ns = span.end_time_ns
            if start_ns and end_ns:
                span_ms = (end_ns - start_ns) / 1_000_000
                if span_ms > slowest_span_ms:
                    slowest_span_ms = span_ms
                    slowest_span_name = span.name

        # Get token info for this trace
        token_row = token_df[token_df["trace_id"] == trace_id]
        total_tokens = int(token_row["total_tokens"].iloc[0]) if not token_row.empty else 0

        report_rows.append({
            "trace_id": trace_id,
            "total_duration_ms": duration_ms or 0,
            "num_spans": num_spans,
            "slowest_span": slowest_span_name,
            "slowest_span_ms": round(slowest_span_ms, 1),
            "total_tokens": total_tokens,
        })

    report_df = pd.DataFrame(report_rows)

    # Print the report
    print("\n  Trace Analysis Summary:")
    print("  " + "-" * 78)
    print(f"  {'Trace ID':<22} {'Duration':>10} {'Spans':>6} "
          f"{'Slowest Span':<20} {'Tokens':>7}")
    print("  " + "-" * 78)
    for _, row in report_df.iterrows():
        tid = row["trace_id"][:20] + ".."
        dur = f"{row['total_duration_ms']}ms"
        tokens = str(row["total_tokens"]) if row["total_tokens"] > 0 else "N/A"
        print(f"  {tid:<22} {dur:>10} {row['num_spans']:>6} "
              f"{row['slowest_span']:<20} {tokens:>7}")

    # Aggregate stats
    print(f"\n  Total traces analyzed:  {len(report_df)}")
    print(f"  Total spans analyzed:   {report_df['num_spans'].sum()}")
    avg_dur = report_df["total_duration_ms"].mean()
    print(f"  Average trace duration: {avg_dur:.0f} ms")
    max_dur = report_df["total_duration_ms"].max()
    print(f"  Longest trace:          {max_dur} ms")

    # Log the report as an artifact in a new run
    with mlflow.start_run(run_name="trace_analysis_report"):
        report_path = "trace_analysis_report.csv"
        report_df.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)

        # Also log the span-level detail
        if not span_df.empty:
            span_path = "span_latency_detail.csv"
            span_df.to_csv(span_path, index=False)
            mlflow.log_artifact(span_path)

        # Log summary metrics
        mlflow.log_metrics({
            "total_traces": len(report_df),
            "total_spans": int(report_df["num_spans"].sum()),
            "avg_duration_ms": round(avg_dur, 1),
            "max_duration_ms": float(max_dur or 0),
        })

        print(f"\n  Logged report artifacts and metrics to MLflow run.")

    # Clean up local CSV files
    import os
    for f in ["trace_analysis_report.csv", "span_latency_detail.csv"]:
        if os.path.exists(f):
            os.remove(f)

    print()
    return report_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full trace analysis pipeline."""

    # Step 1 — Enable autologging and generate traces
    mlflow.langchain.autolog()
    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )
    generate_traces(llm)

    # Step 2 — Retrieve traces via search_traces
    print("Searching for traces in the experiment...")
    experiment = mlflow.get_experiment_by_name(
        "L2/M4_advanced_tracing/4_trace_analysis"
    )
    if experiment is None:
        print("  ERROR: Experiment not found. Did the chains run successfully?")
        return

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        return_type="list",
        flush=True,
    )
    print(f"  Found {len(traces)} traces\n")

    if not traces:
        print("  No traces to analyze. Exiting.")
        return

    # Step 3 — Latency analysis
    span_df = analyze_latency(traces)

    # Step 4 — Token usage analysis
    token_df = analyze_token_usage(traces)

    # Step 5 — Build and log summary report
    build_analysis_report(traces, span_df, token_df)

    print("=" * 60)
    print("Done! Open the MLflow UI to explore:")
    print("  http://127.0.0.1:5000")
    print()
    print("  - Experiment: L2/M4_advanced_tracing/4_trace_analysis")
    print("  - Traces tab: browse individual trace timelines")
    print("  - Run 'trace_analysis_report': CSV artifacts + metrics")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M4_advanced_tracing/4_trace_analysis")

    main()
