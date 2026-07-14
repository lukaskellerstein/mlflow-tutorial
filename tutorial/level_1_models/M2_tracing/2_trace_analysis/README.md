# L2-M4.4 — Trace-based Debugging and Analysis

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

Traces capture the full execution flow of your LLM applications, but their real value comes from systematic analysis. This lesson shows how to programmatically retrieve traces with `mlflow.search_traces()`, extract span-level latency data to find bottlenecks, inspect token usage for cost estimation, and build summary reports that you can log back to MLflow as artifacts.

## Prerequisites

- Completed: L1-M5.1 (Auto Tracing), L1-M5.2 (Manual Tracing)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why Analyze Traces Programmatically?

The MLflow UI is great for exploring individual traces, but when you have hundreds or thousands of traces across many runs, you need programmatic access to answer questions like:

- **Which operations are slowest?** Identify latency bottlenecks across all your traces.
- **How much does each call cost?** Aggregate token usage to estimate API costs.
- **Are there patterns in failures?** Find traces with errors and correlate with inputs.
- **How do changes affect performance?** Compare latency distributions before and after a code change.

### The Trace Data Model

Each trace contains:
- **TraceInfo** — metadata like `trace_id`, `execution_duration` (ms), `request_time`, `state`, and `token_usage`
- **TraceData** — the actual `spans` list, plus `request` and `response` for the root span

Each span has:
- `name` — the operation name (e.g., `ChatOpenAI`)
- `span_type` — the category (e.g., `LLM`, `CHAIN`)
- `start_time_ns` / `end_time_ns` — nanosecond timestamps for duration calculation
- `inputs` / `outputs` — the data flowing through the span
- Attributes like `mlflow.chat.tokenUsage` for token counts

## Step-by-Step

### Step 1: Generate Traces

We run four different LLM calls to produce a variety of traces with different complexities:

```python
mlflow.langchain.autolog()
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="google/gemma-4-e4b", temperature=0.7)

# Direct LLM invocation — simple Q&A
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant. Answer in one sentence."),
    HumanMessage(content="What is the speed of light?"),
])
```

Each `llm.invoke()` call creates a separate trace with spans captured by autolog.

### Step 2: Retrieve Traces with `search_traces()`

Use `mlflow.search_traces()` to programmatically fetch all traces from an experiment:

```python
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    return_type="list",   # returns list of Trace objects
    flush=True,           # ensure async writes are flushed
)
```

The `return_type="list"` gives you `Trace` objects with full access to `trace.info` and `trace.data.spans`. You can also use `return_type="pandas"` for a DataFrame.

### Step 3: Latency Analysis

Calculate span durations from nanosecond timestamps and find the slowest operations:

```python
for span in trace.data.spans:
    duration_ms = (span.end_time_ns - span.start_time_ns) / 1_000_000
```

Group by span type to see which categories (LLM calls, parsing, etc.) consume the most time.

### Step 4: Token Usage Analysis

Token counts may be available at the trace level or in span attributes:

```python
# Trace-level aggregated usage
trace_usage = trace.info.token_usage  # dict with input_tokens, output_tokens, total_tokens

# Span-level usage (for individual LLM calls)
usage = span.get_attribute("mlflow.chat.tokenUsage")
```

Note: Local models may not report token counts through LangChain autolog. Cloud-hosted LLMs (OpenAI, Anthropic) reliably populate these fields.

### Step 5: Build and Log an Analysis Report

Create a summary DataFrame and log it as an MLflow artifact:

```python
with mlflow.start_run(run_name="trace_analysis_report"):
    report_df.to_csv("trace_analysis_report.csv", index=False)
    mlflow.log_artifact("trace_analysis_report.csv")
    mlflow.log_metrics({
        "total_traces": len(report_df),
        "avg_duration_ms": report_df["total_duration_ms"].mean(),
    })
```

## Running the Lesson

```bash
cd tutorial/level_2/M4_advanced_tracing/4_trace_analysis
uv sync
uv run python main.py
```

## Expected Output

```
Part 1: Generating traces from LLM calls
  Running: Simple Q&A
  Result:  The speed of light in a vacuum is approximately 299,792,458 meters ...
  Running: Translation
  Result:  Bonjour, comment allez-vous aujourd'hui ?...
  Running: Summarization
  Result:  Containerization packages applications and their dependencies into ...
  Running: Multi-step (summarize + title)
  Result:  Deploying Intelligence: A Guide to ML Model Deployment...

Part 2: Latency Analysis
  Top 5 slowest spans:
  Span Name                 Type         Duration (ms)
  ChatOpenAI                LLM                 3200.5
  ChatOpenAI                LLM                 2800.3
  ...

  Average duration by span type:
  LLM                  avg=  2500.0 ms  (n=5)
  CHAIN                avg=  2600.0 ms  (n=5)
  ...

Part 3: Token Usage Analysis
  Token usage data not available in traces.
  (Local models may not report token counts via LangChain autolog.)

Part 4: Analysis Report
  Trace Analysis Summary:
  Trace ID               Duration  Spans Slowest Span         Tokens
  tr-abc123...               3200      3 ChatOpenAI              N/A
  ...

  Logged report artifacts and metrics to MLflow run.
```

In the MLflow UI:
- Navigate to experiment `L2/M4_advanced_tracing/4_trace_analysis`
- **Traces tab**: browse individual trace timelines and span hierarchies
- **Runs tab**: find the `trace_analysis_report` run with CSV artifacts and summary metrics

## Key Takeaways

- `mlflow.search_traces()` provides programmatic access to all traces in an experiment, returning either a pandas DataFrame or a list of `Trace` objects.
- Span-level timing data (`start_time_ns`, `end_time_ns`) lets you pinpoint latency bottlenecks across your entire application.
- Token usage is available at both the trace level (`trace.info.token_usage`) and span level (`span.get_attribute("mlflow.chat.tokenUsage")`), though availability depends on the LLM provider.
- Analysis reports can be logged back to MLflow as artifacts, creating a feedback loop between tracing and tracking.
- This approach scales to production: schedule periodic trace analysis to detect regressions in latency or cost.

## Next Steps

In **Level 2, M5** (Agent Observability), you will apply these trace analysis techniques to LangChain and LangGraph agents, where traces become much richer with tool calls, reasoning steps, and multi-agent handoffs.
