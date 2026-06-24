# L2-M4.2 — Tracing Temporal.io Workflows

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

This lesson integrates MLflow tracing with Temporal.io durable workflows. You will build a text-analysis pipeline that runs as a Temporal workflow with three sequential activities (summarize, sentiment analysis, keyword extraction), each instrumented with MLflow spans. The result is dual observability: Temporal tracks execution durability and retries, while MLflow captures LLM-level inputs, outputs, and timing.

## Prerequisites

- Completed: L2-M4.1 (LangGraph Tracing) -- familiarity with MLflow tracing concepts
- MLflow server running at http://127.0.0.1:5000
- Temporal server running at localhost:7233 (start with `podman compose up -d` from `infra/`)
- Ollama running locally with `gemma4:e2b` pulled

## Concepts

### Why Temporal + MLflow?

Temporal.io provides **durable execution** -- workflows survive process crashes, network failures, and infrastructure restarts. Activities are automatically retried on failure. This makes Temporal ideal for orchestrating multi-step AI pipelines in production.

However, Temporal's built-in observability focuses on workflow execution (task scheduling, retries, timeouts) rather than AI-specific concerns (LLM inputs/outputs, token counts, response quality). MLflow tracing fills this gap by capturing what happens *inside* each activity at the LLM level.

Together, they provide:
- **Temporal**: Was the workflow completed? How many retries? What was the execution timeline?
- **MLflow**: What did the LLM produce? How long did each inference take? What were the inputs?

### Tracing Architecture

Temporal's workflow sandbox restricts imports for determinism, so the workflow class must live in a separate module (`workflow_def.py`) that does not import MLflow. Activities, which run outside the sandbox in a thread pool, freely use MLflow tracing.

The overall workflow invocation is wrapped with `@mlflow.trace` on the caller side. Activity spans are created independently because Temporal activities run in separate thread contexts:

```
Caller side (main.py):
  @mlflow.trace("temporal_workflow_execution")   <-- workflow-level trace
    mlflow.start_span("temporal_dispatch")        <-- dispatch span

Activity side (separate threads):
  mlflow.start_span("activity_summarize")         <-- own trace
  mlflow.start_span("activity_sentiment")         <-- own trace
  mlflow.start_span("activity_keywords")          <-- own trace
```

This means you will see separate traces in MLflow: one for the workflow dispatch and one for each activity. This accurately reflects Temporal's execution model where activities are independently scheduled units of work.

### File Structure

```
2_temporal_tracing/
  main.py           # Activities, MLflow tracing, runner, analysis
  workflow_def.py   # Workflow class + dataclasses (no MLflow imports)
  pyproject.toml
  README.md
```

## Step-by-Step

### Step 1: Define Activities with MLflow Tracing

Each Temporal activity wraps an Ollama LLM call inside an `mlflow.start_span()` context manager. This captures the task type, input text length, output preview, and duration.

```python
@activity.defn
async def analyze_text(request: AnalysisRequest) -> AnalysisResult:
    with mlflow.start_span(name=f"activity_{request.task}") as span:
        span.set_inputs({"task": request.task, "text_length": len(request.text)})
        result_text = _call_llm(system_prompt, request.text)
        span.set_outputs({"result_preview": result_text[:200]})
    return AnalysisResult(task=request.task, result=result_text, ...)
```

### Step 2: Define the Temporal Workflow (workflow_def.py)

The workflow is in a separate module to avoid Temporal's sandbox restrictions on MLflow. It orchestrates three activities in sequence, each with a 120-second timeout (LLM calls can be slow with local models).

```python
@workflow.defn
class TextAnalysisPipeline:
    @workflow.run
    async def run(self, text: str) -> PipelineResult:
        summary = await workflow.execute_activity(analyze_text, ...)
        sentiment = await workflow.execute_activity(analyze_text, ...)
        keywords = await workflow.execute_activity(analyze_text, ...)
        return PipelineResult(...)
```

### Step 3: Execute with Tracing

The `run_workflow` function is decorated with `@mlflow.trace` to create a root span for the workflow invocation. Inside, a `temporal_dispatch` child span captures the Temporal workflow execution timing.

### Step 4: Analyze Traces

After execution, we query MLflow traces to compare per-activity timing, identify bottlenecks, and see how Temporal's orchestration overhead compares to actual LLM inference time.

## Running the Lesson

```bash
cd tutorial/level_2/M4_advanced_tracing/2_temporal_tracing
uv sync
uv run python main.py
```

If Temporal is not running, the lesson falls back to running activities locally with MLflow tracing (no Temporal orchestration). Start Temporal first for the full experience:

```bash
cd infra
podman compose up -d
```

## Expected Output

```
============================================================
Part 1-2: Connecting to Temporal and registering worker
============================================================
  Connected to Temporal at localhost:7233

============================================================
Part 3: Running the Temporal workflow with MLflow tracing
============================================================

  --- Workflow: mlflow-temporal-demo-1-a1b2c3d4 ---
  Input text: Artificial intelligence is reshaping industries worldwide...
  Summary:   AI is transforming industries from healthcare to autonomous...
  Sentiment: The text has a mixed sentiment...
  Keywords:  artificial intelligence, healthcare, autonomous vehicles...

  --- Workflow: mlflow-temporal-demo-2-e5f6a7b8 ---
  ...

============================================================
Part 4: Analyzing MLflow traces
============================================================

  Showing up to 8 most recent trace(s)

--------------------------------------------------
  Trace 1  |  ID: tr-abc123...
--------------------------------------------------
    Spans:
    -> activity_keywords                     4803.8 ms  [UNKNOWN]
    ...

--------------------------------------------------
  Trace 4  |  ID: tr-def456...
--------------------------------------------------
    Spans:
    -> temporal_workflow_execution          63707.6 ms  [WORKFLOW]
      -> temporal_dispatch                   63706.9 ms  [UNKNOWN]
    Total workflow time: 63707.6 ms
```

In the MLflow UI (http://127.0.0.1:5000), navigate to the experiment to see all traces. In the Temporal UI (http://localhost:8080), search for the workflow ID to see the execution history with activity scheduling and completion events.

## Key Takeaways

- Temporal provides durable execution (retries, crash recovery) while MLflow provides AI observability (LLM inputs/outputs, timing)
- Temporal's workflow sandbox restricts non-deterministic imports like MLflow, so the workflow class must be in a separate module
- Use `mlflow.start_span()` inside Temporal activities to trace LLM calls -- activities run outside the sandbox
- Activity traces appear as separate traces in MLflow because Temporal dispatches them in independent thread contexts
- Wrap the workflow invocation with `@mlflow.trace` to capture overall workflow timing on the caller side
- Comparing Temporal's execution timeline with MLflow's trace timeline reveals orchestration overhead vs. actual LLM inference time
- The fallback pattern (running activities directly) is useful for development without Temporal infrastructure

## Next Steps

In **L2-M4.3 (OpenTelemetry Integration)**, you will learn how to export MLflow traces to OpenTelemetry-compatible backends, enabling integration with broader observability platforms like Jaeger and Grafana Tempo.
