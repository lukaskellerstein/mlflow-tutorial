# L1-M1.2 — Tracking LLM Experiments

**Level:** Essentials
**Duration:** 30 min

## Overview

This lesson dives deeper into MLflow's tracking capabilities using real LLM experiments. You will compare different temperature settings, log LLM responses as artifacts, and use step-based metrics to track token usage across multiple prompts.

## Prerequisites

- Completed: L1-M1.1 (Your First MLflow Run)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Track LLM Experiments?

When working with LLMs, you constantly tweak configurations — temperature, max tokens, system prompts, model choice. Without tracking, it's impossible to remember which settings produced the best results. MLflow tracking gives you:

- **Reproducibility** — every configuration is recorded as parameters
- **Comparison** — view runs side-by-side in the UI
- **History** — step-based metrics show how token usage evolves across prompts
- **Artifacts** — save the actual LLM responses for later review

### Tracking APIs Used

| API | Purpose |
|-----|---------|
| `mlflow.log_params()` | Log LLM configuration (model, temperature, max_tokens) |
| `mlflow.log_metrics()` | Log numeric results (response time, token counts) |
| `mlflow.log_metric(key, value, step=N)` | Log a metric at a specific step (for time-series) |
| `mlflow.log_artifact()` | Save a file (response text, summary) to the run |
| `mlflow.set_tags()` | Add metadata for organizing and filtering runs |

## Step-by-Step

### Step 1: Compare Temperatures

We run the same prompt at three temperatures (0.3, 0.7, 1.0) and create a separate MLflow run for each. This lets us compare side-by-side in the UI:

```python
for temp in [0.3, 0.7, 1.0]:
    with mlflow.start_run(run_name=f"temp_{temp}"):
        mlflow.log_params({"model": MODEL, "temperature": temp, ...})
        result = call_llm(client, prompt, temperature=temp)
        mlflow.log_metrics({"response_time_seconds": ..., "total_tokens": ...})
```

Each run also saves the LLM response as a text artifact using `mlflow.log_artifact()`.

### Step 2: Step-Based Metrics

Step-based logging records a metric at sequential steps — useful for tracking how token usage accumulates across multiple prompts:

```python
for step, prompt in enumerate(prompts):
    result = call_llm(client, prompt)
    mlflow.log_metric("cumulative_tokens", cumulative, step=step)
```

In the MLflow UI, this renders as a line chart showing token growth over steps.

### Step 3: Summary Artifact

We generate a Markdown file containing multiple LLM responses and save it as an artifact. This is useful for reviewing and comparing outputs later:

```python
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "summary.md")
    with open(path, "w") as f:
        f.write(summary_content)
    mlflow.log_artifact(path)
```

## Running the Lesson

```bash
cd tutorial/level_1/M1_tracking/2_tracking_basics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Comparing temperatures (0.3, 0.7, 1.0)
============================================================
  temp=0.3  tokens=  85  time=2.1s
  temp=0.7  tokens=  92  time=2.3s
  temp=1.0  tokens= 110  time=2.8s

============================================================
Step 2: Step-based metric logging across multiple prompts
============================================================
  Step 0  'What is a transformer model?'  tokens=45  cumulative=45
  Step 1  'What is attention in machine learning?'  tokens=52  cumulative=97
  ...

============================================================
Step 3: Logging a summary artifact
============================================================
  Logged summary.md with 3 responses
```

## Key Takeaways

- Use separate runs to compare LLM configurations (temperature, model, prompts) side-by-side.
- `log_artifact()` saves files (responses, summaries) attached to a run — reviewable in the UI.
- Step-based `log_metric(..., step=N)` creates time-series charts — useful for tracking cumulative token usage or per-step latency.
- `log_params()` and `log_metrics()` accept dictionaries for bulk logging.

## Next Steps

Continue to **L1-M1.3 Search and Query API** where you will learn to programmatically search, filter, and compare your tracked runs using MLflow's query API.
