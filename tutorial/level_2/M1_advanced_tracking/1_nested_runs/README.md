# L2-1.1 -- Nested Runs and Run Hierarchies

**Level:** Practitioner
**Duration:** 1 hour

## Overview

Learn how to organize related MLflow runs into parent-child hierarchies using nested runs. This lesson builds an LLM configuration sweep where a parent run groups nine child runs (three temperatures times three prompt variants), then demonstrates how to query and compare the children programmatically. Nested runs are the foundation for structured experimentation at scale.

## Prerequisites

- Completed: L1-1.2 Tracking Basics, L1-1.3 Search & Query API
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why Nested Runs?

In Level 1 you created individual runs. That works for quick experiments, but real-world LLM workflows generate dozens or hundreds of runs for a single question ("which temperature and prompt style combination produces the best responses?"). Without structure, the MLflow UI becomes a flat, unsorted list.

**Nested runs** solve this by letting you create parent-child relationships:

- A **parent run** represents the high-level task (e.g., "LLM config sweep" or "prompt engineering experiment").
- **Child runs** represent individual attempts within that task (e.g., one per temperature and prompt variant combination).

The parent groups everything together in the UI, and you can expand or collapse the hierarchy.

### How Nested Runs Work

The key is the `nested=True` parameter in `mlflow.start_run()`:

```python
with mlflow.start_run(run_name="LLM Config Sweep") as parent:
    for temperature in temperatures:
        for variant_name, system_prompt in prompt_variants.items():
            with mlflow.start_run(run_name=f"temp_{temperature}_style_{variant_name}", nested=True):
                # This run is a child of "LLM Config Sweep"
                mlflow.log_params({"temperature": temperature, "prompt_variant": variant_name})
                ...
```

When `nested=True`, MLflow automatically sets the tag `mlflow.parentRunId` on the child run, linking it to the currently active parent.

### Common Use Cases

| Pattern | Parent run | Child runs |
|---------|-----------|------------|
| LLM config sweep | The sweep | One per temperature x prompt variant |
| Prompt engineering | The experiment | One per prompt version |
| Model comparison | The comparison | One per LLM model |
| A/B testing | The experiment | One per variant |
| RAG pipeline tuning | The pipeline | One per chunking/retrieval strategy |

### Querying Child Runs

You can retrieve all children of a parent run using `mlflow.search_runs()` with a filter on the `mlflow.parentRunId` tag:

```python
child_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    order_by=["metrics.response_length DESC"],
)
```

This returns a pandas DataFrame that you can sort, filter, and analyze programmatically.

## Step-by-Step

### Step 1: Define the sweep grid

We test three temperatures with three prompt variants, giving nine configurations total:

```python
TEMPERATURES = [0.3, 0.7, 1.0]

PROMPT_VARIANTS = {
    "concise": "You are a helpful assistant. Be concise and brief. Answer in 2-3 sentences maximum.",
    "detailed": "You are a helpful assistant. Be detailed and thorough. Provide comprehensive explanations with examples.",
    "creative": "You are a helpful assistant. Be creative and engaging. Use analogies and vivid language to explain concepts.",
}

TEST_QUESTION = "Explain what MLflow is and why it is useful."
```

Each combination of temperature and prompt variant produces a different LLM response, letting you compare how these parameters affect output quality.

### Step 2: Create the parent run

```python
with mlflow.start_run(run_name="LLM Config Sweep") as parent_run:
    mlflow.set_tags({
        "sweep_type": "llm_config_sweep",
        "model": MODEL_NAME,
        "num_configs": str(num_configs),
    })
```

The parent run captures metadata about the sweep as a whole.

### Step 3: Create nested child runs

Inside the parent's context, each child run is created with `nested=True`:

```python
for temperature in TEMPERATURES:
    for variant_name, system_prompt in PROMPT_VARIANTS.items():
        with mlflow.start_run(run_name=f"temp_{temperature}_style_{variant_name}", nested=True):
            mlflow.log_params({"temperature": temperature, "prompt_variant": variant_name, "model": MODEL_NAME})

            response_text, token_count, latency = call_llm(client, temperature, system_prompt, TEST_QUESTION)

            mlflow.log_metrics({
                "response_length": len(response_text),
                "token_count": token_count,
                "latency_seconds": latency,
            })
```

Each child logs its own parameters, metrics, tags, and the LLM response as a text artifact.

### Step 4: Summarize on the parent

After all children complete, the parent run logs aggregate statistics:

```python
best_by_length = max(results, key=lambda r: r["response_length"])
fastest = min(results, key=lambda r: r["latency_seconds"])

mlflow.log_params({"best_config_by_length": best_by_length["run_name"]})
mlflow.log_metrics({
    "avg_latency_seconds": avg_latency,
    "avg_token_count": avg_tokens,
    "max_response_length": best_by_length["response_length"],
})
```

### Step 5: Query children with search_runs()

After the sweep, use `search_runs()` to retrieve all children, sorted by response length:

```python
child_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
    order_by=["metrics.response_length DESC"],
)
```

The result is a pandas DataFrame you can print as a summary table.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_tracking/1_nested_runs
uv sync
uv run python main.py
```

## Expected Output

Terminal output will look like:

```
============================================================
Step 1: LLM Configuration Sweep
============================================================
  Model:           google/gemma-4-e4b
  Temperatures:    [0.3, 0.7, 1.0]
  Prompt variants: ['concise', 'detailed', 'creative']
  Total configs:   9
  Question:        Explain what MLflow is and why it is useful.

============================================================
Step 2: Running sweep (nested runs)
============================================================
  temp_0.3_style_concise             length=  142  tokens=   58  latency=1.23s
  temp_0.3_style_detailed            length=  891  tokens=  234  latency=3.45s
  temp_0.3_style_creative            length=  523  tokens=  145  latency=2.10s
  temp_0.7_style_concise             length=  158  tokens=   63  latency=1.31s
  temp_0.7_style_detailed            length=  947  tokens=  251  latency=3.82s
  temp_0.7_style_creative            length=  612  tokens=  167  latency=2.44s
  temp_1.0_style_concise             length=  175  tokens=   71  latency=1.42s
  temp_1.0_style_detailed            length= 1023  tokens=  268  latency=4.15s
  temp_1.0_style_creative            length=  698  tokens=  189  latency=2.78s

============================================================
Step 3: Logging parent-run summary
============================================================
  Most detailed:   temp_1.0_style_detailed  (length=1023)
  Most concise:    temp_0.3_style_concise  (length=142)
  Fastest:         temp_0.3_style_concise  (latency=1.23s)
  Avg latency:     2.52s
  Avg tokens:      161
  Parent run ID:   <generated-id>

============================================================
Step 4: Querying child runs with search_runs()
============================================================

 run_id  temperature  prompt_variant  response_length  token_count  latency_seconds
 ...     1.0          detailed        1023             268          4.15
 ...     0.7          detailed        947              251          3.82
 ...     (remaining rows sorted by response_length)

============================================================
Done! View the nested run hierarchy in the MLflow UI:
  http://127.0.0.1:5000/#/experiments
  Expand the 'LLM Config Sweep' parent run to see children.
============================================================
```

In the MLflow UI you will see:

- The experiment **L2/M1_advanced_tracking/1_nested_runs** with a parent run named "LLM Config Sweep"
- Expanding the parent reveals nine child runs, each with its own parameters, metrics, and response artifact
- The parent run has summary metrics (`avg_latency_seconds`, `avg_token_count`) and tags pointing to the best child
- Each child run's artifacts folder contains the full LLM response text
- You can compare child runs side-by-side using the MLflow comparison view

## Key Takeaways

- Use `nested=True` in `mlflow.start_run()` to create parent-child run hierarchies.
- MLflow automatically sets `mlflow.parentRunId` on child runs, linking them to the parent.
- The parent run is the right place for sweep-level metadata and aggregate summary metrics.
- Use `search_runs()` with a `tags.mlflow.parentRunId` filter to programmatically retrieve children.
- Tags on child runs (`prompt_variant`, `temperature`) make filtering easy across large sweeps.
- LLM configuration sweeps (temperature, prompt style, model) are a natural fit for nested runs.
- Logging LLM responses as text artifacts lets you review and compare outputs in the MLflow UI.

## Next Steps

In **L2-1.2 -- Async and Batch Logging** you will learn how to log large volumes of data efficiently using MLflow's async logging API, which is critical when your training loop or sweep generates thousands of metrics.
