# L2-3.3 — GenAI Evaluation Framework

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

This lesson demonstrates the full MLflow GenAI evaluation framework. You will use `mlflow.genai.evaluate()` with both built-in and custom scorers to evaluate LLM outputs, then run batch evaluations across multiple configurations to find the best setup for a given task.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L1-M4.3 (LLM-as-Judge), L2-M3.1 (Custom Metrics), L2-M3.2 (RAG Evaluation)
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### The GenAI Evaluation Framework

MLflow's GenAI evaluation framework (`mlflow.genai.evaluate()`) is a unified API for assessing LLM and agent outputs. Unlike the older `mlflow.evaluate()` API, the GenAI framework is built around **scorers** — modular evaluation components that each assess one quality dimension.

### Scorers

A scorer takes evaluation data (inputs, outputs, expectations, traces) and produces a score. MLflow provides two categories:

**Built-in scorers** are ready to use:
- `ResponseLength` — checks output length is within bounds (deterministic)
- `RegexMatch` — matches output against a regex pattern (deterministic)
- `PIIDetection` — detects PII in outputs (deterministic)
- `Guidelines` — checks adherence to specified guidelines (LLM judge)
- `Correctness` — compares output against expected response (LLM judge)
- `Completeness` — checks if all parts of the question are addressed (LLM judge)
- `Fluency` — evaluates linguistic quality (LLM judge)
- `Safety` — detects harmful content (LLM judge)
- `RelevanceToQuery` — checks response relevance (LLM judge)

**Custom scorers** use the `@scorer` decorator to define domain-specific evaluation logic. They can return `bool`, `int`, `float`, `str`, or `Feedback` objects (which include a rationale).

### Batch Evaluation

By running `mlflow.genai.evaluate()` inside different MLflow runs — each with different LLM configurations — you can systematically compare models, temperatures, prompts, or any other variable across the same dataset and scorers.

## Step-by-Step

### Step 1: Create the Evaluation Dataset

The dataset is a list of dictionaries with `inputs`, and `expectations` fields. Each `inputs` must be a dict. The `expectations` field holds ground truth for comparison scorers.

```python
EVAL_DATA = [
    {
        "inputs": {"question": "What is Python's GIL?"},
        "expectations": {
            "expected_response": "The Global Interpreter Lock (GIL) is a mutex..."
        },
    },
    # ... more Q&A pairs
]
```

When a `predict_fn` is provided to `mlflow.genai.evaluate()`, the framework calls it with each row's `inputs` as keyword arguments and captures the output automatically.

### Step 2: Use Built-in Scorers

Built-in scorers are instantiated with configuration parameters:

```python
from mlflow.genai.scorers import ResponseLength

length_scorer = ResponseLength(min_length=20, max_length=500, unit="words")
```

Pass them as a list to `mlflow.genai.evaluate()`.

### Step 3: Create Custom Scorers

Use the `@scorer` decorator. The function's parameter names tell MLflow what data to inject (`inputs`, `outputs`, `expectations`, `trace`):

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer(name="keyword_coverage")
def keyword_coverage(inputs, outputs) -> Feedback:
    # ... evaluation logic ...
    return Feedback(value=0.85, rationale="5/6 keywords found")

@scorer(name="has_example")
def has_example(outputs) -> bool:
    return "for example" in str(outputs).lower()
```

Returning `Feedback` lets you include a rationale alongside the score. Returning `bool` gives a simple pass/fail.

### Step 4: Batch Evaluate Across Configurations

For each configuration, build a predict function and run evaluation inside its own MLflow run:

```python
result = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=predict_fn,
    scorers=[length_scorer, keyword_coverage, has_example],
)
```

The `EvaluationResult` object provides:
- `result.metrics` — aggregated scores (dict of metric name to float)
- `result.result_df` — per-row results with individual scores and rationales

### Step 5: Compare Results

Collect metrics from each run and compare side-by-side to identify the best configuration per metric.

## Running the Lesson

```bash
cd tutorial/level_2/M3_deep_evaluation/3_genai_evaluation
uv sync
uv run python main.py
```

## Expected Output

The script prints:
1. The evaluation dataset (6 questions)
2. The list of scorers being used
3. Per-configuration evaluation results with aggregate metrics
4. A comparison table showing all configs side-by-side with the winner per metric

In the MLflow UI at http://127.0.0.1:5000, you will see:
- Experiment `L2/M3_deep_evaluation/3_genai_evaluation`
- One run per configuration, each with logged parameters and metrics
- Traces showing each LLM call and its evaluation scores

## Key Takeaways

- `mlflow.genai.evaluate()` is the primary API for GenAI evaluation, built around composable scorers
- Built-in scorers cover common needs (length, regex, PII, safety, correctness, fluency)
- The `@scorer` decorator lets you create custom scorers that return `bool`, `float`, or `Feedback` with rationale
- Scorer functions declare what data they need via parameter names (`inputs`, `outputs`, `expectations`, `trace`)
- Batch evaluation across configurations with the same dataset and scorers enables systematic comparison

## Next Steps

Continue to L2-M3.4 (Human-in-the-Loop Evaluation) to learn how to incorporate human feedback into the evaluation loop using `mlflow.log_assessment()` and the labeling UI.
