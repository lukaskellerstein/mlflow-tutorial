# L1-M4.2.1 -- GenAI Custom Metrics

**Level:** Essentials
**Duration:** 40m

## Overview

Build custom evaluation scorers that go beyond MLflow's built-in options, then use them in batch evaluation across multiple LLM configurations. This lesson covers deterministic scorers with `Feedback` objects, LLM-based scorers, and programmatic threshold gates -- the foundation for CI/CD quality controls.

## Prerequisites

- Completed: L1-M4.1.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway up (`cd infra && podman compose up -d`), with LMStudio
  serving `google/gemma-4-26b-a4b` behind the `gemma-chat` alias

## Concepts

### The @scorer Decorator

The `@scorer` decorator from `mlflow.genai.scorers` wraps a Python function into a `Scorer` object compatible with `mlflow.genai.evaluate()`. Your function receives a subset of these parameters -- declare only what you need:

| Parameter | Description |
|-----------|-------------|
| `inputs` | The input to the model (from the `inputs` column) |
| `outputs` | The model's output (from `outputs` column or `predict_fn`) |
| `expectations` | Ground truth / expected values |
| `trace` | The MLflow trace object for the prediction |

### Return Types

Your scorer function can return:
- **`bool`** -- simple pass/fail
- **`int` / `float`** -- numeric score
- **`Feedback`** -- includes value, rationale, and source metadata

### Feedback Objects

`Feedback` provides rich metadata alongside the score:

```python
from mlflow.entities import AssessmentSource, Feedback

return Feedback(
    value=0.85,
    rationale="5/6 keywords found in answer",
    source=AssessmentSource(source_type="CODE", source_id="my_scorer"),
)
```

`source_type` can be `"CODE"` (deterministic) or `"LLM_JUDGE"` (LLM-based). This metadata appears in the MLflow UI.

### Batch Evaluation

By running `mlflow.genai.evaluate()` inside different MLflow runs -- each with different LLM parameters -- you can systematically compare models, temperatures, or prompts across the same dataset and scorers.

### Threshold Gates

Programmatically check that aggregate metrics meet minimum standards. This pattern is the foundation for CI/CD quality gates.

## Step-by-Step

### Step 1: Deterministic Scorer with Feedback

The `formatting_quality` scorer checks sentence structure, keyword overlap, and length -- all without an LLM call:

```python
@scorer
def formatting_quality(outputs, expectations) -> Feedback:
    # ... compute composite score from sentence count, keyword overlap, length
    return Feedback(
        value=composite_score,
        rationale="sentences=3, keyword_overlap=0.65, word_count=45",
        source=AssessmentSource(source_type="CODE", source_id="formatting_quality"),
    )
```

### Step 2: LLM-based Scorer with Feedback

The `llm_technical_quality` scorer uses the LLM as a judge to score accuracy, completeness, and clarity:

```python
@scorer
def llm_technical_quality(inputs, outputs, expectations) -> Feedback:
    # ... prompt the LLM to score on three criteria
    return Feedback(
        value=avg_score,
        rationale="accuracy=0.9, completeness=0.8, clarity=0.85",
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gemma-chat"),
    )
```

### Step 3: Additional Custom Scorers

- `keyword_coverage` -- checks question keywords in the answer
- `answer_conciseness` -- scores word count in the 30-150 word sweet spot
- `has_example` -- detects code snippets or example indicators

### Step 4: Batch Evaluation

Two configurations (temperature=0.3 and temperature=0.9) are evaluated with all scorers:

```python
result = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=predict_fn,
    scorers=ALL_SCORERS,
)
```

### Step 5: Comparison and Threshold Gates

Compare metrics side by side and check against minimum thresholds:

```python
thresholds = {"formatting_quality/mean": 0.4, "llm_technical_quality/mean": 0.5}
for metric, min_val in thresholds.items():
    actual = results.metrics.get(metric)
    passed = actual >= min_val
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M4_evaluation/2_offline/1_genai_custom_metrics
uv sync
uv run python main.py
```

## Expected Output

The script prints:

1. The 5-question evaluation dataset and 6 scorers
2. Scorer definitions (Parts 1-3)
3. Per-configuration evaluation results with aggregate metrics (Part 4)
4. A comparison table with the winner per metric, plus threshold gate results (Part 5)

In the MLflow UI, navigate to experiment **L1/M4_evaluation/2_offline/1_genai_custom_metrics** to see one run per configuration with all metrics, scorer values, and rationales.

## Key Takeaways

- The `@scorer` decorator turns any function into an evaluator compatible with `mlflow.genai.evaluate()`.
- Scorers declare only the parameters they need (`inputs`, `outputs`, `expectations`, `trace`).
- Return `Feedback` objects for rich metadata (rationale, source type) instead of bare values.
- `AssessmentSource` distinguishes `CODE` (deterministic) from `LLM_JUDGE` (LLM-based) evaluations.
- Combine deterministic and LLM-based scorers in a single evaluation for both speed and depth.
- Batch evaluation across configurations with the same dataset enables systematic comparison.
- Use `results.metrics` for programmatic threshold checks -- the foundation for CI/CD quality gates.

## Next Steps

Continue to **L1-M5 (Prompt Engineering)** to learn about MLflow's prompt registry and prompt optimization features. In Level 2, you will explore RAG evaluation, human-in-the-loop assessment, and agent evaluation.
