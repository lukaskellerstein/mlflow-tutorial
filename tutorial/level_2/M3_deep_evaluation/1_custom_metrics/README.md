# L2-M3.1 — Custom Metrics and Evaluators

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

Learn how to build custom evaluation scorers that go beyond MLflow's built-in options. You will create deterministic (rule-based) scorers for fast, reproducible checks and LLM-based scorers for nuanced quality judgments — then combine them in a single evaluation run with programmatic threshold gates.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L1-M6.2 (Scorers & Judges)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why custom scorers?

MLflow ships with built-in scorers like `Correctness`, `Safety`, and `ResponseLength`, but real-world applications need domain-specific quality checks. Custom scorers let you:

- Enforce formatting standards (sentence structure, length, keyword coverage)
- Judge technical accuracy against a rubric using an LLM
- Combine multiple quality signals into a single evaluation pass
- Gate deployments on minimum quality thresholds

### The @scorer decorator

The `@scorer` decorator from `mlflow.genai.scorers` wraps a plain Python function into a `Scorer` object that `mlflow.genai.evaluate()` can call. Your function receives a **subset** of these parameters — declare only what you need:

| Parameter | Description |
|-----------|-------------|
| `inputs` | The input to the model (from the `inputs` column) |
| `outputs` | The model's output (from `outputs` column or `predict_fn`) |
| `expectations` | Ground truth / expected values (from `expectations` column) |
| `trace` | The MLflow trace object for the prediction |

Your function can return:
- A **primitive** (`bool`, `int`, `float`, `str`) — simple pass/fail or numeric score
- A **`Feedback`** object — includes value, rationale, and source metadata

### Deterministic vs. LLM-based scorers

| | Deterministic | LLM-Based |
|---|---|---|
| **Speed** | Fast (milliseconds) | Slow (seconds per row) |
| **Reproducibility** | Perfectly reproducible | May vary between runs |
| **Cost** | Free | Token costs (or local GPU) |
| **Best for** | Format checks, keyword matching, length | Nuanced quality, accuracy, reasoning |

Use both in combination: deterministic scorers as fast guardrails, LLM scorers for deeper quality assessment.

## Step-by-Step

### Step 1: Deterministic scorer with Feedback

The `formatting_quality` scorer checks response quality without any LLM call:

```python
@scorer
def formatting_quality(outputs, expectations) -> Feedback:
    # Check sentence count, keyword overlap, length adequacy
    # Return a Feedback object with composite score + rationale
    return Feedback(
        value=composite_score,
        rationale="sentences=3, keyword_overlap=0.65, ...",
        source=AssessmentSource(source_type="CODE", source_id="formatting_quality"),
    )
```

Key points:
- Returns `Feedback` instead of a bare float — this gives you rationale and source metadata in the MLflow UI
- `AssessmentSource(source_type="CODE")` marks this as a deterministic/heuristic evaluation
- The composite score blends multiple signals (sentence structure, keyword overlap, length)

### Step 2: LLM-based scorer

The `llm_technical_quality` scorer uses `gemma4:e2b` as a judge:

```python
@scorer
def llm_technical_quality(inputs, outputs, expectations) -> Feedback:
    llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
    # Prompt the LLM to score accuracy, completeness, clarity
    # Parse JSON response, compute average
    return Feedback(
        value=avg_score,
        rationale="accuracy=0.9, completeness=0.8, clarity=0.85",
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gemma4:e2b"),
    )
```

Key points:
- `AssessmentSource(source_type="LLM_JUDGE")` correctly categorizes this as LLM-judged
- The scorer receives `inputs`, `outputs`, and `expectations` — MLflow passes only what the function signature declares
- JSON parsing includes fallback handling for malformed LLM outputs

### Step 3: Combined evaluation

Both scorers run together via `mlflow.genai.evaluate()`:

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=answer_question,
    scorers=[formatting_quality, llm_technical_quality],
)
```

The result object contains:
- `results.metrics` — aggregated metrics (mean by default) for each scorer
- `results.result_df` — per-row DataFrame with `<scorer>/value`, `<scorer>/rationale`, etc.

### Step 4: Threshold checking

Programmatically verify that metrics meet minimum standards:

```python
thresholds = {
    "formatting_quality/mean": 0.4,
    "llm_technical_quality/mean": 0.5,
}
for metric, min_val in thresholds.items():
    actual = results.metrics.get(metric)
    passed = actual >= min_val
```

This pattern is the foundation for CI/CD quality gates (covered in L3-M3.4).

## Running the Lesson

```bash
cd tutorial/level_2/M3_deep_evaluation/1_custom_metrics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Custom Deterministic Scorer (formatting_quality)
============================================================
  This scorer checks sentence structure, keyword overlap,
  and length adequacy — all without an LLM call.

============================================================
Part 2: Custom LLM-Based Scorer (llm_technical_quality)
============================================================
  Uses gemma4:e2b as a judge to score accuracy,
  completeness, and clarity on a 0-1 scale.

============================================================
Part 3: Combined Evaluation — all scorers together
============================================================
  Dataset: 6 questions about Python concepts
  Scorers: formatting_quality + llm_technical_quality
  Running predict_fn + scorers (this may take a minute)...

--- Aggregate Metrics ---
  formatting_quality/mean: 0.XXX
  llm_technical_quality/mean: 0.XXX

--- Per-Row Results ---
  Q1: What are Python decorators?
     Answer: Decorators are functions that...
     formatting_quality/value: 0.XXX
     llm_technical_quality/value: 0.XXX

  ...

============================================================
Part 4: Programmatic Threshold Checking
============================================================
  [PASS] formatting_quality/mean: 0.XXX (threshold: 0.4)
  [PASS] llm_technical_quality/mean: 0.XXX (threshold: 0.5)

  All quality thresholds met.
```

In the MLflow UI, navigate to the experiment "L2/M3_deep_evaluation/1_custom_metrics" to see:
- The evaluation run with all metrics logged
- Per-row scorer values and rationales in the evaluation table
- Source metadata distinguishing CODE vs LLM_JUDGE assessments

### MLflow + EvalHub Integration

EvalHub is Red Hat's evaluation control plane that stores results in MLflow experiments. Understanding MLflow's evaluation data model — scorers, Feedback objects, assessment sources, and threshold gates — is foundational for using EvalHub on OpenShift AI. The custom metrics and scorers you build here are directly compatible with EvalHub's evaluation pipelines.

## Key Takeaways

- The `@scorer` decorator turns any function into an evaluator compatible with `mlflow.genai.evaluate()`
- Scorers declare only the parameters they need (`inputs`, `outputs`, `expectations`, `trace`)
- Return `Feedback` objects for rich metadata (rationale, source type) instead of bare values
- Combine deterministic and LLM-based scorers in a single evaluation for both speed and depth
- Use `results.metrics` for programmatic threshold checks — the foundation for CI/CD quality gates
- Red Hat EvalHub builds on this same evaluation data model — custom scorers work directly with EvalHub on OpenShift AI

## Next Steps

In L2-M3.2 (RAG Evaluation), you will apply these custom scorer patterns to evaluate Retrieval-Augmented Generation pipelines, including retrieval relevance, context faithfulness, and answer grounding.
