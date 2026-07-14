# L1-M4.3 — GenAI Scorers and Judges

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

When evaluating LLM outputs, you need scoring mechanisms that go beyond simple string matching. This lesson introduces two complementary approaches: **custom scorers** (deterministic Python functions) and **LLM judges** (using one LLM to grade another's output). You will build both, combine them, and log everything to MLflow.

## Prerequisites

- Completed: L1-M4.2 (LLM-as-Judge)
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Scorers vs. Judges

| Aspect | Custom Scorer | LLM Judge |
|--------|---------------|-----------|
| **How it works** | Deterministic Python function | An LLM evaluates the output |
| **Speed** | Fast (milliseconds) | Slow (seconds per call) |
| **Reproducibility** | Perfectly reproducible | May vary between runs |
| **Cost** | Free | Token cost per evaluation |
| **Best for** | Length, format, keyword presence | Relevance, coherence, nuance |

### Custom Scorers

A custom scorer is any Python function that takes an input/output pair and returns a numeric score. Common patterns:

- **Length checks** -- is the response appropriately detailed?
- **Keyword overlap** -- does it cover the key terms from the expected answer?
- **Format validation** -- does it follow the required structure?
- **Regex matching** -- does it contain required patterns?

Custom scorers are ideal for fast, cheap, repeatable checks.

### LLM Judges

An LLM judge uses a language model to evaluate another model's output. The judge receives a structured prompt with the question, expected answer, and actual response, then scores on criteria like:

- **Relevance** -- does the response address the question?
- **Completeness** -- does it cover all important aspects?
- **Clarity** -- is it well-written and easy to understand?

LLM judges capture nuance that deterministic scorers miss, but they are slower and non-deterministic.

### Why Combine Both?

In practice, you want both. Custom scorers catch obvious failures fast (too short, missing keywords), while LLM judges assess deeper quality. Logging both to MLflow lets you track which scoring approach best predicts real-world quality.

## Step-by-Step

### Step 1: Build a Custom Scorer

The custom scorer evaluates three aspects with simple heuristics:

```python
def custom_scorer(question, expected, response):
    len_ratio = min(len(response) / max(len(expected), 1), 1.5) / 1.5
    # ... keyword overlap and detail depth ...
    composite = 0.4 * len_ratio + 0.4 * overlap + 0.2 * detail
    return {"custom_composite": composite, ...}
```

Each sub-score ranges from 0 to 1, and the composite is a weighted average.

### Step 2: Build an LLM Judge

The LLM judge sends a structured prompt to `google/gemma-4-e4b` asking it to score on relevance, completeness, and clarity:

```python
def llm_judge(llm, question, expected, response):
    prompt = JUDGE_PROMPT.format(...)
    raw = llm.invoke(prompt).content
    scores = json.loads(raw)
    return {"judge_relevance": scores["relevance"], ...}
```

The judge returns JSON with scores and a brief justification.

### Step 3: Combine and Log to MLflow

Both scorers run on the same Q&A pairs. All scores are logged as MLflow metrics -- per-question and averaged -- enabling comparison across evaluation runs.

```python
with mlflow.start_run(run_name="combined_scorer_judge"):
    for i, qa in enumerate(QA_PAIRS):
        cs = custom_scorer(...)
        js = llm_judge(...)
        mlflow.log_metric(f"q{i+1}_custom_composite", cs["custom_composite"])
        mlflow.log_metric(f"q{i+1}_judge_relevance", js["judge_relevance"])
```

## Running the Lesson

```bash
cd tutorial/level_1/M4_evaluations/3_scorers_judges
uv sync
uv run python main.py
```

## Expected Output

The script prints three sections:

1. **Custom Scorer** -- deterministic scores for each Q&A pair
2. **LLM Judge** -- LLM-generated scores with justifications
3. **Comparison table** -- side-by-side view of custom vs. judge scores

You should see that short, incomplete answers (like "Tides happen because of the Moon") score low on both scorers, while detailed answers score high. The comparison table makes patterns easy to spot.

In the MLflow UI, the run will contain metrics like `avg_custom_composite`, `avg_judge_relevance`, `avg_judge_completeness`, and `avg_judge_clarity`.

## Key Takeaways

- **Custom scorers** are fast, reproducible Python functions -- great for format and coverage checks.
- **LLM judges** use a language model to assess nuanced quality -- slower but more flexible.
- **Combining both** gives you a comprehensive evaluation view: fast checks plus deep assessment.
- **Log everything to MLflow** so you can compare evaluation approaches across runs.
- Judge prompts should request structured output (JSON) for reliable parsing.

## Next Steps

Continue to L1-M4.4 (Datasets and Labeling) to learn how to manage evaluation datasets and capture human feedback. In Level 2, we will explore this in more depth with custom metrics, RAG evaluation, and human-in-the-loop workflows.
