# L1-M4.1.1 -- Evaluation Fundamentals

**Level:** Essentials
**Duration:** 35m

## Overview

Learn three complementary approaches to evaluating LLM outputs, all using one shared dataset: the `mlflow.genai.evaluate()` API with built-in and custom scorers, the manual LLM-as-Judge pattern where one LLM grades another, and custom deterministic scorer functions for fast, reproducible checks. All results are combined and logged to MLflow for comparison.

## Prerequisites

- Completed: L1-M3 (Models and Registry)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway up (`cd infra && podman compose up -d`), with LMStudio
  serving `google/gemma-4-26b-a4b` behind the `gemma-chat` alias

## Concepts

### Why Evaluate LLMs?

Unlike traditional ML models that output numbers or classes, LLMs produce free-form text. You cannot simply compute accuracy against a label. Instead, LLM evaluation uses a combination of:

- **Deterministic checks** -- length, format, keyword presence (fast, reproducible)
- **Custom scorers** -- domain-specific Python functions with the `@scorer` decorator
- **LLM-as-Judge** -- use another LLM to assess quality (nuanced, slower)

### The mlflow.genai.evaluate() API

MLflow provides `mlflow.genai.evaluate()` specifically for GenAI evaluation:

1. **data** -- a DataFrame with `inputs` (dict), optional `outputs`, and optional `expectations` (ground truth)
2. **predict_fn** -- a callable that generates outputs from inputs
3. **scorers** -- a list of scorer objects that grade each output

### Built-in Scorers

| Scorer | What it checks |
|--------|---------------|
| `ResponseLength` | Output length within min/max bounds |
| `RegexMatch` | Output matches a regex pattern |
| `PIIDetection` | Output contains PII |

### LLM-as-Judge

One LLM reads the question, expected answer, and model answer, then produces a structured score with a justification. Benefits: scalable, semantic understanding, explainable. Limitations: self-preference bias, verbosity bias, inconsistency.

### Custom Scorer Functions

Any Python function that takes input/output and returns a numeric score. Common patterns: keyword overlap, sentence structure analysis, format validation.

### Why Combine All Three?

Custom scorers catch obvious failures fast (too short, missing keywords). LLM judges assess deeper quality. Using both gives a comprehensive evaluation view.

## Step-by-Step

### Step 1: mlflow.genai.evaluate() with Scorers

Use `ResponseLength` (built-in) and `contains_expected` (custom `@scorer`) to evaluate Q&A pairs:

```python
@scorer
def contains_expected(inputs, outputs, expectations) -> bool:
    expected = expectations.get("expected_response", "")
    return expected.lower() in outputs.lower()


results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=answer_question,
    scorers=[ResponseLength(min_length=1, max_length=500, unit="words"), contains_expected],
)
```

### Step 2: Manual LLM-as-Judge

Send a structured prompt asking the LLM to compare the model's answer against ground truth and return a 1-5 score with justification in JSON:

```python
def judge_answer(question, ground_truth, model_answer) -> dict:
    prompt = JUDGE_PROMPT.format(...)
    response = client.chat.completions.create(model=..., messages=[...], temperature=0.0)
    # Parse JSON response for score and justification
```

### Step 3: Custom Deterministic Scorers

Fast, reproducible Python functions that compute keyword overlap, detail depth, and a weighted composite score without any LLM call.

### Step 4: Combined View

All scoring approaches run on the same dataset. Results are logged to MLflow as per-question metrics and averages, enabling cross-approach comparison.

## Running the Lesson

```bash
cd tutorial/level_1_models/M4_evaluation/1_fundamentals/1_evaluation_fundamentals
uv sync
uv run python main.py
```

## Expected Output

Four sections in the terminal:

1. **Part 1** -- mlflow.genai.evaluate() aggregate metrics and per-row scorer results
2. **Part 2** -- LLM-as-Judge scores (1-5) with justifications for each question
3. **Part 3** -- Deterministic scorer values (keyword overlap, detail depth, composite)
4. **Part 4** -- Comparison table showing judge, keyword, and custom scores side by side

In the MLflow UI, navigate to experiment **L1/M4_evaluation/1_fundamentals/1_evaluation_fundamentals** to see all logged metrics and evaluation runs.

## Key Takeaways

- `mlflow.genai.evaluate()` is the primary API for GenAI evaluation, built around composable scorers.
- The `@scorer` decorator lets you write custom evaluation logic in a few lines.
- LLM-as-Judge provides scalable semantic evaluation -- use structured JSON output and `temperature=0`.
- Custom deterministic scorers are fast and reproducible -- great for keyword and format checks.
- Combining all three approaches gives both speed (deterministic) and depth (LLM judge).
- All results should be logged to MLflow for tracking and comparison across runs.

## Next Steps

Continue to **L1-M4.2.1 (GenAI Custom Metrics)** to learn about `make_metric()` for custom metric functions, the `Feedback` object for rich scorer metadata, and batch evaluation across multiple LLM configurations.
