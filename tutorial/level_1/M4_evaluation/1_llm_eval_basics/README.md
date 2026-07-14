# L1-M4.2 — LLM Evaluation Basics

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

Learn how to evaluate LLM outputs programmatically using `mlflow.genai.evaluate()`. You will build a small Q&A evaluation dataset, run a local LLM against it, and score the results with both built-in and custom scorers. This is the foundation for all GenAI evaluation work in MLflow.

## Prerequisites

- Completed: L1-M4.1 (Traditional ML Evaluation)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running locally with `google/gemma-4-e4b` model loaded

## Concepts

### Why evaluate LLMs?

Unlike traditional ML models that output numbers or classes, LLMs produce free-form text. You cannot simply compute accuracy against a label. Instead, LLM evaluation uses a combination of:

- **Deterministic checks** — does the output meet length, format, or keyword requirements?
- **Custom scorers** — does the output contain the expected answer?
- **LLM-as-judge** — use another LLM to assess quality (covered in L1-M4.3).

### The mlflow.genai.evaluate() API

MLflow provides `mlflow.genai.evaluate()` specifically for GenAI evaluation. It takes:

1. **data** — a DataFrame with `inputs` (dict), optional `outputs`, and optional `expectations` (ground truth) columns.
2. **predict_fn** — a callable that generates outputs from inputs. If provided, MLflow calls it for each row.
3. **scorers** — a list of scorer objects that grade each output.

The function returns an `EvaluationResult` with aggregate metrics and a per-row results table.

### Built-in Scorers

MLflow ships with deterministic scorers that require no LLM:

| Scorer | What it checks |
|--------|---------------|
| `ResponseLength` | Output length is within min/max bounds (chars or words) |
| `RegexMatch` | Output matches a regex pattern |
| `PIIDetection` | Output contains personally identifiable information |

### Custom Scorers

Use the `@scorer` decorator to define your own evaluation logic:

```python
from mlflow.genai.scorers import scorer

@scorer
def my_check(inputs, outputs, expectations) -> bool:
    return expectations["answer"] in outputs
```

The function receives `inputs`, `outputs`, and `expectations` from the dataset and returns a boolean (pass/fail) or a `Feedback` object for richer results.

## Step-by-Step

### Step 1: Create the evaluation dataset

We build a pandas DataFrame with factual Q&A pairs. Each row has an `inputs` dict (the question) and an `expectations` dict (the ground truth answer).

```python
eval_data = pd.DataFrame([
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    ...
])
```

### Step 2: Define a predict function

The predict function takes keyword arguments matching the keys inside `inputs` and returns a string:

```python
def answer_question(question: str) -> str:
    response = llm.invoke(question)
    return response.content
```

### Step 3: Choose scorers

We use one built-in scorer and one custom scorer:

```python
ResponseLength(min_length=1, max_length=500, unit="words")

@scorer
def contains_expected(inputs, outputs, expectations) -> bool:
    expected = expectations.get("expected_response", "")
    return expected.lower() in outputs.lower()
```

### Step 4: Run the evaluation

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=answer_question,
    scorers=[ResponseLength(...), contains_expected],
)
```

MLflow will call `answer_question` for each row, then run every scorer against the output. Results are logged to the active MLflow experiment automatically.

## Running the Lesson

```bash
cd tutorial/level_1/M4_evaluation/2_llm_eval_basics
uv sync
uv run python main.py
```

## Expected Output

You should see console output showing:

1. The 5-question evaluation dataset
2. Progress as the LLM answers each question
3. Aggregate metrics (pass rates for each scorer)
4. A per-row results table showing each question, the LLM's answer, and scorer verdicts

In the MLflow UI, navigate to experiment **L1/M4_evaluation/2_llm_eval_basics** to see:
- The evaluation run with logged metrics
- A results table with per-row scores
- Traces for each LLM call

## Key Takeaways

- `mlflow.genai.evaluate()` is the primary API for evaluating LLM and GenAI applications.
- The evaluation dataset uses `inputs`, `outputs`, and `expectations` columns (all dicts).
- Built-in scorers like `ResponseLength` handle common checks without an LLM.
- The `@scorer` decorator lets you write custom evaluation logic in a few lines.
- All results are automatically logged to MLflow for tracking and comparison.

## Next Steps

In L1-M4.3 (LLM as Judge), you will use an LLM to judge the quality of another LLM's outputs using built-in scorers like `Correctness` and `RelevanceToQuery`. In Level 2, we will explore custom metrics, RAG evaluation, and human-in-the-loop assessment.
