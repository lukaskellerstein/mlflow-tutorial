# L1-M4.2 — LLM-as-Judge Evaluation

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

LLM-as-Judge is an evaluation technique where one LLM acts as an automated evaluator ("judge") to score the outputs of another LLM ("student"). This lesson demonstrates the pattern end-to-end: generate answers, judge them, and log everything to MLflow for analysis.

## Prerequisites

- Completed: L1-M4.1 (LLM Evaluation Basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why LLM-as-Judge?

Evaluating LLM outputs is hard because there is often no single "correct" answer. Traditional metrics like BLEU or ROUGE measure surface-level text overlap but miss semantic correctness. Human evaluation is accurate but slow and expensive.

LLM-as-Judge bridges the gap: a capable LLM reads the question, the expected answer, and the model's answer, then produces a structured score with a justification. This gives you:

- **Scalable evaluation** — judge thousands of outputs without human reviewers
- **Semantic understanding** — the judge understands meaning, not just word overlap
- **Explainability** — the justification tells you *why* a score was given

### The Pattern

1. Prepare an **evaluation dataset** with questions and ground-truth answers.
2. Run the **student model** to generate answers.
3. For each answer, ask the **judge model** to compare it against the ground truth and produce a score.
4. Parse the structured output (JSON with score + justification).
5. Log scores and artifacts to **MLflow** for tracking and comparison.

### Known Limitations and Biases

LLM-as-Judge is powerful but imperfect. Be aware of these issues:

- **Self-preference bias** — LLMs tend to rate their own outputs higher. When possible, use a different (ideally stronger) model as the judge.
- **Position bias** — judges may favor the first or last answer in a comparison. Randomize presentation order when comparing multiple models.
- **Verbosity bias** — longer answers often receive higher scores even if shorter answers are equally correct.
- **Inconsistency** — the same judge can give different scores on repeated evaluations. Use `temperature=0` for the judge to reduce variance.
- **Prompt sensitivity** — small changes to the judge prompt can shift scores significantly. Standardize and version your judge prompts.

In this lesson we use the same small model (`google/gemma-4-e4b`) for both student and judge to keep things fast. In production you would typically use a larger, more capable model as the judge (e.g., `google/gemma-4-26b-a4b` or a frontier model).

## Step-by-Step

### Step 1: Define the Evaluation Dataset

We create a small set of question/ground-truth pairs covering factual knowledge:

```python
EVAL_DATA = [
    {
        "question": "What is the capital of France?",
        "ground_truth": "The capital of France is Paris.",
    },
    # ... more pairs
]
```

### Step 2: Generate Answers with the Student Model

The student model receives each question and produces an answer using the OpenAI client connected to LMStudio:

```python
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def generate_answer(question: str) -> str:
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": f"Answer concisely: {question}"}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
```

We use a low temperature (0.3) for more consistent answers.

### Step 3: Judge Each Answer

The judge uses the same client but with `temperature=0.0` for maximum consistency. It receives a structured prompt with the question, ground truth, and model answer, and returns a JSON object with a 1-5 score and justification:

```python
def judge_answer(question: str, ground_truth: str, model_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    # ... parse JSON from response
```

### Step 4: Log Results to MLflow

We log per-question scores as individual metrics and the full results table as an artifact:

```python
with mlflow.start_run(run_name="llm_as_judge_eval"):
    mlflow.log_param("student_model", "google/gemma-4-e4b")
    mlflow.log_param("judge_model", "google/gemma-4-e4b")
    mlflow.log_metric("avg_score", avg_score)
    mlflow.log_table(df, artifact_file="evaluation_results.json")
```

## Running the Lesson

```bash
cd tutorial/level_1/M4_evaluations/2_llm_as_judge
uv sync
uv run python main.py
```

## Expected Output

You should see output similar to:

```
============================================================
LLM-as-Judge Evaluation
============================================================

--- Q1: What is the capital of France?
  Student answer : The capital of France is Paris.
  Judge score    : 5/5
  Justification  : The answer is fully correct and matches the ground truth.

--- Q2: What is photosynthesis?
  Student answer : Photosynthesis is the process by which plants use sunlight...
  Judge score    : 4/5
  Justification  : Mostly correct but omits the role of carbon dioxide.

...

============================================================
Average judge score: 4.25 / 5
============================================================

Results logged to MLflow.
```

In the MLflow UI, navigate to experiment `L1/M4_evaluations/2_llm_as_judge` to see:
- Per-question scores (`q1_score`, `q2_score`, etc.)
- The average score metric
- The full evaluation table artifact

## Key Takeaways

- **LLM-as-Judge** uses one LLM to evaluate another, providing scalable semantic evaluation.
- Always use **structured output** (JSON) from the judge for reliable parsing.
- Set the judge's **temperature to 0** for consistent scoring.
- Log all evaluation data to **MLflow** so you can compare runs across model changes.
- Be aware of **biases** (self-preference, verbosity, position) when interpreting scores.

## Next Steps

In Level 2 (M3 — Deep Evaluation), we will build custom metrics, evaluate RAG pipelines, and use MLflow's built-in GenAI evaluation framework with LLM judges at scale.
