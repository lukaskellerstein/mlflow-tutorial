# L2-M3.4 — Human-in-the-Loop Evaluation

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

Automated evaluation is fast and cheap, but it is not always reliable — especially for subjective quality, nuanced correctness, or novel domains where the LLM judge itself may be wrong. Human-in-the-loop evaluation solves this by combining automated pre-screening with targeted human review. This lesson shows how to build that workflow using MLflow's Assessment API.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L1-M4.3 (LLM-as-Judge), L2-M3.1 (Custom Metrics)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why Human-in-the-Loop?

Automated LLM judges can score thousands of examples per hour, but they have blind spots: they may miss subtle factual errors, misunderstand domain-specific requirements, or disagree with human preferences. Human review is expensive but high-quality. The optimal approach is a hybrid:

1. **Auto-judge** scores all examples quickly
2. **Triage** separates clear passes/failures from borderline cases
3. **Human reviewers** focus only on borderline cases (typically 10-30% of the total)
4. **Feedback loop** — human corrections feed back to improve the dataset and the judge

### MLflow Assessment API

MLflow provides first-class support for attaching assessments to traces:

- **`mlflow.log_feedback()`** — Record a judgment (human, LLM, or code) on a trace. Includes a value, rationale, and source metadata.
- **`mlflow.log_expectation()`** — Record the expected/ground-truth answer on a trace.
- **`mlflow.override_feedback()`** — Let a human correct an automated judgment while preserving the original for audit and judge fine-tuning.
- **`AssessmentSource`** — Tracks who/what produced the assessment (HUMAN, LLM_JUDGE, or CODE).

These assessments are stored on the trace itself, making them visible in the MLflow UI alongside the full execution trace.

### Feedback Loops

Over multiple evaluation rounds:
1. Generate model outputs and auto-score them
2. Human reviewers correct mistakes in the auto-scores
3. Corrected examples are added to the evaluation dataset
4. The dataset grows in coverage, catching more edge cases over time

## Step-by-Step

### Step 1: Generate Model Outputs

Create traced LLM calls for a set of Q&A pairs. Each call produces a trace with a unique `trace_id` that we can attach assessments to later.

```python
@mlflow.trace(name=f"qa_pair_{i}")
def traced_qa(question: str, expected: str) -> dict:
    answer = generate_answer(question)
    return {"question": question, "expected": expected, "answer": answer}

result = traced_qa(qa["question"], qa["expected"])
trace_id = mlflow.get_last_active_trace_id()
```

### Step 2: Attach Human Assessments

Use `log_expectation()` for ground truth and `log_feedback()` for human judgments:

```python
human_source = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN,
    source_id="reviewer@example.com",
)

mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_answer",
    value="Paris",
    source=human_source,
)

mlflow.log_feedback(
    trace_id=trace_id,
    name="human_correctness",
    value="correct",
    source=human_source,
    rationale="Exact match with expected answer.",
    metadata={"confidence": "0.95"},
)
```

### Step 3: Combined Auto + Human Evaluation

Auto-judge scores all examples, then triage by score:
- Score >= 4: **auto-approved** (no human needed)
- Score <= 2: **auto-rejected** (no human needed)
- Score 3: **borderline** — routed to human review

For borderline cases, humans override the automated score:

```python
mlflow.override_feedback(
    trace_id=trace_id,
    assessment_id=auto_feedback.assessment_id,
    value=4,  # human correction
    rationale="Human override: marked as correct",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="senior_reviewer@example.com",
    ),
)
```

### Step 4: Feedback Loop

Over multiple rounds, human-reviewed examples expand the evaluation dataset. Each round:
1. Runs the model on the current dataset
2. Scores with the auto-judge
3. Adds new examples from human corrections
4. Tracks dataset size and accuracy over rounds using nested MLflow runs

## Running the Lesson

```bash
cd tutorial/level_2/M3_deep_evaluation/4_human_in_loop
uv sync
uv run python main.py
```

## Expected Output

```
PART 1: Generate Model Outputs for Review
  Q1: What is the capital of France?
      Answer: The capital of France is Paris...
      Trace:  tr-abc123...
  ...
  Logged 5 Q&A outputs as table artifact

PART 2: Simulate Human Assessments
  Q1: label=correct, confidence=0.95, notes=Exact match with expected answer.
  ...
  Assessment summary: {'correct': 4, 'partial': 1}
  Average confidence: 0.85

PART 3: Combined Automated + Human Evaluation
  Workflow: auto-judge -> flag borderline -> human review -> final verdict
  Q1: auto_score=5, verdict=AUTO_APPROVED
  ...
  Triage summary (5 items):
    Auto-approved:     3
    Auto-rejected:     0
    Human review:      2
    Human review rate: 40%

PART 4: Feedback Loop — Growing the Evaluation Dataset
  --- Evaluation Round 1 ---
  Dataset size: 5, Accuracy: 80% (4/5)
  --- Evaluation Round 2 ---
  Added new example: 'What is the boiling point of water in Celsius?'
  Dataset size: 6, Accuracy: 83% (5/6)
  --- Evaluation Round 3 ---
  Added new example: 'Who wrote Romeo and Juliet?'
  Dataset size: 7, Accuracy: 86% (6/7)
```

In the MLflow UI, navigate to the experiment and:
- Click on any trace to see attached Feedback and Expectation assessments
- See the overridden feedback for borderline cases (original preserved, new assessment linked)
- Compare accuracy across nested runs in Part 4

## Key Takeaways

- **`mlflow.log_feedback()`** and **`mlflow.log_expectation()`** attach structured assessments directly to traces, visible in the MLflow UI
- **`mlflow.override_feedback()`** preserves the original automated judgment while recording the human correction — critical for auditing and judge improvement
- **Triage by auto-score** reduces human review to borderline cases only, cutting costs by 60-90%
- **Feedback loops** grow evaluation datasets over time, improving coverage of edge cases and domain-specific scenarios
- The `AssessmentSource` metadata tracks provenance: who reviewed what, and whether the source was human, LLM, or code

## Next Steps

In Level 3, M1 (Agent Evaluation), you will apply these human-in-the-loop patterns to evaluate complex AI agents — where human judgment is especially important for assessing reasoning quality, tool selection, and multi-step task completion.
