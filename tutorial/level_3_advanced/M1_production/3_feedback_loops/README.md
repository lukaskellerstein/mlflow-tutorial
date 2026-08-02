# L3-3.3 -- Production Feedback Loops

**Level:** Expert
**Duration:** ~1.5 hours

## Overview

Production LLM systems need continuous improvement driven by real user feedback. This lesson builds a complete feedback loop: collecting structured user feedback on LLM responses, associating it with MLflow traces via the assessment API, analyzing feedback to find weak spots, and iterating on prompts based on what the data shows.

## Prerequisites

- Completed: L3-3.1 (Production Tracing), L3-3.2 (Grafana Dashboards)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Why Feedback Loops Matter

A deployed LLM application is never "done." Users interact with it in ways you cannot fully anticipate during development. Feedback loops close the gap between what you tested and what users actually experience.

The cycle looks like this:

1. **Deploy** a prompt/model version
2. **Collect** structured feedback on each response (thumbs up/down, ratings, comments)
3. **Analyze** aggregate metrics -- average satisfaction, common failure modes
4. **Identify** questions or topics where the model underperforms
5. **Improve** the prompt, retrieval strategy, or model, guided by the data
6. **Redeploy** and repeat

### MLflow Feedback API

MLflow provides `mlflow.log_feedback()` to attach feedback directly to traces. Each feedback record includes:

- **name** -- the feedback dimension (e.g., `"user_rating"`, `"thumbs_up"`)
- **value** -- the feedback value (bool, int, float, str, dict)
- **source** -- who provided the feedback (human, code, LLM judge)
- **rationale** -- free-text justification
- **metadata** -- additional context (question text, session ID, etc.)

This is far superior to logging feedback as run tags or metrics, because it ties feedback to the specific trace (and therefore the exact LLM call) that produced the response.

### Feedback Types

This lesson implements three complementary feedback signals:

| Type | Value | Use Case |
|------|-------|----------|
| Thumbs up/down | `bool` | Quick binary signal, high volume |
| Rating (1-5) | `int` | Finer-grained quality signal |
| Text comment | `str` | Root cause analysis, qualitative insights |

In production, you would collect these from a UI widget, API callback, or review queue.

## Step-by-Step

### Step 1: Feedback Collection System

The `FeedbackCollector` class simulates user feedback with configurable sentiment distributions. In production, this would be replaced by real user input.

```python
collector = FeedbackCollector()
fb = collector.simulate_feedback(trace_id, question, response, seed=42)
# Returns: UserFeedback(thumbs_up=True, rating=4, comment="Great explanation!")
```

### Step 2: Logging Feedback to MLflow

Each response gets three feedback assessments attached to its trace:

```python
mlflow.log_feedback(
    trace_id=fb.trace_id,
    name="user_rating",
    value=fb.rating,
    source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="simulated_user"),
    rationale=fb.comment,
)
```

### Step 3: Generate Responses with Feedback Tracking

We generate 8 LLM responses per iteration, each auto-traced by `mlflow.langchain.autolog()`. After each response, we capture the trace ID and attach simulated feedback.

### Step 4: Feedback Analysis

Aggregate metrics are computed from all feedback records:

- **Average rating** -- overall quality signal
- **Satisfaction rate** -- proportion of thumbs-up responses
- **Issue categorization** -- classify negative comments (vague, incorrect, off-topic, confusing)
- **Low-rated questions** -- specific inputs that need prompt improvement

### Step 5: Feedback-Driven Improvement

The lesson runs two iterations with different system prompts:

- **v1 (baseline):** `"Answer the question concisely."`
- **v2 (improved):** `"Answer clearly and completely. Provide specific examples..."`

The improved prompt is designed to address the most common negative feedback (vague, lacking detail). Cross-iteration comparison shows whether the change helped.

## Running the Lesson

```bash
cd tutorial/level_3/M3_production/3_feedback_loops
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-3.3 -- Production Feedback Loops
============================================================

--- Iteration 1: Baseline prompt (v1) ---
    [1/8] What is machine learning?...
    ...

============================================================
  Iteration 1 -- Baseline
============================================================
  Total responses:    8
  Average rating:     3.5/5
  Satisfaction rate:  62%
  ...

--- Iteration 2: Improved prompt (v2) ---
    ...

============================================================
  Cross-Iteration Comparison
============================================================
  Avg rating:        3.5 -> 3.75  (+0.25)
  Satisfaction rate:  62% -> 75%  (+13%)
  Low-rated:         2 -> 1
```

In the MLflow UI:
- Navigate to the experiment `L3/M3_production/3_feedback_loops`
- Open any trace to see attached feedback assessments (thumbs_up, user_rating, user_comment)
- Compare the two nested runs (iteration_1_baseline vs iteration_2_improved) side by side
- Check the `feedback/summary_table.json` artifact for the full feedback dataset

## Key Takeaways

- **`mlflow.log_feedback()`** attaches structured feedback directly to traces, linking user signals to specific LLM calls
- **Three feedback types** (binary, rating, text) give you both volume and depth -- binary for dashboards, ratings for trends, text for root cause analysis
- **Feedback analysis** should focus on actionable insights: which questions fail, what types of issues recur, and what prompt changes to try
- **Iterative improvement** is the core loop: deploy, measure, identify weak spots, change the prompt or model, re-measure
- **AssessmentSource** metadata lets you distinguish human feedback from automated scores, which matters when you mix sources in production

## Next Steps

Continue to L3-3.4 (CI/CD Quality Gates) to learn how to automate quality checks and prevent regressions when deploying prompt or model changes.
