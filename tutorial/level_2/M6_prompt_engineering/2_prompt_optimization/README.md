# L2-6.2 — Prompt Optimization

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

Prompt engineering is often an iterative, trial-and-error process. This lesson turns it into a systematic optimization loop tracked by MLflow. You will define an evaluation dataset, build a custom scorer, iterate through prompt variations, and use MLflow to identify which prompt performs best and why.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L2-M6.1 (Prompt Management)
- MLFlow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why Optimize Prompts Systematically?

Ad-hoc prompt tweaking is unreliable. Without measurement, you cannot tell whether a change actually improved performance or just happened to work on the example you tested. Systematic optimization means:

1. **Fixed evaluation set** — the same questions every time, so results are comparable.
2. **Quantitative scoring** — numeric metrics rather than subjective "looks good."
3. **Full history** — every variant is logged, so you can trace the improvement trajectory and revert if needed.

### Optimization Dimensions

There are several axes along which a prompt can be improved:

- **Instruction clarity** — vague vs. specific instructions (e.g., "answer briefly" vs. "answer in 1-3 words").
- **Role assignment** — giving the LLM a persona ("You are a geography expert") can improve confidence and accuracy.
- **Constraints** — explicit rules that shape output format and length.
- **Few-shot examples** — showing the model what good answers look like.

This lesson explores all four, measuring the impact of each.

### Scoring

The custom scorer evaluates three dimensions:

| Sub-score    | Weight | Measures                                     |
|-------------|--------|----------------------------------------------|
| exact_match | 50%    | Does the expected answer appear in the output? |
| brevity     | 30%    | Is the answer short and direct?               |
| confidence  | 20%    | Does the answer avoid hedging language?        |

The composite score is the weighted sum.

## Step-by-Step

### Step 1: Define the Optimization Problem

We create five geography Q&A pairs as our fixed evaluation dataset and a scoring function that checks correctness, brevity, and confidence. Every prompt variant will be evaluated against the same dataset with the same scorer.

```python
EVAL_DATA = [
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "What is the largest ocean on Earth?", "expected": "Pacific Ocean"},
    # ... 5 total
]
```

### Step 2: Manual Optimization Loop (Instruction Variants)

We define four prompt variants with progressively more specific instructions:

1. **baseline** — minimal instruction ("Answer the following geography question.")
2. **concise_instruction** — asks for brevity explicitly
3. **role_assignment** — assigns an expert persona and asks for confidence
4. **structured_constraints** — numbered rules constraining output format

Each variant is evaluated on all five questions, and the results are logged as a nested MLflow run with metrics, the prompt text, and per-question results.

### Step 3: Few-Shot Example Optimization

Taking the best instruction variant from Step 2 as a base, we test adding 0, 1, 2, and 3 few-shot examples. This isolates the effect of examples from instruction quality.

```python
FEW_SHOT_EXAMPLES = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the largest desert in the world?", "answer": "Sahara Desert"},
    {"question": "What is the deepest lake in the world?", "answer": "Lake Baikal"},
]
```

### Step 4: Systematic Comparison

All results are collected into a summary table. The optimization trajectory (composite score over iterations) is logged as a stepped metric on the parent run, making it easy to visualize improvement in the MLflow UI chart view.

## Running the Lesson

```bash
cd tutorial/level_2/M6_prompt_engineering/2_prompt_optimization
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Optimization Problem
============================================================
  Dataset:  5 Q&A pairs (geography)
  Model:    gemma4:e2b
  Scoring:  exact_match (50%) + brevity (30%) + confidence (20%)

============================================================
Part 2: Manual Prompt Optimization (instruction variants)
============================================================
  [0] baseline                        match=0.80  brevity=0.60  composite=0.78  (12.3s)
  [1] concise_instruction             match=1.00  brevity=0.80  composite=0.94  (8.1s)
  [2] role_assignment                 match=1.00  brevity=0.70  composite=0.91  (9.5s)
  [3] structured_constraints          match=1.00  brevity=1.00  composite=1.00  (7.2s)

============================================================
Part 3: Few-Shot Example Optimization
============================================================
  Base prompt: 'structured_constraints' (composite=1.00)

  [4] few_shot_0_examples            match=1.00  brevity=1.00  composite=1.00  (7.3s)
  [5] few_shot_1_examples            match=1.00  brevity=1.00  composite=1.00  (7.8s)
  [6] few_shot_2_examples            match=1.00  brevity=1.00  composite=1.00  (8.1s)
  [7] few_shot_3_examples            match=1.00  brevity=1.00  composite=1.00  (8.5s)

============================================================
Part 4: Systematic Comparison
============================================================
  (summary table with all variants and scores)

  Best variant: structured_constraints
  Best composite score: 1.00
```

Exact scores will vary depending on LLM behavior, but you should see a clear trend from baseline to more constrained prompts.

In the MLflow UI, expand the `prompt_optimization` parent run to see all eight nested runs. Use the chart view on `optimization_trajectory` to visualize the improvement curve.

## Key Takeaways

- **Measure, do not guess.** A fixed evaluation dataset and consistent scoring turns prompt engineering from art into science.
- **Instruction specificity matters.** Vague prompts produce verbose, hedging answers. Structured constraints dramatically improve output quality.
- **Few-shot examples help, but instructions come first.** Adding examples to a weak prompt may not help as much as improving the instructions.
- **MLflow makes optimization reproducible.** Every variant is logged with its prompt text, scores, and per-question results. You can always go back and understand why one variant outperformed another.
- **Log the trajectory.** Stepped metrics let you visualize the optimization path and confirm that changes are genuinely improving performance.

## Next Steps

Continue to L2-M7.1 (AI Gateway Routing) to learn how MLflow's AI Gateway can route LLM requests across providers. For deeper evaluation techniques, see L2-M3 (Deep Evaluation).
