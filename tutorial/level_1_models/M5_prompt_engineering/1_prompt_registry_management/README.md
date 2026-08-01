# L1-M5.1 — Prompt Registry and Management

**Level:** Essentials
**Duration:** 40 min

## Overview

The MLflow Prompt Registry provides centralized, versioned storage for prompt templates. This lesson covers the full prompt management lifecycle: registering versions, setting aliases for deployment, loading prompts dynamically, A/B testing multiple variants against the same questions, and making data-driven decisions about which prompt to use in production.

## Prerequisites

- Completed: L1-M4 (Evaluations)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why a Prompt Registry?

Prompts are a core part of any LLM application, yet they are often hardcoded in source files. The Prompt Registry solves this by treating prompts as first-class versioned artifacts:

- **Version** prompts independently of code deployments
- **Compare** prompt variants across evaluation runs
- **Roll back** to a previous prompt when a new one underperforms
- **Share** prompts across teams and services via aliases

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Prompt** | A named entry in the registry |
| **Prompt Version** | An immutable snapshot of a template |
| **Template** | Text with `{{variable}}` placeholders |
| **Alias** | A mutable pointer (e.g., `production`) to a specific version |
| **Tags** | Key-value metadata on prompts or versions |

### A/B Testing Prompts

The most reliable way to choose between prompt variants is empirical testing:
1. Register multiple versions with different instructions
2. Run the same evaluation questions through each variant
3. Measure objective metrics (response length, word count)
4. Compare results in a structured way

## Step-by-Step

### Step 1: Register Prompt Versions

`mlflow.genai.register_prompt()` creates a new prompt if the name does not exist, or adds a new version if it does:

```python
pv = mlflow.genai.register_prompt(
    name="qa_prompt",
    template="Answer in 1-2 sentences.\n\nQuestion: {{question}}",
    commit_message="Concise variant",
    tags={"style": "concise"},
)
```

### Step 2: Set Aliases

Aliases decouple deployed code from version numbers:

```python
mlflow.genai.set_prompt_alias("qa_prompt", alias="production", version=2)
```

### Step 3: Load and Format

Load by version or alias, then format with variables:

```python
prompt = mlflow.genai.load_prompt("prompts:/qa_prompt@production")
text = prompt.format(question="What is recursion?")
```

### Step 4: A/B Test Variants

Each variant answers the same questions in its own MLflow run:

```python
formatted = prompt_version.format(question=question)
response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": formatted}],
)
```

### Step 5: Compare and Decide

Build a comparison table and determine the best variant:

```python
summary = results_df.groupby("variant").agg(avg_words=("word_count", "mean"))
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M5_prompt_engineering/1_prompt_registry_management
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Register Prompt Versions and Set Aliases
============================================================
  v1 [concise] -- Answer the question in 1-2 sentences...
  v2 [detailed] -- You are a thorough assistant...
  v3 [creative] -- You are a creative and engaging assistant...

  Alias 'production' -> v2 (detailed)

============================================================
Part 2: A/B Test Prompt Variants
============================================================
  --- Variant: concise (v1) ---
    Q1: What is a hash table?
       A: A hash table is a data structure...
       [18 words]
    ...

  --- Variant: detailed (v2) ---
    ...

============================================================
Part 3: Compare Results Across Variants
============================================================
  Per-variant summary:
           avg_words  min_words  max_words  total_responses
  concise       18.7         12         25                3
  creative      42.3         35         52                3
  detailed      55.0         45         65                3

  Best balanced variant (closest to ~60 words): detailed
  Comparison artifacts logged to MLflow.
```

In the MLflow UI:
- **Prompt Registry**: see the registered prompt with all three versions
- **Experiment**: 3 A/B test runs + 1 comparison run with artifacts

## Key Takeaways

- `mlflow.genai.register_prompt()` creates immutable, versioned prompt snapshots with a full audit trail
- Aliases (`production`, `staging`) decouple deployed code from specific version numbers
- `mlflow.genai.load_prompt()` retrieves prompts by version, alias, or URI
- A/B testing is a structured process: same questions, same model, different prompts -- then measure and compare objectively
- MLflow runs capture the full experiment, making comparison easy in the UI or programmatically

## Next Steps

In L1-M5.2 (Prompt Optimization), you will go beyond manual A/B testing and explore systematic prompt tuning using evaluation metrics as the optimization target.
