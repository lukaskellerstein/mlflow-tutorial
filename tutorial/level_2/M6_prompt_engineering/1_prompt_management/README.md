# L2-M6.1 — Prompt Management at Scale

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

In production LLM systems, prompts are not static strings buried in code -- they are versioned assets that evolve over time. This lesson covers how to manage prompts at scale using MLflow's Prompt Registry: registering multiple variants, running A/B tests to compare them, and making data-driven decisions about which prompt to deploy. You will learn patterns that enable team collaboration on prompt engineering without code changes.

## Prerequisites

- Completed: L1-M6.1 (Prompt Registry basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why Prompt Versioning Matters

As teams iterate on LLM applications, prompts change frequently. Without versioning:
- There is no audit trail of what changed and when.
- Rolling back a bad prompt change requires code redeployment.
- Multiple team members editing prompts creates conflicts.

MLflow's Prompt Registry solves this by treating prompts as first-class versioned artifacts, similar to model versioning in the Model Registry. Each version is immutable once created, and aliases (like `production` or `staging`) let you swap active prompts without touching code.

### A/B Testing Prompts

The most reliable way to choose between prompt variants is to test them empirically. A/B testing prompts means:
1. Registering multiple prompt versions with different instructions.
2. Running the same set of evaluation questions through each variant.
3. Measuring objective metrics (response length, word count, latency).
4. Comparing results in a structured way.

This is more rigorous than subjective "this one feels better" evaluation, and MLflow makes it straightforward by logging each variant's results as a separate run.

### Team Collaboration Patterns

With the Prompt Registry, teams can adopt workflows like:
- **Version-controlled prompts**: Each prompt change creates a new version with a commit message, providing a changelog.
- **Alias-based deployment**: Application code loads `prompts:/my_prompt@production`. Moving the alias to a new version changes the prompt without redeploying.
- **Tag-based organization**: Tags like `style=concise` or `team=search` make it easy to filter and find prompts.
- **Review before promotion**: Test a new version as `@staging`, review metrics, then promote to `@production`.

## Step-by-Step

### Step 1: Register Multiple Prompt Variants

We register three versions of a Q&A prompt, each with a different system instruction style:

- **Concise** (v1): Direct, factual, 1-2 sentences
- **Detailed** (v2): Thorough explanations with examples, 3-5 sentences
- **Creative** (v3): Vivid language, analogies, and metaphors

```python
VARIANTS = [
    {
        "label": "concise",
        "template": [
            {"role": "system", "content": "You are a concise assistant. Answer in 1-2 sentences."},
            {"role": "user", "content": "{{question}}"},
        ],
        "commit_message": "Concise variant — short, factual answers",
        "tags": {"style": "concise", "target_length": "short"},
    },
    # ... detailed and creative variants
]

for variant in VARIANTS:
    pv = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=variant["template"],
        commit_message=variant["commit_message"],
        tags=variant["tags"],
    )
```

Each call to `register_prompt()` with the same name auto-increments the version number. The chat template format (list of role/content dicts) is used because it maps cleanly to LLM message structures.

### Step 2: A/B Test Each Variant

For each prompt version, we load it, convert to LangChain format, and run the same three test questions:

```python
prompt_version = mlflow.genai.load_prompt(PROMPT_NAME, version=version)
lc_messages = prompt_version.to_single_brace_format()
lc_prompt = ChatPromptTemplate.from_messages(
    [(msg["role"], msg["content"]) for msg in lc_messages]
)
chain = lc_prompt | llm
```

Key detail: `to_single_brace_format()` converts MLflow's `{{variable}}` syntax to LangChain's `{variable}` syntax. Each variant runs in its own MLflow run, logging per-question metrics and averages.

### Step 3: Compare and Decide

After all variants are tested, we build a comparison table and determine which variant best fits our target profile. The comparison is logged as CSV artifacts to MLflow for future reference.

```python
summary = results_df.groupby("variant").agg(
    avg_chars=("char_length", "mean"),
    avg_words=("word_count", "mean"),
    ...
)
```

## Running the Lesson

```bash
cd tutorial/level_2/M6_prompt_engineering/1_prompt_management
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: Register multiple prompt versions
============================================================
  v1 [concise] — registered
    System: You are a concise assistant. Answer the question in 1-2 sentences...
  v2 [detailed] — registered
    System: You are a thorough assistant. Answer the question with a detailed e...
  v3 [creative] — registered
    System: You are a creative and engaging assistant. Answer the question usin...

  Total versions registered: 3

============================================================
Part 2: A/B test prompts with ChatOpenAI
============================================================

  --- Variant: concise (v1) ---
    Q1: What is a hash table?
       A: A hash table is a data structure that stores key-value pairs...
       [120 chars, 18 words]
    ...

  --- Variant: detailed (v2) ---
    ...

  --- Variant: creative (v3) ---
    ...

============================================================
Part 3: Compare results across variants
============================================================

  Per-variant summary:
           avg_chars  avg_words  min_chars  max_chars  total_responses
  concise      120.3       18.7         95        150                3
  creative     280.7       42.3        210        340                3
  detailed     350.0       55.0        290        420                3

  Best balanced variant (closest to ~60 words): detailed

  Comparison artifacts logged to MLflow.

============================================================
Done! Check the MLflow UI at http://127.0.0.1:5000
============================================================
```

In the MLflow UI, you will see:
- **Experiment**: `L2/M6_prompt_engineering/1_prompt_management`
- **4 runs**: one per variant (concise, detailed, creative) plus a comparison run
- **Artifacts**: `ab_test_full_results.csv` and `ab_test_summary.csv` on the comparison run
- **Metrics**: per-question character and word counts on each variant run

## Key Takeaways

- **Prompt versions are immutable**: each call to `register_prompt()` creates a new version, providing a full audit trail of prompt evolution.
- **Chat templates map to LLM messages**: using the `[{"role": ..., "content": ...}]` format lets you control system instructions separately from user input.
- **`to_single_brace_format()` bridges MLflow and LangChain**: MLflow uses `{{var}}`, LangChain uses `{var}` -- this method handles the conversion.
- **A/B testing is a structured process**: same questions, same model, different prompts -- then measure and compare objectively.
- **MLflow runs capture the full experiment**: each variant's results are logged separately, making comparison easy in the UI or programmatically.

## Next Steps

In L2-M6.2 (Prompt Optimization), you will go beyond manual A/B testing and explore automated prompt tuning with `mlflow.genai.optimize`, using evaluation metrics as the optimization target.
