# L1-M5.1 — Prompt Registry

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

The MLflow Prompt Registry provides centralized, versioned storage for prompt templates. Instead of scattering prompts across code files, you register them in MLflow where they are versioned, aliased, tagged, and automatically linked to runs and traces. This lesson shows you how to register prompts, version them, load them by version or alias, and use them with LangChain.

## Prerequisites

- Completed: L1-M1 (Tracking), L1-M5 (Tracing basics)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` model loaded

## Concepts

### Why a Prompt Registry?

Prompts are a core part of any LLM application, yet they are often hardcoded in source files. This makes it difficult to:

- **Version** prompts independently of code deployments
- **Compare** prompt variants across evaluation runs
- **Roll back** to a previous prompt when a new one underperforms
- **Share** prompts across teams and services

The MLflow Prompt Registry solves these problems by treating prompts as first-class versioned artifacts, similar to how the Model Registry manages models.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Prompt** | A named entry in the registry (like a model name) |
| **Prompt Version** | An immutable snapshot of a template. Each registration creates a new version |
| **Template** | The text with `{{variable}}` placeholders. Can be a string or a list of chat messages |
| **Variables** | The set of placeholders extracted from the template (e.g., `{topic}`, `{audience}`) |
| **Alias** | A mutable pointer (e.g., `production`, `staging`) to a specific version |
| **Tags** | Key-value metadata on prompts or versions |

### Template Variable Syntax

MLflow uses **double curly braces** for variables: `{{variable}}`. This avoids conflicts with Python f-strings and JSON. When you need to use the prompt with LangChain (which uses single braces), call `prompt.to_single_brace_format()`.

## Step-by-Step

### Step 1: Register a Prompt (Version 1)

`mlflow.genai.register_prompt()` creates a new prompt if the name does not exist, or adds a new version if it does. Each version is immutable.

```python
v1 = mlflow.genai.register_prompt(
    name="explainer_prompt",
    template="Explain {{topic}} to a {{audience}} in 2-3 sentences.",
    commit_message="Initial explainer prompt",
    tags={"style": "concise"},
)
```

### Step 2: Create a New Version

Calling `register_prompt()` again with the same name creates version 2. The original version 1 remains unchanged.

```python
v2 = mlflow.genai.register_prompt(
    name="explainer_prompt",
    template="You are a friendly teacher. Explain {{topic}} in a way that a {{audience}} would understand.",
    commit_message="Add teacher persona",
)
```

### Step 3: Set an Alias

Aliases let you decouple your code from specific version numbers. Point `production` at the version you trust.

```python
mlflow.genai.set_prompt_alias("explainer_prompt", alias="production", version=2)
```

### Step 4: Load a Prompt

Load by version number or by alias. The special `@latest` alias always resolves to the highest version.

```python
# By version
v1 = mlflow.genai.load_prompt("explainer_prompt", version=1)

# By alias (URI format)
prod = mlflow.genai.load_prompt("prompts:/explainer_prompt@production")
```

### Step 5: Format and Use with LangChain

The `format()` method replaces `{{variable}}` placeholders. For LangChain, convert to single-brace format first.

```python
from langchain_core.prompts import ChatPromptTemplate

lc_template = prod.to_single_brace_format()
lc_prompt = ChatPromptTemplate.from_template(lc_template)
chain = lc_prompt | llm
response = chain.invoke({"topic": "recursion", "audience": "10-year-old"})
```

### Step 6: Search Prompts

List all registered prompts, optionally filtering by name pattern.

```python
prompts = mlflow.genai.search_prompts()
# Or filter: mlflow.genai.search_prompts(filter_string="name LIKE 'explainer%'")
```

## Running the Lesson

```bash
cd tutorial/level_1/M5_genai_features/1_prompt_registry
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Register a prompt template (version 1)
============================================================
  Registered: L1_M6_explainer_prompt v1
  Template:   Explain {{topic}} to a {{audience}} in 2-3 sentences.
  Variables:  {'audience', 'topic'}
  ...

Step 7: Use the production prompt with ChatOpenAI
============================================================
  LLM response:
  <LLM-generated explanation of recursion for a 10-year-old>
```

In the MLflow UI, navigate to the Prompt Registry to see the registered prompt with both versions and the `production` alias.

## Key Takeaways

- **`mlflow.genai.register_prompt()`** creates or versions a prompt in the registry.
- **Templates use `{{double_braces}}`** for variables; call `.to_single_brace_format()` for LangChain.
- **Aliases** (e.g., `production`) decouple deployed code from specific version numbers.
- **`mlflow.genai.load_prompt()`** retrieves prompts by version, alias, or URI.
- **Prompts are automatically linked** to runs and traces when loaded inside them.

## Next Steps

In the next lesson (L1-M5.2 Scorers and Judges), you will learn how to evaluate LLM outputs using built-in and custom scoring functions. In Level 2, we will explore prompt A/B testing and automated prompt optimization.
