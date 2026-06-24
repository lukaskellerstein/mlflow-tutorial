# L3-2.2 — Codex SDK + MLflow Integration

**Level:** Expert
**Duration:** 2.5 hours

## Overview

This lesson demonstrates how to build a custom MLflow integration for a code generation agent modelled after the Codex SDK pattern. Since Codex SDK is TypeScript-based and requires an OpenAI API key, we build a simulated Codex-style agent using a local LLM and focus on the MLflow integration pattern -- tracing multi-step code generation pipelines, scoring code quality, and comparing generation strategies.

## Prerequisites

- Completed: L1-5.2 (Manual Tracing), L2-5.1 (LangChain Agent Tracking)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Code Generation Agent Pattern

The Codex SDK follows a multi-step code generation workflow:

1. **Plan** -- break a coding task into implementation steps
2. **Generate** -- produce code from a natural-language prompt (optionally guided by the plan)
3. **Review** -- evaluate the generated code for correctness and quality
4. **Refine** -- improve the code based on review feedback

This pipeline is common across code generation tools (Codex, Copilot, Cursor). The MLflow integration pattern shown here applies to any of them.

### Why Custom Integration?

Unlike LangChain (which has `mlflow.langchain.autolog()`), code generation agents from non-LangChain frameworks need manual instrumentation. This lesson shows the pattern:

- Use `@mlflow.trace` on each pipeline stage
- Use `mlflow.start_span()` for orchestration-level tracing
- Build custom quality scorers for code-specific metrics
- Compare strategies using nested runs

### Code Quality Metrics

We define three custom scorers:

| Metric | What it measures | Score |
|--------|-----------------|-------|
| `code_completeness` | Has functions/classes, return statements, sufficient length | 0-1 |
| `has_error_handling` | Contains try/except, validation, error types | 0-1 |
| `follows_conventions` | Docstrings, type hints, proper indentation | 0-1 |

## Step-by-Step

### Step 1: Define the CodeGenAgent

The agent wraps a local LLM with four traced methods -- `plan()`, `generate_code()`, `review_code()`, and `refine()`. Each method is decorated with `@mlflow.trace` so every invocation creates a span in MLflow.

```python
class CodeGenAgent:
    @mlflow.trace(name="codegen_plan")
    def plan(self, prompt: str) -> str:
        # Break task into numbered steps
        ...

    @mlflow.trace(name="codegen_generate")
    def generate_code(self, prompt: str, plan: str | None = None) -> str:
        # Generate Python code from prompt (optionally guided by plan)
        ...
```

### Step 2: Build the Pipeline Orchestrator

The `run_pipeline()` method chains the four stages together. It uses `mlflow.start_span()` at the orchestration level to capture the overall inputs/outputs alongside the individual stage traces.

```python
@mlflow.trace(name="codegen_pipeline")
def run_pipeline(self, prompt: str, *, use_plan: bool = False) -> dict:
    with mlflow.start_span(name="pipeline_orchestration") as span:
        span.set_inputs({"prompt": prompt, "use_plan": use_plan})
        plan_text = self.plan(prompt) if use_plan else None
        code = self.generate_code(prompt, plan=plan_text)
        review = self.review_code(code, prompt)
        final_code = self.refine(code, review, prompt)
        ...
```

### Step 3: Implement Quality Scorers

Three rule-based scorers evaluate the generated code without requiring an LLM judge:

```python
def score_code_completeness(code: str, prompt: str) -> float:
    checks = [
        "def " in code or "class " in code,
        "return " in code,
        len(code.strip().splitlines()) >= 3,
        ":" in code,
    ]
    return sum(checks) / len(checks)
```

### Step 4: Run Tasks and Log Metrics

Each task runs inside a nested MLflow run. The parent run logs the strategy parameters and aggregate metrics; each nested child logs per-task metrics and artifacts (generated code, plan, review feedback).

### Step 5: Compare Strategies

Two strategies are compared:

- **Direct generation** -- prompt goes straight to code generation
- **Plan-then-generate** -- the agent first produces a plan, then generates code following it

Both run the same three tasks. Aggregate quality scores are compared to determine which strategy produces better code.

## Running the Lesson

```bash
cd tutorial/level_3/M2_custom_integrations/2_codex_sdk
uv sync
uv run python main.py
```

## Expected Output

The script prints each task with a code snippet preview, quality scores, and a final comparison table:

```
Strategy Comparison: Direct vs Plan-Then-Generate
======================================================================
Metric                        Direct      Planned      Delta
-----------------------------------------------------------------
  latency                      12.30        18.50      +6.20
  code_lines                   15.00        22.00      +7.00
  code_completeness             0.75         1.00      +0.25
  has_error_handling             0.50         0.83      +0.33
  follows_conventions            0.58         0.75      +0.17

  Overall Quality                0.61         0.86

  Winner: Plan-Then-Generate
```

In the MLflow UI you will see:

- **Experiment:** `L3/M2_custom_integrations/2_codex_sdk`
- **Two parent runs:** `strategy_direct` and `strategy_planned`
- **Nested runs:** one per task per strategy, with logged code artifacts
- **Traces:** click any run to see the full `codegen_pipeline` trace with spans for plan, generate, review, and refine stages

## Key Takeaways

- Code generation agents follow a predictable pipeline (plan, generate, review, refine) that maps cleanly to MLflow spans.
- `@mlflow.trace` on each stage and `mlflow.start_span()` at the orchestration level give full visibility into the generation process.
- Rule-based code quality scorers (completeness, error handling, conventions) provide fast, deterministic evaluation without needing an LLM judge.
- Comparing generation strategies with nested runs lets you make data-driven decisions about agent architecture.
- This integration pattern works for any code generation tool -- Codex, Copilot, or custom agents.

## Next Steps

In L3-2.3 (DeepAgents), we will apply similar custom integration patterns to a multi-agent orchestration framework and compare it with LangGraph-based multi-agent systems.
