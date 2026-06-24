# L3-1.4 — Agent Optimization

**Level:** Expert
**Duration:** 2 hours

## Overview

Agent quality depends on more than just the LLM — system prompts, temperature settings, and tool descriptions all have a measurable impact on correctness and tool selection. This lesson shows how to systematically tune each dimension, track every variant as a nested MLflow run, and identify the best configuration using data rather than intuition.

## Prerequisites

- Completed: L3-1.3 (Architecture Comparison)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model available

## Concepts

### Why Optimize Agents Systematically?

Agent behavior is sensitive to many knobs: the system prompt, the LLM temperature, and even the wording of tool descriptions. Changing any of these can shift correctness, tool selection accuracy, and latency in non-obvious ways. Without systematic evaluation, developers rely on manual spot-checks that miss regressions.

This lesson treats agent optimization as a search problem:

1. **Define a fixed evaluation benchmark** — the same 5 test cases are used for every variant so results are comparable.
2. **Vary one dimension at a time** — system prompt, temperature, tool descriptions — while holding the others constant.
3. **Log everything** — each variant becomes a nested MLflow run with parameters and metrics.
4. **Compare and select** — a summary table across all dimensions identifies the best configuration.

### Optimization Dimensions

| Dimension | What Changes | Why It Matters |
|---|---|---|
| System prompt | Instructions given to the LLM | Guides reasoning strategy and tool usage patterns |
| Temperature | Sampling randomness (0.0 = deterministic) | Affects consistency and creativity of responses |
| Tool descriptions | Docstrings on each tool function | Determines whether the LLM picks the right tool |

### Evaluation Metrics

- **Correctness** — does the answer contain the expected value?
- **Tool Selection** — did the agent use the correct tool for the question?
- **Quality Score** — average of correctness and tool selection (composite metric)
- **Latency** — wall-clock time per evaluation run

## Step-by-Step

### Step 1: Define the Agent and Evaluation Benchmark

We build a ReAct agent with three tools (`calculator`, `unit_converter`, `fact_lookup`) and define 5 test cases that each require a specific tool:

```python
EVAL_CASES = [
    {"question": "What is 15 * 24 + 100?", "expected": "460", "needs_tool": "calculator"},
    {"question": "Convert 42 km to miles.", "expected": "26.1", "needs_tool": "converter"},
    {"question": "What is the speed of light?", "expected": "299", "needs_tool": "fact"},
    {"question": "How many pounds is 10 kg?", "expected": "22.0", "needs_tool": "converter"},
    {"question": "Who created Python?", "expected": "Guido", "needs_tool": "fact"},
]
```

### Step 2: System Prompt Optimization

Three system prompts are tested — from minimal to highly structured:

- **Minimal**: `"You are a helpful assistant."` — no tool guidance
- **Detailed**: Mentions available tools and encourages their use
- **Structured**: Numbered rules mapping question types to specific tools

The structured prompt is expected to produce the best tool selection because it explicitly tells the LLM which tool to use for each question type.

### Step 3: Temperature Tuning

Using the best system prompt from Step 2, we sweep four temperatures (0.0, 0.3, 0.7, 1.0). Lower temperatures are more deterministic and typically produce more consistent tool usage, while higher temperatures introduce randomness that can cause the agent to skip tools or hallucinate.

### Step 4: Tool Description Optimization

Two sets of tool descriptions are compared:

- **Original** — terse one-line docstrings (e.g., `"Evaluate a math expression."`)
- **Improved** — detailed descriptions with usage guidance and argument examples

Better descriptions help the LLM understand when and how to use each tool, improving selection accuracy.

### Step 5: Optimization Summary

All results are combined into a comparison table. MLflow logs the optimization trajectory as stepped metrics so you can visualize the quality curve in the MLflow UI chart view.

## Running the Lesson

```bash
cd tutorial/level_3/M1_agent_evaluation/4_agent_optimization
uv sync
uv run python main.py
```

## Expected Output

The script prints per-case results for each variant, followed by a summary table:

```
  Variant                Dimension          Correct  ToolSel  Quality  Latency
  --------------------------------------------------------------------------
  prompt_minimal         system_prompt        0.600    0.400    0.500   1.234s
  prompt_detailed        system_prompt        0.800    0.800    0.800   1.456s
  prompt_structured      system_prompt        0.800    1.000    0.900   1.389s
  temp_0.0               temperature          0.800    1.000    0.900   1.345s
  temp_0.3               temperature          0.800    0.800    0.800   1.567s
  ...
  tools_original         tool_descriptions    0.800    1.000    0.900   1.345s
  tools_improved         tool_descriptions    1.000    1.000    1.000   1.234s

  BEST CONFIGURATION: tools_improved
```

In the MLflow UI:
- Navigate to experiment `L3/M1_agent_evaluation/4_agent_optimization`
- The parent run `agent_optimization` contains all nested variant runs
- Each nested run has parameters (`dimension`, `variant`, `temperature`) and metrics (`avg_correctness`, `avg_tool_selection`, `quality_score`)
- The parent run has stepped metrics (`opt_quality`, `opt_correctness`) showing the optimization trajectory

## Key Takeaways

- System prompts with explicit tool-routing instructions significantly improve tool selection accuracy.
- Lower temperatures (0.0-0.3) generally produce more consistent agent behavior than higher values.
- Detailed tool descriptions with usage guidance and argument examples help the LLM choose the right tool.
- Systematic evaluation with a fixed benchmark is essential — intuition about which settings "should" work often does not match reality.
- MLflow nested runs make it easy to compare variants side-by-side and identify the best configuration.

## Next Steps

In L3-1.5 (End-to-End Agent Evaluation Pipeline), you will combine all the evaluation techniques from this module into a complete automated pipeline with CI/CD integration, regression detection, and quality dashboards.
