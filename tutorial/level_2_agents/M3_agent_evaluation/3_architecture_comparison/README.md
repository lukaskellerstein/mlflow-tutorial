# L3-1.3 — Agent Architecture Comparison

**Level:** Expert
**Duration:** 2.5 hours

## Overview

Different agent architectures make fundamentally different tradeoffs between quality, latency, cost, and complexity. This lesson builds a controlled evaluation framework that runs three agent architectures on the same benchmark and compares them across multiple metrics. You will learn how to identify Pareto-optimal designs -- architectures that cannot be improved on one axis without sacrificing another.

## Prerequisites

- Completed: L3-1.1 (Agent Testing), L3-1.2 (Quality Metrics)
- Completed: L2-5.1 (LangChain Agents), L2-5.2 (LangGraph Agents)
- MLflow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Why Compare Architectures?

Choosing an agent architecture is rarely about picking "the best" one. Instead, it is about finding the right tradeoff for your use case:

- A **simple chain** (prompt -> LLM -> answer) is fast and cheap but cannot use tools or reason iteratively.
- A **ReAct agent** can reason and use tools but costs more tokens and has higher latency due to its reasoning loop.
- A **multi-step pipeline** (classify -> process -> respond) provides structured control flow but adds complexity and LLM calls even when they are not needed.

### Controlled Evaluation Methodology

A valid architecture comparison requires:

1. **Same task** -- all architectures answer the same questions
2. **Same model** -- all use the same LLM (google/gemma-4-26b-a4b) at the same temperature
3. **Same tools** -- all have access to identical tool implementations
4. **Same metrics** -- all are scored with the same evaluation functions
5. **Same environment** -- all run in the same session, logged to the same MLflow experiment

Without these controls, you cannot attribute performance differences to the architecture itself.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Correctness | Does the answer contain the expected keyword? (0 or 1) |
| Tool Usage | Did the architecture use tools appropriately? (0, 0.5, or 1) |
| Latency | Wall-clock time per question (seconds) |
| Token Efficiency | Quality per token -- higher means better answers with fewer tokens |

### The Pareto Frontier

An architecture is **Pareto-optimal** if no other architecture is better on every metric simultaneously. The Pareto frontier is the set of all such non-dominated architectures. When choosing an architecture for production, you pick from this frontier based on your priorities (e.g., "I need the lowest latency that still has > 0.8 correctness").

## Step-by-Step

### Step 1: Define Shared Tools and Dataset

All three architectures share a knowledge lookup tool and a word count tool, plus a 5-question evaluation dataset with expected answers:

```python
@tool
def lookup(topic: str) -> str:
    """Look up factual information about a technology topic."""
    ...


EVAL_DATASET = [
    {"question": "What is Python?", "expected_keyword": "readability", "needs_tool": True},
    ...,
]
```

### Step 2: Implement Three Architectures

1. **Simple Chain** -- Direct LLM call with a system message. No tool access.
2. **ReAct Agent** -- `langchain.agents.create_agent` with the shared tools.
3. **Multi-step Pipeline** -- A `StateGraph` with classify, process, and respond nodes.

### Step 3: Evaluate Each Architecture

Each architecture runs on all 5 test cases. Per-case metrics (correctness, tool usage, latency, token estimate) are collected and logged as nested MLflow runs under a parent `architecture_comparison` run.

### Step 4: Build Comparison Table and Pareto Analysis

Aggregate metrics are computed per architecture using pandas, and a comparison table is printed. The script identifies the Pareto frontier -- architectures that are not dominated on quality vs. latency.

## Running the Lesson

```bash
cd tutorial/level_3/M1_agent_evaluation/3_architecture_comparison
uv sync
uv run python main.py
```

## Expected Output

```
======================================================================
  L3-1.3 — Agent Architecture Comparison
======================================================================

----------------------------------------------------------------------
  Evaluating: simple_chain
----------------------------------------------------------------------
  [PASS] Q1: What is Python and what is it known for?
         Answer: Python is a high-level programming language...
         Correctness=1  ToolUsage=0.0  Latency=1.23s
  ...

----------------------------------------------------------------------
  Evaluating: react_agent
----------------------------------------------------------------------
  [PASS] Q1: What is Python and what is it known for?
         Answer: Python is a high-level programming language...
         Correctness=1  ToolUsage=1.0  Latency=4.56s
  ...

======================================================================
  COMPARISON TABLE: Architecture x Metric
======================================================================

  Architecture            Correct  ToolUse  Latency  Tokens  Quality  Efficiency
  --------------------------------------------------------------------------
  simple_chain              0.600    0.400    1.20s      85    0.500       0.588
  react_agent               0.800    0.800    4.50s     250    0.800       0.320
  multi_step_pipeline       0.800    0.600    3.20s     120    0.700       0.583

======================================================================
  COST-QUALITY TRADEOFF ANALYSIS
======================================================================

  Best quality:       react_agent (score=0.800)
  Fastest:            simple_chain (latency=1.200s)
  Most efficient:     simple_chain (efficiency=0.588)

  Pareto Frontier (quality vs latency):
    * simple_chain: quality=0.500, latency=1.200s
    * react_agent: quality=0.800, latency=4.500s
```

(Exact numbers will vary depending on LLM responses.)

## Key Takeaways

- A simple chain is fast and cheap but cannot leverage tools, limiting its accuracy on questions that require external knowledge.
- A ReAct agent achieves higher quality through tool use but at significantly higher latency and token cost.
- A multi-step pipeline offers a middle ground with structured control flow, though it adds unnecessary LLM calls for simple questions.
- The Pareto frontier helps you identify which architectures are worth considering -- dominated architectures can be eliminated from consideration.
- MLflow nested runs make it straightforward to organize and compare architecture evaluations side by side.

## Next Steps

In **L3-1.4 (Agent Optimization)**, you will take the best-performing architecture from this comparison and systematically optimize it -- tuning prompts, temperature, and tool descriptions to push quality higher while managing cost.
