# L2-M4.2 — GAIA General Assistant Benchmark

**Level:** Agents
**Duration:** 90 min

## Overview

GAIA (General AI Assistants) is a benchmark of ~450 real-world questions that test multi-step reasoning, tool use, and factual knowledge. Unlike coding-focused SWE-Bench, GAIA evaluates general assistant capabilities — the kind of tasks users actually ask AI agents to solve. This lesson builds a benchmark harness that runs a ReAct agent against GAIA tasks and tracks results in MLflow.

## Prerequisites

- Completed: L2-M4.1 (SWE-Bench) for benchmark harness patterns
- Completed: L2-M1.1 (LangChain Agents) for ReAct agent basics
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-26b-a4b` loaded
- HuggingFace `datasets` library installed

## Concepts

### What is GAIA?

GAIA tests whether AI assistants can solve questions that require:
- **Multi-step reasoning** — combining multiple facts to reach an answer
- **Tool use** — calculators, search, file analysis
- **Factual knowledge** — geography, science, history, current events

Questions are organized into 3 difficulty levels. Level 1 questions require 1-2 reasoning steps, Level 2 requires 3-5 steps, and Level 3 requires complex multi-tool workflows.

Each question has a single unambiguous correct answer, enabling exact-match evaluation — no LLM judge needed.

### Benchmark Harness Pattern

The harness follows the same pattern as SWE-Bench:
1. Load the dataset from HuggingFace
2. Build an agent with appropriate tools
3. Run each task in a nested MLflow run
4. Compare configurations side-by-side
5. Aggregate results with accuracy and latency metrics

### Key Metrics

- **Exact match accuracy** — does the agent's final answer match the ground truth?
- **Per-level accuracy** — performance breakdown by difficulty
- **Latency** — time per task
- **Tool usage** — number of tool calls per task

## Step-by-Step

### Step 1: Load the GAIA Dataset

We load Level 1 questions from HuggingFace. Each instance has a `Question`, `Final answer`, `Level`, and `task_id`.

```python
ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
```

### Step 2: Build the Agent

A ReAct agent with three tools: `calculator` (math), `knowledge_lookup` (factual retrieval), and `text_analyzer` (content extraction). The tools are intentionally simple — the focus is on the benchmark harness, not tool sophistication.

### Step 3: Run the Benchmark

Each GAIA instance runs in a nested MLflow run. We compare two configurations:
- **focused** (temperature=0.3) — precise, deterministic reasoning
- **creative** (temperature=0.7) — more exploratory reasoning

### Step 4: Analyze Results

Aggregate metrics per configuration: accuracy, average latency, success rate. If multiple difficulty levels are present, break down accuracy by level.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M4_agent_benchmarks/2_gaia
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-M4.2 -- GAIA General Assistant Benchmark
============================================================

Step 1: Loading GAIA dataset ...
  Loaded 165 Level-1 instances, using 5 for demo

Step 2: Running benchmark ...

============================================================
Config: focused  (temperature=0.3)
============================================================
  [a1b2c3d4] L1 match=True latency=3.2s
  [e5f6g7h8] L1 match=False latency=4.1s
  ...

============================================================
Summary Comparison
============================================================
         accuracy  avg_latency  success_rate
config
creative     0.40         3.85          1.0
focused      0.60         3.42          1.0
```

In the MLflow UI: a parent run with two config children, each containing per-task nested runs with exact_match metrics and response artifacts.

## Key Takeaways

- GAIA tests general assistant capabilities with exact-match evaluation — no judge model needed
- The same benchmark harness pattern (load → agent → nested runs → aggregate) works across benchmarks
- Per-level accuracy breakdown reveals which reasoning depths the agent handles well
- Lower temperature typically improves accuracy on factual tasks (less hallucination)

## Next Steps

In L2-M4.3, you'll learn to design your own domain-specific agent benchmark from scratch — defining task taxonomies, creating evaluation datasets, and building reusable harness patterns.
