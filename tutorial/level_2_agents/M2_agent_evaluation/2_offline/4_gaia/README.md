# L2-M2.2.4 — GAIA General Assistant Benchmark

**Level:** Agents
**Duration:** 90 min

## Overview

GAIA (General AI Assistants) is a benchmark of ~450 real-world questions that test multi-step reasoning, tool use, and factual knowledge. Unlike coding-focused SWE-Bench, GAIA evaluates general assistant capabilities — the kind of tasks users actually ask AI agents to solve. This lesson builds a benchmark harness that runs a Claude Agent SDK agent against GAIA tasks and tracks results in MLflow with hand-built tracing.

## Prerequisites

- Completed: L2-M2.2.3 (SWE-Bench) for the benchmark harness and SDK tracing patterns
- Completed: L2-M1.3 (Claude Agent SDK) for SDK basics
- MLflow server running at <http://127.0.0.1:5555>
- Claude Code CLI installed and authenticated (the SDK uses your existing login — LMStudio is not involved)
- **GAIA access granted on Hugging Face**: the dataset is gated. Log in with `hf auth login`, then visit <https://huggingface.co/datasets/gaia-benchmark/GAIA> and accept the terms — access is granted automatically. Without this, `load_dataset` fails with `DatasetNotFoundError`.

> **Cost note:** the demo run (5 tasks x 2 configurations) calls real Claude models and costs a few cents; every task is capped with `max_budget_usd`.

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
2. Give the agent its tools (in-process MCP server)
3. Run each task in a nested MLflow run
4. Compare configurations side-by-side
5. Aggregate results with accuracy, latency, and cost metrics

### Key Metrics

- **Exact match accuracy** — does the agent's final answer match the ground truth?
- **Per-level accuracy** — performance breakdown by difficulty
- **Latency** — time per task
- **Tool usage** — number of tool calls per task
- **Cost** — `ResultMessage.total_cost_usd`, aggregated per configuration

## Step-by-Step

### Step 1: Load the GAIA Dataset

We load Level 1 questions from HuggingFace. Each instance has a `Question`, `Final answer`, `Level`, and `task_id`.

```python
ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
```

### Step 2: Give the Agent Tools

Three in-process MCP tools: `calculator` (math), `knowledge_lookup` (factual retrieval), and `text_analyzer` (content extraction). The tools are intentionally simple — the focus is on the benchmark harness, not tool sophistication.

```python
@tool("calculator", "Evaluate a mathematical expression and return the result", {"expression": str})
async def calculator(args: dict[str, Any]) -> dict[str, Any]: ...


BENCH_SERVER = create_sdk_mcp_server(name="bench", version="1.0.0", tools=[calculator, knowledge_lookup, text_analyzer])
```

The agent is locked to exactly these tools: `tools=[]` disables the built-ins, `strict_mcp_config=True` blocks ambient MCP servers, and `setting_sources=[]` keeps local settings out of the context (see L2-M2.2.3 for why each matters).

### Step 3: Run the Benchmark

Each GAIA instance runs in a nested MLflow run, traced with `@mlflow.trace` plus a child span per tool call. We fix the model to `claude-sonnet-5` and compare two effort levels (the SDK has no temperature knob):

- **low_effort** — fast, cheap, shallow reasoning
- **high_effort** — more reasoning per step, higher latency and cost

### Step 4: Analyze Results

Aggregate metrics per configuration: accuracy, average latency, success rate, and total cost. If multiple difficulty levels are present, break down accuracy by level.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/4_gaia
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
L2-M2.2.4 -- GAIA General Assistant Benchmark
============================================================

Step 1: Loading GAIA dataset ...
  Loaded 53 Level-1 instances, using 5 for demo

Step 2: Running benchmark ...

============================================================
Config: low_effort  (model=claude-sonnet-5, effort=low)
============================================================
  [e1fc63a2] L1 match=True latency=8.0s
  [8e867cd7] L1 match=True latency=6.4s
  ...

============================================================
Summary Comparison
============================================================
             accuracy  avg_latency  total_cost_usd  success_rate
config
high_effort       0.6       27.168        0.241364           1.0
low_effort        0.6        5.322        0.061871           1.0
```

Note what the comparison shows on this sample: high effort cost 4x more and ran 5x slower for identical accuracy — on tasks this shallow, extra reasoning buys nothing. That is exactly the kind of finding a benchmark with cost tracking exists to surface.

In the MLflow UI: a parent run with two config children, each containing per-task nested runs with exact_match and cost_usd metrics, plus traces showing a `gaia_agent.run` root span with `tool_call.*` children.

## Key Takeaways

- GAIA tests general assistant capabilities with exact-match evaluation — no judge model needed
- The same benchmark harness pattern (load → agent → nested runs → aggregate) works across benchmarks and across agent frameworks
- Per-level accuracy breakdown reveals which reasoning depths the agent handles well
- With the Claude Agent SDK the tradeoff axis is effort, not temperature — and cost per configuration is measurable directly from the run data

## Next Steps

In L2-M2.2.5, you'll learn to design your own domain-specific agent benchmark from scratch — defining task taxonomies, creating evaluation datasets, and building reusable harness patterns.
