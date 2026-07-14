# L2-M4.3 — Building a Custom Domain-Specific Agent Benchmark

**Level:** Agents
**Duration:** 90 min

## Overview

Established benchmarks like SWE-Bench and GAIA test general capabilities, but your agents operate in specific domains with unique requirements. This lesson teaches the methodology of designing, building, and running your own agent benchmark — using a customer support agent as the example domain. You'll build a reusable harness pattern that works for any domain.

## Prerequisites

- Completed: L2-M4.1 (SWE-Bench) and L2-M4.2 (GAIA) for benchmark harness patterns
- Completed: L2-M1.1 (LangChain Agents) for ReAct agent basics
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-26b-a4b` loaded

## Concepts

### Why Build a Custom Benchmark?

General benchmarks tell you if an agent can reason and use tools. They don't tell you if your customer support agent correctly applies your refund policy, or if your coding agent follows your team's style guide. Domain-specific benchmarks measure what matters for your use case.

### Benchmark Design Methodology

1. **Define a task taxonomy** — categorize the types of tasks your agent handles, with difficulty levels
2. **Create an evaluation dataset** — input/expected output pairs covering each category
3. **Choose metrics** — accuracy, tool selection, latency, domain-specific scores
4. **Build a harness** — automated runner with MLflow tracking
5. **Establish baselines** — run multiple configurations to set reference performance
6. **Analyze statistically** — per-category breakdown, difficulty curves, variance

### Task Taxonomy Design

A good taxonomy has:
- **Categories** that map to real user intents (not artificial splits)
- **Difficulty levels** based on the number of reasoning steps required
- **Required tools** per category (validates tool selection)
- **Clear success criteria** — what constitutes a correct answer

### Evaluation Dataset Principles

- **Cover all categories** — at least 2-3 examples per category
- **Include edge cases** — boundary conditions, ambiguous inputs
- **Unambiguous answers** — each task has one correct answer (enables exact-match scoring)
- **Realistic inputs** — use language your actual users would use
- **Version the dataset** — log it as an MLflow artifact for reproducibility

## Step-by-Step

### Step 1: Define the Task Taxonomy

We define three task categories for a customer support agent, each with increasing difficulty:

```python
TASK_TAXONOMY = {
    "order_lookup": {"difficulty": 1, "required_tools": ["check_order"]},
    "refund_request": {"difficulty": 2, "required_tools": ["check_order", "check_refund_policy"]},
    "complex_complaint": {"difficulty": 3, "required_tools": ["check_order", "check_refund_policy", "escalate_ticket"]},
}
```

### Step 2: Create the Evaluation Dataset

Six tasks covering all categories and difficulty levels. Each has `task_id`, `category`, `difficulty`, `input`, `expected_answer`, and `expected_tools`.

### Step 3: Build Domain-Specific Tools

Three tools that simulate a real support system: `check_order` (database lookup), `check_refund_policy` (policy engine), `escalate_ticket` (handoff to human).

### Step 4: Run the Benchmark Harness

Each task runs in a nested MLflow run. We compare a conservative (temp=0.2) and balanced (temp=0.5) configuration.

### Step 5: Statistical Analysis

Break down accuracy by category and difficulty level. Compute latency statistics (mean, stdev, min, max) per configuration. This reveals which task types the agent struggles with.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M4_agent_benchmarks/3_custom_benchmark
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-M4.3 -- Custom Domain-Specific Agent Benchmark
============================================================

Step 1: Task Taxonomy
  order_lookup (difficulty=1): Look up order status and details
  refund_request (difficulty=2): Process a refund based on policy rules
  complex_complaint (difficulty=3): Handle multi-step complaints

Step 2: Dataset — 6 tasks
  [OL-001] order_lookup D1
  [OL-002] order_lookup D1
  ...

Step 3-4: Running benchmark ...

============================================================
Config: conservative  (temperature=0.2)
============================================================
  [OL-001] D1 correct=True latency=2.1s
  [RR-001] D2 correct=True latency=3.8s
  [CC-001] D3 correct=True latency=5.2s
  ...

--- Accuracy by Category ---
category     complex_complaint  order_lookup  refund_request
config
balanced                  0.5           1.0             0.5
conservative              1.0           1.0             1.0

--- Accuracy by Difficulty ---
difficulty        1     2     3
config
balanced        1.0   0.5   0.5
conservative    1.0   1.0   1.0
```

In MLflow UI: benchmark definition artifact (taxonomy + dataset JSON), per-config CSV results, category-level accuracy metrics.

## Key Takeaways

- Domain-specific benchmarks measure what general benchmarks can't — your agent's fit for your use case
- A good taxonomy maps to real user intents and has clear difficulty levels
- The harness pattern (taxonomy → dataset → tools → runner → analysis) is reusable across any domain
- Per-category and per-difficulty breakdowns reveal specific failure modes
- Version your benchmark definition as an MLflow artifact for reproducibility

## Next Steps

You've now completed the Agent Benchmarks module. You can:
- Expand your custom benchmark with more tasks and edge cases
- Add tool selection accuracy scoring (compare actual vs expected tools)
- Build CI/CD integration to run benchmarks automatically on agent changes (see L3-M1.4)
