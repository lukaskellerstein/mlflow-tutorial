# L2-M2.2.5 — Building a Custom Domain-Specific Agent Benchmark

**Level:** Agents
**Duration:** 90 min

## Overview

Established benchmarks like SWE-Bench and GAIA test general capabilities, but your agents operate in specific domains with unique requirements. This lesson teaches the methodology of designing, building, and running your own agent benchmark — using a customer support agent built on the Claude Agent SDK as the example domain. You'll build a reusable harness pattern that works for any domain.

## Prerequisites

- Completed: L2-M2.2.3 (SWE-Bench) and L2-M2.2.4 (GAIA) for benchmark harness patterns
- Completed: L2-M1.3 (Claude Agent SDK) for SDK basics
- MLflow server running at <http://127.0.0.1:5555>
- Claude Code CLI installed and authenticated (the SDK uses your existing login — LMStudio is not involved)

> **Cost note:** the demo run (6 tasks x 2 configurations) calls real Claude models and costs about $0.20 total; every task is capped with `max_budget_usd`.

## Concepts

### Why Build a Custom Benchmark?

General benchmarks tell you if an agent can reason and use tools. They don't tell you if your customer support agent correctly applies your refund policy, or if your coding agent follows your team's style guide. Domain-specific benchmarks measure what matters for your use case.

### Benchmark Design Methodology

1. **Define a task taxonomy** — categorize the types of tasks your agent handles, with difficulty levels
2. **Create an evaluation dataset** — input/expected output pairs covering each category
3. **Choose metrics** — accuracy, tool selection, latency, cost, domain-specific scores
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

### Tool Selection Accuracy

Because the taxonomy declares `required_tools` per category and the SDK streams every `ToolUseBlock` with its exact tool name, the harness can score **tool recall** — the fraction of expected tools the agent actually called. This is a first-class agent metric that pure answer-matching misses: an agent can luck into the right answer while skipping the policy check it was supposed to run.

## Step-by-Step

### Step 1: Define the Task Taxonomy

We define three task categories for a customer support agent, each with increasing difficulty:

```python
TASK_TAXONOMY = {
    "order_lookup": {"difficulty": 1, "required_tools": ["check_order"]},
    "refund_request": {"difficulty": 2, "required_tools": ["check_order", "check_refund_policy"]},
    "complex_complaint": {
        "difficulty": 3,
        "required_tools": ["check_order", "check_refund_policy", "escalate_ticket"],
    },
}
```

### Step 2: Create the Evaluation Dataset

Six tasks covering all categories and difficulty levels. Each has `task_id`, `category`, `difficulty`, `input`, `expected_answer`, and `expected_tools`.

### Step 3: Build Domain-Specific Tools

Three in-process MCP tools that simulate a real support system: `check_order` (database lookup), `check_refund_policy` (policy engine), `escalate_ticket` (handoff to human). They are served by `create_sdk_mcp_server` and the agent is locked to exactly this tool set (`tools=[]`, `strict_mcp_config=True`, `setting_sources=[]` — see L2-M2.2.3).

### Step 4: Run the Benchmark Harness

Each task runs in a nested MLflow run, traced with `@mlflow.trace` plus a child span per tool call. The model is fixed to `claude-sonnet-5` and we compare two effort levels — `low_effort` vs `high_effort` (the SDK has no temperature knob; effort is its speed/quality/cost axis).

### Step 5: Statistical Analysis

Break down accuracy by category and difficulty level. Compute latency statistics (mean, stdev, min, max) and total cost per configuration. This reveals which task types the agent struggles with — and what improving them costs.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/5_custom_benchmark
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
L2-M2.2.5 -- Custom Domain-Specific Agent Benchmark
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
Config: low_effort  (model=claude-sonnet-5, effort=low)
============================================================
  [OL-001] D1 correct=True tool_recall=1.00 latency=5.3s
  [RR-001] D2 correct=True tool_recall=1.00 latency=6.5s
  [CC-001] D3 correct=True tool_recall=1.00 latency=11.4s
  ...

--- Overall Results ---
             accuracy  tool_recall  avg_latency  total_cost_usd
config
high_effort  0.833333          1.0     8.331667        0.109918
low_effort   0.833333          1.0     7.461667        0.098625

--- Accuracy by Category ---
category     complex_complaint  order_lookup  refund_request
config
high_effort                1.0           1.0             0.5
low_effort                 1.0           1.0             0.5
```

In MLflow UI: benchmark definition artifact (taxonomy + dataset JSON), per-config CSV results, category-level accuracy metrics, and traces with `tool_call.*` spans.

Note the `refund_request` miss: RR-002 expects the literal phrase "within policy", which the agent rarely says verbatim — a deliberate reminder that exact-match scoring is only as good as the expected answers you write.

## Key Takeaways

- Domain-specific benchmarks measure what general benchmarks can't — your agent's fit for your use case
- A good taxonomy maps to real user intents and has clear difficulty levels
- The harness pattern (taxonomy → dataset → tools → runner → analysis) is reusable across any domain and any agent framework
- Tool recall catches failures that answer-matching misses — right answer, wrong process
- Per-category and per-difficulty breakdowns reveal specific failure modes
- Version your benchmark definition as an MLflow artifact for reproducibility

## Next Steps

You've now completed the Agent Benchmarks module. You can:

- Expand your custom benchmark with more tasks and edge cases
- Tighten the scoring — replace exact-match with an LLM judge (L2-M2.1.2) for phrasing-tolerant evaluation
- Build CI/CD integration to run benchmarks automatically on agent changes (see L3-M1.4)
