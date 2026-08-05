# L2-M2.2.2 — Offline Gates and Regression Detection

**Level:** AI Agents
**Duration:** 90 min

## Overview

The lesson where offline evaluation becomes a pipeline you can put in CI. It
answers exactly one question — *is this version good enough to ship?* — from a
curated dataset with known expectations, scoring every case, on a trigger **you**
pull.

- **Part 1** builds the pipeline: dataset → agent → score → quality gates → report.
- **Part 2** adds regression detection against a stored baseline.

## Prerequisites

- Completed: L2-M2.1.1 (Test Generation), L2-M2.1.2 (Judges), L2-M2.1.3 (Quality Metrics)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)

## The four axes

This lesson lives entirely in the left-hand column. The right-hand column is
[L2-M2.3.1](../../3_online/1_online_scoring/), and the contrast is the reason M2
is split into groups at all.

| | offline (here) | online (M2.3.1) |
|:--|:--|:--|
| **input** | curated dataset | production traces |
| **ground truth** | expectations | none |
| **coverage** | every case | sampled |
| **trigger** | you, in CI | the server, on a schedule |

Neither replaces the other. Offline evaluation cannot tell you that real users
ask things your dataset never imagined; online scoring cannot tell you whether a
change is safe *before* you ship it.

## Concepts

### Evaluation Pipelines

An evaluation pipeline is a repeatable, automated sequence of steps that measures agent quality. Unlike ad-hoc evaluation, a pipeline enforces consistency: the same dataset, scorers, and thresholds are applied every time, making results comparable across runs.

The pipeline in this lesson has five stages:

1. **Load Dataset** — Create or load a structured evaluation dataset with inputs, expected outputs, and metadata (category, expected tool usage).
2. **Run Agent** — Execute the agent on every input, capturing outputs, latency, tool calls, and errors.
3. **Score Results** — Apply deterministic scorers to measure accuracy (keyword match) and tool selection correctness.
4. **Check Quality Gates** — Compare scored metrics against minimum thresholds. If any gate fails, the pipeline flags the build as failing.
5. **Generate Report** — Produce a human-readable report summarizing results, gate outcomes, and per-test breakdowns.

### Quality Gates

Quality gates are minimum thresholds that an agent must meet before it can be promoted or deployed. Common gates include:

| Gate | Description | Example Threshold |
|------|-------------|-------------------|
| Accuracy | Does the agent produce correct answers? | >= 60% |
| Tool Accuracy | Does the agent select the right tools? | >= 50% |
| Avg Latency | Is the agent fast enough for production? | <= 120s |
| Error Rate | Does the agent crash on valid inputs? | <= 10% |

In CI/CD, a failed quality gate blocks deployment — the same way a failing test suite blocks a merge.

### Regression Detection

When you update an agent (new model, changed prompt, modified tools), you need to know if quality improved or degraded. Regression detection compares current evaluation metrics against a stored baseline and flags any metric that dropped beyond a configurable tolerance.

This lesson simulates a baseline to demonstrate the pattern. In production, you would:
1. Store baseline metrics from the last "known-good" evaluation run in MLflow.
2. After each new evaluation, compare current metrics to the baseline.
3. Alert or block deployment if regressions exceed the threshold.

## Step-by-Step

### Step 1: Build the Agent Under Test

We create a ReAct agent with two tools — `search_knowledge` (knowledge base lookup) and `calculate` (math evaluation). This gives us a concrete subject to evaluate.

```python
@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information on a topic."""
    ...


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    ...


def build_agent():
    llm = ChatOpenAI(
        model="gemma-large",  # a LiteLLM alias, not a provider model id
        base_url="http://localhost:4000/v1",
        api_key=SecretStr("sk-litellm-master"),
        temperature=0.0,
    )
    return create_agent(llm, [search_knowledge, calculate])
```

### Step 2: Define the Evaluation Dataset

The dataset includes 6 test cases across two categories (knowledge and math), each specifying which tool should be used:

```python
(
    {
        "input": "What is Python?",
        "expected": "high-level programming language",
        "category": "knowledge",
        "needs_tool": "search_knowledge",
    },
)
({"input": "What is 25 * 4?", "expected": "100", "category": "math", "needs_tool": "calculate"},)
```

### Step 3: The Pipeline Class

`AgentEvaluationPipeline` encapsulates the full workflow. Each method handles one stage and logs results to MLflow:

- **Nested runs**: Each test case gets its own nested MLflow run, making it easy to drill into individual results in the UI.
- **Artifacts**: The dataset, results table, and text report are all logged as artifacts.
- **Tags**: Quality gate pass/fail status is logged as tags for easy filtering.

### Step 4: Quality Gates

After scoring, the pipeline checks each metric against its threshold:

```python
@dataclass
class QualityGates:
    min_accuracy: float = 0.6
    min_tool_accuracy: float = 0.5
    max_avg_latency_s: float = 120.0
```

### Step 5: Regression Detection

A second MLflow run compares current metrics against a simulated baseline and logs any regressions found:

```python
regressions = detect_regressions(current_metrics, baseline, threshold=0.1)
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/2_offline_gates
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
  Agent Evaluation Pipeline — Starting
============================================================

  [Step 1/5] Loading evaluation dataset...
    Loaded 6 test cases (2 categories)

  [Step 2/5] Running agent on test cases...
    [OK] Test 1/6: What is Python?...                (8.2s)
    [OK] Test 2/6: What is MLflow used for?...       (7.5s)
    ...

  [Step 3/5] Scoring results...
    Accuracy:      83.3% (5/6)
    Tool accuracy: 66.7% (4/6)
    Avg latency:   8.1s
    Error rate:    0.0%

  [Step 4/5] Checking quality gates...
    [PASS] accuracy: 0.833 (threshold: 0.6)
    [PASS] tool_accuracy: 0.667 (threshold: 0.5)
    [PASS] avg_latency: 8.100 (threshold: 120.0)

    Overall: ALL GATES PASSED

  [Step 5/5] Generating evaluation report...
  ...

============================================================
  Part 2: Regression Detection
============================================================
  Simulated baseline (previous run):
    accuracy: 1.000
    ...
  REGRESSIONS DETECTED (1):
    accuracy: 1.000 -> 0.833 (delta: -0.167)
```

In the MLflow UI at <http://127.0.0.1:5555>, you will see:
- An `evaluation_pipeline` parent run with aggregate metrics and artifacts.
- Nested `test_1` through `test_6` runs with per-test parameters and tags.
- A `regression_check` run with baseline vs. current comparison.
- Artifacts: `datasets/eval_dataset.csv`, `results/eval_results.csv`, `reports/eval_report.txt`, `reports/regression_report.json`.

## Key Takeaways

- An evaluation pipeline makes agent quality measurement repeatable and automated — the same process runs in development, CI, and production.
- Quality gates translate subjective "is this agent good enough?" into objective, enforceable thresholds.
- Regression detection catches degradations early, before they reach users — compare every new evaluation against a stored baseline.
- MLflow nested runs and artifacts provide full traceability: you can drill from an aggregate "pipeline failed" down to the exact test case that caused the failure.
- This pipeline pattern maps directly to CI/CD: run it in GitHub Actions, fail the build on gate violations, and store the baseline for the next comparison.

## Next Steps

This completes the Agent Evaluation module (L3-M1). Continue to L3-M2 (Custom Framework Integrations) to learn how to integrate non-LangChain agent frameworks — Claude Agent SDK, Codex SDK, and DeepAgents — with MLflow's evaluation and tracing infrastructure.
