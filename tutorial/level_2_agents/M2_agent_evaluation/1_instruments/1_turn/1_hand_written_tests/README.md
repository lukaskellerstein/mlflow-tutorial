# L2-M2.1.1.1 — Hand-Written Agent Test Suites

**Level:** AI Agents
**Duration:** 45 min

## Overview

The suite you write by hand is where agent testing starts, and it is worth
building properly before replacing any of it. This lesson builds one in full —
structured cases, pass/fail on both the answer and the tool calls, one nested
MLflow run per case, and a regression baseline — then breaks the agent on
purpose so you can watch the suite catch it.

It ends by measuring the three things this approach cannot do. Those three gaps
are lessons L2-M2.1.2.1, L2-M2.1.2.4 and L2-M2.1.4.

## Prerequisites

- Completed: L2-M1.1.1 (LangChain + LangGraph Agents), L1-M4.1.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### A test case for an agent has two halves

A model test asserts on the answer. An agent test has to assert on the **route**
as well, because half of agent failure is the right answer reached the wrong way
— or the right answer invented without calling anything.

```python
@dataclass
class TestCase:
    name: str
    input: str
    expected_output: str  # substring expected in the answer
    expected_tools: list[str]  # tools the agent should call
    difficulty: str
```

A case passes only when **both** halves pass. `test_framework.py` next to this
lesson holds the runner, the reporting and the baseline helpers.

### Nested runs give you a case-level view

`run_suite()` opens a nested run per case, so the MLflow UI shows one parent run
per suite execution with a child per case, each carrying its own `passed`,
`output_correct`, `tool_usage_correct` and `duration_s`. Sorting the children by
`duration_s` is how you find the case that quietly costs 20 seconds.

### The baseline belongs in MLflow, not on your disk

A regression check needs a previous result to compare against. Writing it to
`/tmp/baseline.json` works exactly once, on one machine. Logging it as an
artifact on the run makes it addressable by run id from anywhere that can reach
the tracking server — including CI.

```python
save_baseline(df_v1, run_v1.info.run_id)  # logs baselines/agent_test_baseline.json
baseline = load_baseline(baseline_run_id)  # downloads it back, anywhere
```

## Step-by-Step

### Step 1: Run six hand-written cases

Six cases, three difficulties, both tools. The agent has `order_status` and
`policy_lookup`.

```python
agent_v1 = build_agent([order_status, policy_lookup])
with mlflow.start_run(run_name="suite_v1_both_tools") as run_v1:
    results_v1 = AgentTestRunner(agent_v1, SUITE).run_suite()
```

### Step 2: Freeze the result as a baseline

```python
    save_baseline(df_v1, run_v1.info.run_id)
```

### Step 3: Ship a regression on purpose

`build_agent()` takes its tool list as a parameter, so v2 is the same agent with
`policy_lookup` missing — a plausible refactor accident:

```python
agent_v2 = build_agent([order_status])
```

Every policy case now fails on the tool half, and the comparison names them:

```python
baseline = load_baseline(baseline_run_id)
regressions, improvements = compare_to_baseline(df_v2, baseline)
```

### Step 4: Read the ceiling

The suite caught the regression. What it structurally cannot catch is printed at
the end, and is the reason the next three lessons exist.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/1_turn/1_hand_written_tests
uv sync
uv run python main.py
```

Expect about 3 minutes — 12 agent calls against the local model.

## Expected Output

```text
Step 1: run 6 hand-written cases against v1
  [1/6] order_shipped          PASS  (15.5s)
  ...
  passed             : 5/6  (83%)
  tool usage correct : 6/6

  faulty_return_cost called the right tool but missed 'free':
    Return shipping is covered for faulty items, so it will not cost you...
    That is the substring matcher, not the agent. A judge would pass it.

Step 3: v2 -- somebody dropped policy_lookup in a refactor
  baseline pass rate : 83%  (run 45d741a2)
  current pass rate  : 50%
  delta              : -33%

  REGRESSIONS (2):
    - return_window
    - warranty_length
```

> [!note]
> **The baseline is 5/6, not 6/6, and that is the point.** `faulty_return_cost`
> calls the right tool and gives a correct answer that does not happen to
> contain the word `free`. Substring matching cannot tell a wrong answer from a
> differently-phrased right one. Which case trips this depends on the model's
> wording, so you may see 6/6 instead — the lesson prints the diagnosis either
> way. L2-M2.1.1.2 replaces the matcher with a judge.

## Key Takeaways

- An agent test case asserts on two things: the answer and the tools used. A
  case that only checks text cannot tell a lookup from a hallucination.
- Nested runs make a suite navigable — parent per run, child per case.
- Store the regression baseline as an artifact on the run. A file on your laptop
  is not a baseline CI can use.
- Substring matching passes and fails for reasons that have nothing to do with
  agent quality.
- The real limits are structural: **single-turn**, **only what you imagined**,
  and **linear cost per case**. More cases does not fix any of them.

## Next Steps

**L2-M2.1.2.1 (Multi-Turn Conversation Simulation)** replaces the question with a
goal and a persona, and lets a second model play the user across several turns —
so failures that only appear at turn 3 become visible.
