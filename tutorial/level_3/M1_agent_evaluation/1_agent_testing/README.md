# L3-1.1 — Agent Testing Framework

**Level:** Expert
**Duration:** 2 hours

## Overview

Build a production-quality testing framework for LangGraph agents. You will create structured test suites, run automated pass/fail checks on agent outputs and tool usage, log every result to MLflow with nested runs, and establish regression baselines so future agent changes can be compared against known-good behavior.

## Prerequisites

- Completed: L2-5.2 (LangGraph Agent Observability)
- Completed: L2-3.1 (Custom Metrics)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why test agents?

Agents are non-deterministic systems. A small change to the prompt, model, or tools can silently degrade quality. Unlike traditional software where unit tests cover deterministic functions, agent testing must handle variable outputs, optional tool usage, and reasoning quality.

A proper agent testing framework gives you:

- **Functional correctness**: Does the agent produce the right answer?
- **Tool selection accuracy**: Does it pick the right tools for each task?
- **Regression detection**: Did a change break something that used to work?
- **Performance tracking**: Is the agent getting slower over time?

### Test case design

Each test case specifies:

| Field | Purpose |
|---|---|
| `name` | Unique identifier for the test |
| `input` | The user query sent to the agent |
| `expected_output` | Substring expected in the agent's response |
| `expected_tools` | List of tools the agent should invoke |
| `difficulty` | easy / medium / hard (for stratified reporting) |

Good test suites cover: simple cases (sanity checks), multi-tool scenarios, edge cases (no tools needed), and varying difficulty levels.

### Nested runs for test tracking

The framework uses MLflow nested runs:

```
agent_test_suite (parent run)
  +-- test_simple_addition (child run)
  +-- test_multiplication (child run)
  +-- test_complex_expression (child run)
  ...
```

The parent run holds aggregate metrics (pass rate, average duration). Each child run logs the individual test result, parameters, and pass/fail status.

### Regression baselines

After a test run, results are saved as a JSON baseline artifact. On subsequent runs, you can load the baseline and compare: which tests regressed (were passing, now failing)? Which improved? This is the foundation for CI/CD quality gates in L3-3.4.

## Step-by-Step

### Step 1: Build the agent

We create a LangGraph ReAct agent with two tools — `calculator` for math and `text_analyzer` for text statistics. The agent uses `gemma4:e2b` with `temperature=0.0` for maximum determinism during testing.

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    ...

@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return word count, char count, etc."""
    ...

agent = create_agent(
    ChatOllama(model="gemma4:e2b", temperature=0.0),
    tools=[calculator, text_analyzer],
)
```

### Step 2: Define the test suite

Test cases are structured dataclasses with clear expected outputs:

```python
TestCase(
    name="simple_addition",
    input="What is 25 + 37?",
    expected_output="62",
    expected_tools=["calculator"],
    difficulty="easy",
)
```

The suite includes 8 cases covering easy arithmetic, complex expressions, text analysis, multi-tool usage, and a no-tool scenario.

### Step 3: Run the automated test suite

The `AgentTestRunner` class executes each test case, extracts tool calls from the message history, checks output correctness (substring matching), and verifies tool usage. Each result is logged to MLflow as a nested child run.

```python
runner = AgentTestRunner(agent, TEST_SUITE)
results = runner.run_suite()
```

For each test the runner logs:
- **Parameters**: test name, difficulty, expected tools, input
- **Metrics**: passed (0/1), output_correct (0/1), tool_usage_correct (0/1), duration
- **Tags**: status (PASS/FAIL), difficulty level

### Step 4: Save and compare baselines

Results are saved as a JSON baseline artifact under `baselines/` in the MLflow run. The `compare_to_baseline()` function loads a previous baseline and reports regressions and improvements per test case.

```python
baseline_path = save_baseline(df, run_id)
compare_to_baseline(current_df, baseline_path)
```

## Running the Lesson

```bash
cd tutorial/level_3/M1_agent_evaluation/1_agent_testing
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-1.1 — Agent Testing Framework
============================================================

--- Part 1: Building LangGraph ReAct agent ---
  Agent created with tools: ['calculator', 'text_analyzer']

--- Part 2: Test suite (8 cases) ---
  [easy  ] simple_addition: What is 25 + 37?
  [easy  ] multiplication: Calculate 12 * 15.
  ...

--- Part 3: Running automated test suite ---
  [1/8] Running: simple_addition ... PASS  (3.2s)
  [2/8] Running: multiplication ... PASS  (2.8s)
  ...

============================================================
  Test Suite Summary
============================================================
  Total tests:           8
  Passed:                6/8  (75%)
  Output correct:        7/8
  Tool usage correct:    7/8
  Average duration:      3.15s
  ...

--- Part 4: Regression baseline ---
  Baseline saved: /tmp/agent_test_baseline.json
  ...
```

In the MLflow UI, navigate to the `L3/M1_agent_evaluation/1_agent_testing` experiment to see the parent `agent_test_suite` run with nested child runs for each test case. Each child run shows pass/fail metrics and test parameters.

## Key Takeaways

- Agent testing requires structured test cases with expected outputs AND expected tool usage — checking only the final answer misses tool-selection bugs.
- Nested MLflow runs provide clean organization: one parent per test suite, one child per test case, with aggregate metrics on the parent.
- Regression baselines let you detect quality regressions when you change the model, prompt, or tools — save them as MLflow artifacts for traceability.
- Setting `temperature=0.0` during testing reduces flakiness but does not eliminate it — LLM outputs are inherently non-deterministic.
- Substring matching is a simple but effective correctness check for factual outputs; L3-1.2 introduces LLM-as-judge for more nuanced quality assessment.

## Next Steps

In **L3-1.2 (Agent Quality Metrics Design)**, you will build custom scorers for agent-specific behaviors — task completion rate with partial credit, tool selection precision/recall, and reasoning quality assessment using LLM judges. These metrics replace the simple pass/fail checks used here with nuanced, production-grade quality measurement.
