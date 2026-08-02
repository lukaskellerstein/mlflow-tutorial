# L3-5.2 — Framework Benchmark Capstone

**Level:** Expert
**Duration:** 3 hours

## Overview

This capstone builds a reusable benchmarking system that systematically compares different agent implementation approaches using MLflow. You will benchmark a simple LLM chain, a ReAct agent, and a custom StateGraph agent on the same task set, then analyze the results to understand when each approach is the right choice. The focus is on building production-quality benchmarking infrastructure, not just running a one-off comparison.

## Prerequisites

- Completed: L3-1.3 (Architecture Comparison) -- this capstone extends and productionizes that lesson's approach
- Completed: L3-1.5 (Evaluation Pipeline) -- reusable pipeline patterns
- Completed: L2-5.1 (LangChain Agents), L2-5.2 (LangGraph Agents)
- MLflow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### From Comparison to Benchmark

In L3-1.3 you compared three architectures in a single script. This capstone elevates that into a reusable `BenchmarkSuite` class that can be applied to any set of agents and test cases. The key difference is in the design:

- **L3-1.3**: Inline comparison logic, hardcoded architectures, single-use
- **This capstone**: Pluggable agents, standardized metrics, reusable infrastructure, production artifact generation

### Benchmarking Methodology

A valid benchmark requires controlled conditions. All agents must share:

1. **The same LLM** -- ChatOpenAI with `google/gemma-4-26b-a4b` at temperature 0.0
2. **The same tools** -- identical `lookup` and `calculate` implementations
3. **The same test cases** -- 6 questions spanning three difficulty categories
4. **The same scoring functions** -- deterministic correctness and tool-usage metrics
5. **The same MLflow experiment** -- all runs are nested under a single parent for side-by-side comparison

### Three Agent Approaches

| Approach | Architecture | Tools | LLM Calls per Question |
|----------|-------------|-------|----------------------|
| Simple Chain | prompt -> LLM -> answer | None | 1 |
| ReAct Agent | reason-act loop | lookup, calculate | 1-5 (variable) |
| Custom StateGraph | classify -> route -> process -> respond | lookup, calculate (via nodes) | 2-3 |

### Test Case Categories

| Category | Description | Example |
|----------|-------------|---------|
| simple | Answerable with general knowledge, no tools needed | "What is 2 + 2?" |
| tool_required | Needs a specific tool to answer correctly | "What is Python known for?" |
| multi_step | Requires reasoning plus tool use | "What is 125 * 8?" |

### BenchmarkSuite API

The `BenchmarkSuite` class provides four methods:

- `add_agent(name, run_fn, description)` -- register any callable that takes a question string and returns a result dict
- `add_test_cases(cases)` -- add TestCase instances with expected keywords and metadata
- `run_benchmark()` -- execute all agents on all cases with three-level nested MLflow runs
- `generate_report()` -- produce a comparison table, tradeoff analysis, Pareto frontier, and per-use-case recommendations

### Results Interpretation

The benchmark produces several metrics per agent:

| Metric | What It Measures | Higher Is Better? |
|--------|-----------------|-------------------|
| Correctness | Does the answer contain the expected keyword? (0 or 1) | Yes |
| Tool Usage | Did the agent use tools appropriately? (0, 0.5, or 1) | Yes |
| Latency | Wall-clock time per question (seconds) | No |
| Quality | Average of correctness and tool usage (composite) | Yes |
| Token Efficiency | Quality per token -- quality / total_tokens * 100 | Yes |

A **Pareto-optimal** agent is one that cannot be improved on both quality and latency simultaneously. The Pareto frontier identifies which agents are worth considering for production use.

## Step-by-Step

### Step 1: Define Shared Tools and Test Cases

The benchmark uses two tools (knowledge lookup and calculator) and six test cases spanning three difficulty categories:

```python
@tool
def lookup(topic: str) -> str:
    """Look up factual information about a technology topic."""
    ...


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    ...


test_cases = [
    TestCase(question="What is 2 + 2?", expected_keyword="4", category="simple", needs_tool=False),
    TestCase(
        question="What is Python known for?",
        expected_keyword="readability",
        category="tool_required",
        needs_tool=True,
    ),
    ...,
]
```

### Step 2: Implement Three Agent Approaches

Each approach is built via a factory function that returns a callable:

- **Simple Chain**: Single LLM call with a system prompt. Fast but cannot use tools.
- **ReAct Agent**: Uses `langgraph.prebuilt.create_react_agent` with both tools. Can reason and iterate.
- **Custom StateGraph**: Builds a `StateGraph` with classify, route (conditional edge), process, and respond nodes.

### Step 3: Build the BenchmarkSuite

Register agents and test cases, then run the benchmark:

```python
suite = BenchmarkSuite()
suite.add_agent("simple_chain", build_simple_chain(), "No tools, single call")
suite.add_agent("react_agent", build_react_agent(), "ReAct with tools")
suite.add_agent("custom_stategraph", build_stategraph_agent(), "StateGraph pipeline")
suite.add_test_cases(create_test_cases())
suite.run_benchmark()
```

### Step 4: Analyze Results

The benchmark automatically generates a comparison table, identifies the Pareto frontier, and produces per-use-case recommendations. Results are logged as MLflow artifacts (CSV tables and text report).

## Running the Lesson

```bash
cd tutorial/level_3/M5_capstones/2_framework_benchmark
uv sync
uv run python main.py
```

## Expected Output

```
======================================================================
  L3-5.2 — Framework Benchmark Capstone
======================================================================

  Agents registered: 3
  Test cases loaded: 6

----------------------------------------------------------------------
  Benchmarking: simple_chain
  Prompt -> LLM -> answer (no tools, single call)
----------------------------------------------------------------------
  [PASS] Q1 (simple): What is 2 + 2?
         Correctness=1  ToolUse=1.0  Latency=0.95s
  [PASS] Q2 (simple): Say hello in French.
         Correctness=1  ToolUse=1.0  Latency=0.88s
  [FAIL] Q3 (tool_required): What is Python and what is it known for?
         Correctness=0  ToolUse=0.0  Latency=1.10s
  ...

----------------------------------------------------------------------
  Benchmarking: react_agent
  ReAct loop with tool access (langgraph prebuilt)
----------------------------------------------------------------------
  [PASS] Q1 (simple): What is 2 + 2?
         Correctness=1  ToolUse=1.0  Latency=1.50s
  ...

======================================================================
  FRAMEWORK BENCHMARK REPORT
======================================================================

  Comparison Table: Agent x Metric
  ------------------------------------------------------------------
  Agent                  Correct  ToolUse  Latency  Tokens  Quality   Effic.
  ------------------------------------------------------------------
  simple_chain             0.667    0.417    1.00s      90    0.542    0.602
  react_agent              0.833    0.833    4.20s     280    0.833    0.298
  custom_stategraph        0.833    0.667    3.10s     130    0.750    0.577

  ------------------------------------------------------------------
  Cost-Quality Tradeoff Analysis
  ------------------------------------------------------------------
  Best quality:     react_agent (score=0.833)
  Fastest:          simple_chain (latency=1.000s)
  Most efficient:   simple_chain (efficiency=0.602)

  Pareto Frontier (quality vs latency):
    * simple_chain: quality=0.542, latency=1.000s
    * react_agent: quality=0.833, latency=4.200s

  ------------------------------------------------------------------
  Recommendations
  ------------------------------------------------------------------
  - Latency-sensitive apps: use 'simple_chain' (1.00s avg)
  - Quality-critical apps:  use 'react_agent' (0.833 quality)
  - Cost-constrained apps:  use 'simple_chain' (0.602 efficiency)
  - 'simple' tasks:  best agent is 'simple_chain' (100.0% correct)
  - 'tool_required' tasks:  best agent is 'react_agent' (100.0% correct)
  - 'multi_step' tasks:  best agent is 'react_agent' (50.0% correct)

======================================================================
```

(Exact numbers will vary depending on LLM responses.)

## Key Takeaways

- A reusable `BenchmarkSuite` class makes it straightforward to compare any number of agent approaches with consistent methodology.
- Simple chains are fast and cheap but fundamentally limited by their lack of tool access -- they cannot retrieve external knowledge.
- ReAct agents achieve the highest quality through iterative reasoning and tool use, but at significantly higher latency and token cost.
- Custom StateGraph agents offer structured control flow (routing based on classification) which can be more predictable than ReAct loops, but add LLM calls even for simple questions.
- The Pareto frontier analysis helps eliminate dominated approaches and focus the decision on genuine tradeoffs.
- Three-level nested MLflow runs (benchmark -> agent -> test case) provide both high-level comparison and drill-down debugging.
- Per-category analysis reveals that the best agent depends on the task type -- there is no universally "best" approach.

## Next Steps

This is the final lesson in the tutorial. You now have production-quality infrastructure for benchmarking, evaluating, and comparing AI agents with MLflow. Use the `BenchmarkSuite` pattern as a starting point for your own agent evaluation workflows.
