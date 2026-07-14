# L3-1.2 — Agent Quality Metrics Design

**Level:** Expert
**Duration:** 2 hours

## Overview

Design and implement comprehensive quality metrics for evaluating AI agents. This lesson builds custom scorers that measure four distinct quality dimensions: task completion, tool selection accuracy, reasoning quality, and overall response quality. You will evaluate a LangGraph agent across these dimensions and statistically compare two configurations.

## Prerequisites

- Completed: L1-M4.2 (LLM Eval Basics), L1-M6.2 (Scorers & Judges), L2-M3.1 (Custom Metrics)
- MLFlow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Why Agent-Specific Metrics Matter

Standard LLM evaluation metrics (fluency, coherence, accuracy) are insufficient for agents. Agents make decisions: which tools to call, when to stop reasoning, how to combine intermediate results. Each decision point is a potential failure mode that needs its own metric.

This lesson addresses four quality dimensions:

1. **Task Completion** — Did the agent accomplish what the user asked? Supports binary (yes/no) and partial credit (0.0 to 1.0) to capture cases where the agent got close but missed details.

2. **Tool Selection Accuracy** — Did the agent pick the right tools? Measured with precision (were the chosen tools relevant?) and recall (were all needed tools used?), combined into an F1 score.

3. **Reasoning Quality** — Is the agent's chain-of-thought coherent? Assessed via LLM-as-judge, because reasoning quality is inherently subjective.

4. **Response Quality** — A composite score combining length adequacy, structural quality, and content relevance.

### Scorer Architecture

Each scorer is a function decorated with `@scorer` from `mlflow.genai.scorers`. Scorers receive `inputs`, `outputs`, and `expectations` and return a `Feedback` object containing the score value, a rationale, and the assessment source.

### Statistical Comparison

Comparing two agent configurations on a single run is unreliable. This lesson runs the same evaluation suite on two configurations (temperature=0.3 vs 0.9) and compares aggregate metrics to identify which configuration performs better on which dimensions.

## Step-by-Step

### Step 1: Build the Agent

A simple LangGraph agent is constructed with three tools:
- `calculator` — evaluates math expressions
- `dictionary_lookup` — returns word definitions
- `text_formatter` — applies text transformations

The agent uses a ReAct-style loop: call the LLM, check if it wants to use tools, execute tools, return results.

```python
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
```

### Step 2: Define the Evaluation Dataset

Six test cases cover single-tool, multi-tool, and no-tool scenarios. Each case specifies:
- The query to send to the agent
- The expected output (ground truth)
- The expected tools the agent should use
- A category label for grouping

### Step 3: Implement Custom Scorers

**task_completion_scorer**: Compares answer keywords against expected output. Full credit (1.0) for 60%+ keyword overlap, partial credit (0.5) for 30-60%, zero for less.

**tool_selection_scorer**: Computes precision, recall, and F1 of tool choices. Handles edge cases: no tools expected, tools expected but none used, etc.

**reasoning_quality_scorer**: Sends the question and answer to `google/gemma-4-26b-a4b` as a judge, requesting a 0.0-1.0 score for reasoning coherence.

**response_quality_scorer**: Combines three sub-dimensions (length, structure, relevance) into a weighted composite.

### Step 4: Run Evaluation with `mlflow.genai.evaluate()`

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[task_completion_scorer, tool_selection_scorer,
             reasoning_quality_scorer, response_quality_scorer],
)
```

### Step 5: Compare Configurations

Two agent configurations (temperature 0.3 and 0.9) are evaluated with identical test cases. Aggregate metrics are compared per-scorer, and an overall winner is declared.

## Running the Lesson

```bash
cd tutorial/level_3/M1_agent_evaluation/2_quality_metrics
uv sync
uv run python main.py
```

## Expected Output

The script prints three sections:

1. **Config A report** — per-case scores and aggregate metrics for temperature=0.3
2. **Config B report** — per-case scores and aggregate metrics for temperature=0.9
3. **Statistical comparison** — side-by-side metric comparison with delta and winner

You will also see two evaluation runs in the MLflow UI under the experiment `L3/M1_agent_evaluation/2_quality_metrics`, each with logged parameters and metrics.

## Key Takeaways

- Agent evaluation requires domain-specific metrics beyond standard NLP quality scores.
- Tool selection accuracy (precision/recall/F1) captures whether the agent made correct decisions about which tools to invoke.
- LLM-as-judge scorers handle subjective dimensions (reasoning quality) that deterministic metrics cannot.
- Composite scorers combine multiple sub-dimensions with explicit weights, making the scoring criteria transparent and tunable.
- Statistical comparison of configurations reveals which quality dimensions are affected by hyperparameter changes.

## Next Steps

In L3-1.3 (Agent Architecture Comparison), you will extend this metrics framework to systematically compare different agent architectures (ReAct vs. Plan-and-Execute) using controlled evaluation methodology.
