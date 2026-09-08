# L2-M2.1.1.3 — Turn-Level Quality Metrics

**Level:** AI Agents
**Duration:** 90 min

## Overview

Everything here judges **one turn** — one request, one final answer. That is a
boundary in time, not blindness to the steps: a turn scorer can see every tool
call inside that turn, it simply cannot see the next turn.

**Part 1** builds four custom scorers — task completion, tool selection
accuracy, reasoning quality, response quality — and compares two agent
configurations. **Part 2** hands two built-in judges the **trace** instead of a
flattened dict, which is the only way to see a tool's arguments or a repeated
call.

Judging a whole conversation is a different scope, and it lives in
[`../../2_conversation/3_judging_conversations`](../../2_conversation/3_judging_conversations/).

## Prerequisites

- Completed: L2-M2.1.1.1 to L2-M2.1.3.1 (the test material), L2-M2.1.1.2 (Judges),
  L1-M4.1.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>
- An `UNSLOTH_API_KEY` in the environment — every alias, `gemma-chat` included, routes to Unsloth

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

**reasoning_quality_scorer**: An INLINE judge. Sends the question and answer to the `gemma-judge` alias and asks for a 0.0-1.0 score on reasoning coherence. A lesson never names a raw model key — the alias is resolved in `infra/mlflow/gateway/seed_gateway.py`.

**response_quality_scorer**: Combines three sub-dimensions (length, structure, relevance) into a weighted composite.

### Step 4: Run Evaluation with `mlflow.genai.evaluate()`

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        task_completion_scorer,
        tool_selection_scorer,
        reasoning_quality_scorer,
        response_quality_scorer,
    ],
)
```

### Step 5: Compare Configurations

Two agent configurations (temperature 0.3 and 0.9) are evaluated with identical test cases. Aggregate metrics are compared per-scorer, and an overall winner is declared.

## Part 2 — Scorers that read the trace

Every scorer above reads a **flattened dict**. `run_agent()` puts a list of tool
NAMES into `outputs["tools_used"]`, and that is all `tool_selection_scorer` ever
sees. Two things are already gone by then: the arguments each tool was called
with, and whether the same call was made twice.

MLflow's built-in tool scorers take the **trace** instead:

```python
TRACE_SCORERS = [
    ToolCallCorrectness(model=BUILTIN_MODEL),
    ToolCallEfficiency(model=BUILTIN_MODEL),
]
data = pd.DataFrame({"trace": [mlflow.get_trace(tid) for tid in trace_ids]})
results = mlflow.genai.evaluate(data=data, scorers=TRACE_SCORERS)
```

Both take `model=`, so both are **judges**. `ToolCallCorrectness` runs
ground-truth-free by default: with no expectations it asks whether the calls
made were reasonable for the request.

### Getting a trace per turn

Wrap the call so you own the root span, then flush before reading back:

```python
@mlflow.trace(name="agent_turn")
def run_agent_traced(agent, query):
    return run_agent(agent, query)


mlflow.flush_trace_async_logging()  # traces export asynchronously
```

> [!warning]
> **A mean over 5 of 6 cases is not a mean over 6.** A small local model
> sometimes fails a judge's structured-output parser. MLflow warns
> (`Some scorer invocations failed during evaluation`) and drops that case from
> the mean rather than scoring it zero. Read the count before comparing two runs.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/1_turn/3_quality_metrics
uv sync
uv run python main.py
```

> [!note]
> Expect 10–15 minutes. Two agent configurations over six cases each, one of the
> four scorers being an LLM judge, then six cases re-run under tracing for the
> two built-in judges.

## Expected Output

The script prints four sections:

1. **Config A report** — per-case scores and aggregate metrics for temperature=0.3
2. **Config B report** — the same for temperature=0.9
3. **Comparison** — side-by-side metrics with delta and winner
4. **Scope B** — the two trace-reading judges over the same six cases

```text
  Metric                                   temp=0.3   temp=0.9      Delta     Winner
  -------------------------------------- ---------- ---------- ---------- ----------
  reasoning_quality                           0.950      0.917     -0.033   temp=0.3
  response_quality                            0.814      0.747     -0.067   temp=0.3
  task_completion                             1.000      0.833     -0.167   temp=0.3
  tool_selection                              1.000      1.000     +0.000        TIE

  temp=0.3 wins 3, temp=0.9 wins 0, ties 1
```

```text
  Scope B: scorers that read the trace
  re-running 6 cases under an explicit root span...
  captured 6 traces, evaluating with 2 built-in judges...

  Aggregate Metrics:
    tool_call_correctness/mean: 1.000
    tool_call_efficiency/mean: 1.000
```

**Read `tool_selection` against `tool_call_efficiency`.** They can agree, as they
do above, and they are not measuring the same thing. `tool_selection` compares a
list of tool names against a list you wrote. `tool_call_efficiency` reads the
spans, so it can see that the same tool was called twice with the same arguments
— something a list of names cannot express.

In the MLflow UI, experiment
`L2/M2_agent_evaluation/1_instruments/1_turn/3_quality_metrics` holds the two
config runs plus a `trace_based_scorers` run.

## Key Takeaways

- Agent evaluation needs domain-specific metrics beyond standard NLP scores.
- Tool selection precision/recall/F1 captures whether the agent chose correctly.
- A **judge** is a scorer that decides with an LLM. The test is whether it takes
  `model=`: `tool_selection_scorer` computes, `ToolCallCorrectness` decides.
- Composite scorers put the weights in code, so your definition of "good" is
  readable and tunable rather than buried in a prompt.
- **A flattened `outputs` dict throws away the arguments and the repeats.** If a
  scorer needs either, give it the `trace`.
- A judge that failed to parse its own answer is dropped from the mean, not
  scored zero — so check the count before comparing runs.
- **Every scorer here is blind to the next turn.** Task completion across a
  conversation, retention and user frustration are properties of the session.

## Next Steps

**L2-M2.2.1.1 (Agent Architecture Comparison)** reuses this metric suite to compare
agent architectures under a controlled methodology, scoring through
`mlflow.genai.evaluate()` with the registered judges from L2-M2.1.1.2 so results stay
comparable outside the script that produced them.
