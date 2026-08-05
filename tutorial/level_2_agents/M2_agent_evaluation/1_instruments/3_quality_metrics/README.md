# L2-M2.1.3 — Agent Quality Metrics and Session Scorers

**Level:** AI Agents
**Duration:** 90 min

## Overview

Design comprehensive quality metrics for AI agents in two passes. **Part 1**
builds four custom single-turn scorers — task completion, tool selection
accuracy, reasoning quality, response quality — and statistically compares two
agent configurations. **Part 2** adds the dimension none of them can express:
session-level scorers that judge a whole multi-turn conversation.

## Prerequisites

- Completed: L2-M2.1.1 (Test Generation and Simulation), L2-M2.1.2 (Judges), L1-M4.1.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in `infra/.env` — the `gemma-large` alias routes there

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

## Part 2 — Session-level scorers

Every scorer in Part 1 sees exactly one question and one answer. Agents are
multi-turn, and the interesting failures are conversational:

- the agent that answers each question adequately while never completing the task
- the agent that forgets on turn 4 what the user told it on turn 1
- the user who is visibly getting more frustrated with every reply

None of those is expressible as a function of `(inputs, outputs)`. MLflow ships
**session-level scorers** for exactly this, and they take a different argument:

```python
SESSION_SCORERS = [
    ConversationCompleteness(model=SESSION_MODEL),
    UserFrustration(model=SESSION_MODEL),
    KnowledgeRetention(model=SESSION_MODEL),
    ConversationalToolCallEfficiency(model=SESSION_MODEL),
]
feedback = session_scorer(session=traces)  # not inputs=/outputs=
```

`Scorer.is_session_level_scorer` distinguishes them, and they reject the
single-turn parameters outright.

The scripted conversation ends with **"Remind me what the total was again?"** —
deliberately, without restating the number. That is the only way to see whether
the agent retained anything, and it is invisible to every scorer in Part 1.

### Two traps worth knowing

**1. Every trace in a session needs a `session_id`.** Collecting traces is not
enough — the scorer validates it and raises
`All traces in 'session' must have a session_id`. Wrap each turn so you own the
root span, then stamp it:

```python
@mlflow.trace(name="conversation_turn")
def run_turn(agent, history, session_id):
    mlflow.update_current_trace(session_id=session_id)
    return agent.invoke({"messages": history})
```

Traces are also exported asynchronously, so `mlflow.flush_trace_async_logging()`
before reading them back — otherwise you race the exporter.

**2. These judges answer with the string `"yes"` or `"no"`.** Not a float, not a
bool. Coercing with `bool(value)` scores every failure as a pass, because
`bool("no")` is `True`. The lesson normalises against an explicit whitelist:

```python
if isinstance(value, str):
    return 1.0 if value.strip().lower() in TRUTHY else 0.0
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/3_quality_metrics
uv sync
uv run python main.py
```

> [!note]
> Expect 10–15 minutes. Two agent configurations over six cases each, one of the
> four scorers being an LLM judge, then four session judges over a four-turn
> conversation.

## Expected Output

The script prints four sections:

1. **Config A report** — per-case scores and aggregate metrics for temperature=0.3
2. **Config B report** — the same for temperature=0.9
3. **Statistical comparison** — side-by-side metrics with delta and winner
4. **Session-level scoring** — the four session judges over one conversation

```text
Part 2: session-level scorers (multi-turn)
  driving a 4-turn conversation through the agent...
  captured 4 traces, one per turn

    scorer                             value    rationale
    ---------------------------------- -------- ---------------------------
    conversation_completeness          yes      The user made the following explicit requests: ...
    user_frustration                   none     The user is asking several different questions ...
    knowledge_retention                yes      Knowledge retention across 4 turn(s): - Turn 1: ...
    conversational_tool_call_efficiency no      The assistant used the `dictionary_lookup` tool
                                                multiple times ...
```

**Read that last row against Part 1.** The single-turn `tool_selection_scorer`
gave this same agent **1.000** — every tool it called was the right tool. The
session scorer disagrees, because calling the right tool *repeatedly for the same
thing* is only visible once you look at the whole conversation. That is the
entire argument for session-level scoring, and it shows up on the first run.

> [!warning]
> **These scorers do not share a value vocabulary or a polarity.** This run
> returned `yes`, `none` and `no` from three different scorers. Worse,
> "good" is not always `1.0`:
>
> | scorer | good answer | numeric |
> |:--|:--|--:|
> | `conversation_completeness` | `yes` | 1.0 |
> | `knowledge_retention` | `yes` | 1.0 |
> | `conversational_tool_call_efficiency` | `yes` | 1.0 |
> | `user_frustration` | `none` | **0.0** |
>
> For `user_frustration`, zero is the *good* outcome. Never sum these into a
> single "session score" or rank configurations by their total — you would be
> adding a metric that improves as it falls to three that improve as they rise.
> The lesson logs each one separately under `session/<name>` for this reason.

In the MLflow UI, experiment `L2/M2_agent_evaluation/1_instruments/3_quality_metrics` holds the
two config runs plus a `session_level_scoring` run whose metrics are prefixed
`session/`.

## Key Takeaways

- Agent evaluation requires domain-specific metrics beyond standard NLP quality scores.
- Tool selection accuracy (precision/recall/F1) captures whether the agent made correct decisions about which tools to invoke.
- LLM-as-judge scorers handle subjective dimensions (reasoning quality) that deterministic metrics cannot.
- Composite scorers combine multiple sub-dimensions with explicit weights, making the scoring criteria transparent and tunable.
- **Single-turn scorers have a structural blind spot.** Task completion across a
  conversation, retention, and user frustration are properties of the session,
  not of any one turn.
- Session scorers need `session=list[Trace]` where every trace carries a
  `session_id` — and they answer in words, not numbers.

## Next Steps

**L2-M2.2.1 (Agent Architecture Comparison)** reuses this metric suite to compare
agent architectures under a controlled methodology, scoring through
`mlflow.genai.evaluate()` with the registered judges from L2-M2.1.2 so results stay
comparable outside the script that produced them.
