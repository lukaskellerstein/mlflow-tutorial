# L2-M2.2.1 — Agent Architecture Comparison

**Level:** AI Agents
**Duration:** 90 min

## Overview

Different agent architectures make fundamentally different tradeoffs between
quality, latency, cost, and complexity. This lesson runs three architectures on
one benchmark, scores them all with **one registered judge** through
`mlflow.genai.evaluate()`, and identifies the Pareto-optimal designs — those that
cannot be improved on one axis without sacrificing another.

## Prerequisites

- Completed: L2-M2.1.1 (Test Generation), L2-M2.1.2 (Judges), L2-M2.1.3 (Quality Metrics)
- Completed: L2-M1.1 (LangChain + LangGraph Agents)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in `infra/.env` — the `gemma-chat` alias routes there

## The yardstick is the point

A comparison is only as trustworthy as the thing doing the measuring, and the
obvious yardstick — does the expected keyword appear in the answer? — is a bad
one in both directions:

- *"Python is known for its clear, readable syntax"* scores **0** against the
  keyword `readability`, though it is plainly correct.
- An answer that contains the word while saying something false scores **1**.

So correctness here is judged by a **registered judge**, created with
`make_judge` and stored server-side. That matters specifically for comparison:
a registered judge is a named, versioned object, so "react_agent scored 0.83"
stays meaningful outside the script that produced it, and L2-M2.2.2 can put the
very same judge on production traffic.

```python
def get_or_register_judge():
    try:
        return mlflow.genai.get_scorer(name=JUDGE_NAME)  # reuse across runs
    except Exception:
        judge = mlflow.genai.make_judge(name=JUDGE_NAME, instructions=..., model=...)
        return judge.register(name=JUDGE_NAME)
```

Tool usage stays a local `@scorer` — it is deterministic, needs no model, and a
`@scorer` cannot be registered against an open-source server anyway (L2-M2.1.2).

> [!warning]
> **Reading scores back out of `result_df` has a trap that fails silently.** The
> frame is pandas, so a boolean judge verdict arrives as `np.True_` — a *numpy*
> bool, which is **not** a Python `bool`, and under numpy 2.x not an `int`
> either. A normaliser written as
>
> ```python
> if isinstance(value, bool): ...      # misses np.True_
> if isinstance(value, (int, float)): ...  # also misses it
> ```
>
> falls through every branch and returns the default. The first run of this
> lesson scored **every architecture 0.0 on every case** — including
> `"2 + 2 is 4."` against the keyword `4` — with no error, no warning, and a
> perfectly plausible-looking comparison table at the end. Unwrap numpy scalars
> with `.item()` before type-checking. A comparison that silently scores
> everything zero looks exactly like three architectures that are all bad.

## Concepts

### Why Compare Architectures?

Choosing an agent architecture is rarely about picking "the best" one. Instead, it is about finding the right tradeoff for your use case:

- A **simple chain** (prompt -> LLM -> answer) is fast and cheap but cannot use tools or reason iteratively.
- A **ReAct agent** can reason and use tools but costs more tokens and has higher latency due to its reasoning loop.
- A **multi-step pipeline** (classify -> process -> respond) provides structured control flow but adds complexity and LLM calls even when they are not needed.

### Controlled Evaluation Methodology

A valid architecture comparison requires:

1. **Same task** -- all architectures answer the same questions
2. **Same model** -- all use the same LLM (google/gemma-4-26b-a4b) at the same temperature
3. **Same tools** -- all have access to identical tool implementations
4. **Same metrics** -- all are scored with the same evaluation functions
5. **Same environment** -- all run in the same session, logged to the same MLflow experiment

Without these controls, you cannot attribute performance differences to the architecture itself.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Correctness | Does the answer contain the expected keyword? (0 or 1) |
| Tool Usage | Did the architecture use tools appropriately? (0, 0.5, or 1) |
| Latency | Wall-clock time per question (seconds) |
| Token Efficiency | Quality per token -- higher means better answers with fewer tokens |

### The Pareto Frontier

An architecture is **Pareto-optimal** if no other architecture is better on every metric simultaneously. The Pareto frontier is the set of all such non-dominated architectures. When choosing an architecture for production, you pick from this frontier based on your priorities (e.g., "I need the lowest latency that still has > 0.8 correctness").

## Step-by-Step

### Step 1: Define Shared Tools and Dataset

All three architectures share a knowledge lookup tool and a word count tool, plus a 5-question evaluation dataset with expected answers:

```python
@tool
def lookup(topic: str) -> str:
    """Look up factual information about a technology topic."""
    ...


EVAL_DATASET = [
    {"question": "What is Python?", "expected_keyword": "readability", "needs_tool": True},
    ...,
]
```

### Step 2: Implement Three Architectures

1. **Simple Chain** -- Direct LLM call with a system message. No tool access.
2. **ReAct Agent** -- `langchain.agents.create_agent` with the shared tools.
3. **Multi-step Pipeline** -- A `StateGraph` with classify, process, and respond nodes.

### Step 3: Evaluate Each Architecture

Each architecture runs on all 5 test cases. Per-case metrics (correctness, tool usage, latency, token estimate) are collected and logged as nested MLflow runs under a parent `architecture_comparison` run.

### Step 4: Build Comparison Table and Pareto Analysis

Aggregate metrics are computed per architecture using pandas, and a comparison table is printed. The script identifies the Pareto frontier -- architectures that are not dominated on quality vs. latency.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/2_offline/1_architecture_comparison
uv sync
uv run python main.py
```

> [!note]
> Expect 8–12 minutes: three architectures over five cases each, plus a judge
> call per case.

## Expected Output

```text
  registered judge 'answer_correctness' (first run)

  Evaluating: simple_chain
  [PASS] Q1: What is Python and what is it known for?
         Answer: Python is a high-level, interpreted programming language known for its clear, re
         Correctness=1  ToolUsage=0.0  Latency=1.54s
  ...

  COMPARISON TABLE: Architecture x Metric
  Architecture            Correct  ToolUse  Latency  Tokens  Quality Efficiency
  --------------------------------------------------------------------------
  multi_step_pipeline       1.000    1.000    6.49s     101    1.000      0.990
  react_agent               1.000    0.800   11.48s     707    0.900      0.127
  simple_chain              1.000    0.400    4.86s     122    0.700      0.574

  COST-QUALITY TRADEOFF ANALYSIS
  Best quality:       multi_step_pipeline (score=1.000)
  Fastest:            simple_chain (latency=4.864s)
  Most efficient:     multi_step_pipeline (efficiency=0.990)

  Pareto Frontier (quality vs latency):
    * multi_step_pipeline: quality=1.000, latency=6.494s
    * simple_chain: quality=0.700, latency=4.864s
```

Exact numbers vary between runs, but three things in that table are worth
reading carefully.

**Correctness is 1.000 for all three — and that is the interesting part.** On
this benchmark, every architecture gets the answer right; correctness does not
discriminate at all. Everything that separates them lives in the *other*
columns. A comparison that measured only accuracy would have concluded "they are
identical" and stopped.

**`react_agent` is dominated.** It matches the pipeline on nothing, is roughly
twice as slow, and spends **707 tokens against the pipeline's 101** — seven times
the cost for a lower score. It is absent from the Pareto frontier for exactly
that reason: another architecture beats it on every axis at once, so no set of
priorities makes it the right pick here.

**`simple_chain` survives despite the worst quality.** Nothing is faster, and the
frontier keeps whatever is unbeaten on at least one axis. If your requirement is
"lowest latency that clears the quality bar", it stays a candidate.

## Key Takeaways

- A comparison is only as good as its yardstick. A substring check calls
  *"known for its clear, readable syntax"* wrong; a registered judge does not.
- Registering the judge is what makes scores portable — the same named, versioned
  object scores every architecture, later runs, and (in L2-M2.2.2) production.
- Accuracy alone often fails to discriminate between architectures. Cost,
  latency and tool discipline are what separate them.
- The Pareto frontier eliminates dominated designs — `react_agent` here — while
  keeping anything unbeaten on at least one axis.
- Deterministic checks stay local `@scorer`s; only judgement calls need a model.

## Next Steps

**L2-M3.1 (Agent Optimization)** takes an architecture and systematically
improves it — a hand-built grid over prompts and temperature as the baseline,
then `mlflow.genai.optimize_prompts()` for automated instruction tuning, compared
on both quality and token spend.
