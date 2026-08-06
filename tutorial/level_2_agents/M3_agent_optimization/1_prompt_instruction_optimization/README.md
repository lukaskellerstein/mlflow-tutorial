# L2-M3.1 — Prompt and Instruction Optimization

**Level:** AI Agents
**Duration:** 90 min

## Overview

Agent quality depends on more than the LLM — system prompts, temperature, and
tool descriptions all measurably affect correctness and tool selection. Parts 1–3
tune each dimension by hand and track every variant as a nested MLflow run.
**Part 4 then hands the same job to `mlflow.genai.optimize_prompts()`** and asks
the obvious question: did the optimizer beat the human?

## Prerequisites

- Completed: L2-M2.2.1 (Architecture Comparison), L2-M2.1.2 (Judges)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in `infra/.env` — the `gemma-chat` alias routes there

## Concepts

### Why Optimize Agents Systematically?

Agent behavior is sensitive to many knobs: the system prompt, the LLM temperature, and even the wording of tool descriptions. Changing any of these can shift correctness, tool selection accuracy, and latency in non-obvious ways. Without systematic evaluation, developers rely on manual spot-checks that miss regressions.

This lesson treats agent optimization as a search problem:

1. **Define a fixed evaluation benchmark** — the same 5 test cases are used for every variant so results are comparable.
2. **Vary one dimension at a time** — system prompt, temperature, tool descriptions — while holding the others constant.
3. **Log everything** — each variant becomes a nested MLflow run with parameters and metrics.
4. **Compare and select** — a summary table across all dimensions identifies the best configuration.

### Optimization Dimensions

| Dimension | What Changes | Why It Matters |
|---|---|---|
| System prompt | Instructions given to the LLM | Guides reasoning strategy and tool usage patterns |
| Temperature | Sampling randomness (0.0 = deterministic) | Affects consistency and creativity of responses |
| Tool descriptions | Docstrings on each tool function | Determines whether the LLM picks the right tool |

### Evaluation Metrics

- **Correctness** — does the answer contain the expected value?
- **Tool Selection** — did the agent use the correct tool for the question?
- **Quality Score** — average of correctness and tool selection (composite metric)
- **Latency** — wall-clock time per evaluation run

## Step-by-Step

### Step 1: Define the Agent and Evaluation Benchmark

We build a ReAct agent with three tools (`calculator`, `unit_converter`, `fact_lookup`) and define 5 test cases that each require a specific tool:

```python
EVAL_CASES = [
    {"question": "What is 15 * 24 + 100?", "expected": "460", "needs_tool": "calculator"},
    {"question": "Convert 42 km to miles.", "expected": "26.1", "needs_tool": "converter"},
    {"question": "What is the speed of light?", "expected": "299", "needs_tool": "fact"},
    {"question": "How many pounds is 10 kg?", "expected": "22.0", "needs_tool": "converter"},
    {"question": "Who created Python?", "expected": "Guido", "needs_tool": "fact"},
]
```

### Step 2: System Prompt Optimization

Three system prompts are tested — from minimal to highly structured:

- **Minimal**: `"You are a helpful assistant."` — no tool guidance
- **Detailed**: Mentions available tools and encourages their use
- **Structured**: Numbered rules mapping question types to specific tools

The structured prompt is expected to produce the best tool selection because it explicitly tells the LLM which tool to use for each question type.

### Step 3: Temperature Tuning

Using the best system prompt from Step 2, we sweep four temperatures (0.0, 0.3, 0.7, 1.0). Lower temperatures are more deterministic and typically produce more consistent tool usage, while higher temperatures introduce randomness that can cause the agent to skip tools or hallucinate.

### Step 4: Tool Description Optimization

Two sets of tool descriptions are compared:

- **Original** — terse one-line docstrings (e.g., `"Evaluate a math expression."`)
- **Improved** — detailed descriptions with usage guidance and argument examples

Better descriptions help the LLM understand when and how to use each tool, improving selection accuracy.

### Step 5: Optimization Summary

All results are combined into a comparison table. MLflow logs the optimization trajectory as stepped metrics so you can visualize the quality curve in the MLflow UI chart view.

### Step 6: Hand the same job to an optimizer

Parts 1–3 are a hand-built grid. Its ceiling is the same as a hand-written test
suite: it only explores what somebody thought to type.
`mlflow.genai.optimize_prompts()` searches instead — but two constraints shape
how you write it.

**The prompt must live in the prompt registry, not a Python string.** The
optimizer rewrites *registered versions*, which is why it takes `prompt_uris`:

```python
version = mlflow.genai.register_prompt(name=PROMPT_NAME, template=SYSTEM_PROMPTS["minimal"])
PROMPT_URI = f"prompts:/{PROMPT_NAME}/{version.version}"
```

**`predict_fn` must call `PromptVersion.format()` at run time.** That call *is*
the hook — it is how a candidate template reaches the agent. Inline the prompt
text instead and the optimizer will dutifully rewrite something nothing reads:

```python
def optimize_predict_fn(question: str) -> str:
    prompt = mlflow.genai.load_prompt(PROMPT_URI)  # <- not a constant
    agent = create_agent(model=..., tools=ORIGINAL_TOOLS, system_prompt=prompt.format())
    ...
```

The scorer is deliberately the **same** `answer_correct` check Part 1 uses.
Holding the yardstick constant is what makes "did the optimizer beat the human?"
a real question rather than two incomparable numbers.

> [!note]
> `MetaPromptOptimizer`, not `GepaPromptOptimizer`. GEPA defaults to
> `max_metric_calls=100`, which on a free tier turns a 90-minute lesson into an
> afternoon for the same teaching point.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M3_agent_optimization/1_prompt_instruction_optimization
uv sync
uv run python main.py
```

> [!note]
> Expect 15–20 minutes. Nine hand-built variants over seven cases, then the
> optimizer's own rewrite-and-rescore loop.

## Expected Output

The script prints per-case results for each variant, followed by a summary table:

```text
  Variant                Dimension          Correct  ToolSel  Quality  Latency
  --------------------------------------------------------------------------
  prompt_minimal         system_prompt        0.600    0.400    0.500   1.234s
  prompt_detailed        system_prompt        0.800    0.800    0.800   1.456s
  prompt_structured      system_prompt        0.800    1.000    0.900   1.389s
  temp_0.0               temperature          0.800    1.000    0.900   1.345s
  temp_0.3               temperature          0.800    0.800    0.800   1.567s
  ...
  tools_original         tool_descriptions    0.800    1.000    0.900   1.345s
  tools_improved         tool_descriptions    1.000    1.000    1.000   1.234s

  BEST CONFIGURATION: tools_improved
```

In the MLflow UI:
- Navigate to experiment `L2/M3_agent_optimization/1_prompt_instruction_optimization`
- The parent run `agent_optimization` contains all nested variant runs
- Each nested run has parameters (`dimension`, `variant`, `temperature`) and metrics (`avg_correctness`, `avg_tool_selection`, `quality_score`)
- The parent run has stepped metrics (`opt_quality`, `opt_correctness`) showing the optimization trajectory

### Part 4 output

```text
  registered prompts:/agent_system_prompt/1
  starting template: 'You are a helpful assistant.'
  optimizing (the reflection model rewrites the prompt, then re-scores)...

  optimizer (MetaPromptOptimizer):
    score before : 0.857
    score after  : 1.000
  hand-built grid, best correctness : 0.857  (same scale as above)
  hand-built grid, best quality     : 0.929  (blended, not comparable)

  the optimizer rewrote the prompt as:
    You are a helpful assistant. Provide accurate, concise answers. Follow these
    guidelines: 1. Answer the user's question directly. 2. For math or conversion
    problems, compute carefully and verify the result before responding. 3. Give only
    the final answer unless the user asks for an explanation. ...

  The optimizer improved the prompt by +0.143 without anyone editing it.
```

**The optimizer beat the entire hand-built grid.** Look down the `Correct`
column of the summary table: every one of the nine manual variants — three
prompts, four temperatures, two tool-description sets — plateaus at **0.857**.
Not one of them fixes `4839 * 271`. The optimizer reached **1.000** by writing an
instruction no one in the grid had thought to write: *"compute carefully and
verify the result before responding."*

> [!caution]
> **Compare correctness with correctness.** The grid's `Quality` column peaks at
> 0.929, but that is a *blend* of correctness and tool selection, while the
> optimizer is scored by `answer_correct` alone. Printing 0.929 next to 1.000
> would be two different scales pretending to be one. The script prints both and
> labels which is comparable — a mistake worth avoiding whenever an optimizer's
> objective is narrower than your dashboard's headline metric.

> [!important]
> **Two of the seven benchmark cases exist only to leave headroom, and that is a
> lesson in itself.** The original five (`15 * 24 + 100`, "who created Python?")
> are all answerable from memory, so a bare `"You are a helpful assistant."`
> already scored **1.0** on them — and the first run of this lesson reported
> `initial=1.0, final=1.0`. The optimizer was not broken; there was simply
> nothing left to win.
>
> `4839 * 271` and `137 km → miles` need the tool's precision, which drops the
> baseline to 0.857 and gives the optimizer something to find. **A saturated
> benchmark cannot rank prompts.** If your baseline scores 1.0, fix the
> benchmark before reaching for an optimizer. The script prints exactly this
> when it detects a perfect starting score.

## Key Takeaways

- System prompts with explicit tool-routing instructions significantly improve tool selection accuracy.
- Lower temperatures (0.0-0.3) generally produce more consistent agent behavior than higher values.
- Detailed tool descriptions with usage guidance and argument examples help the LLM choose the right tool.
- Systematic evaluation with a fixed benchmark is essential — intuition about which settings "should" work often does not match reality.
- MLflow nested runs make it easy to compare variants side-by-side and identify the best configuration.
- **An optimizer needs the prompt in the registry and a `predict_fn` that calls
  `format()`** — the registry is where it writes, and `format()` is where the
  candidate is read.
- **A saturated baseline makes optimization meaningless.** Measure the starting
  score first; if it is already 1.0, the benchmark is the thing to fix.
- Score the optimizer with the *same* metric as the manual grid, or the two
  numbers are not comparable.

## Next Steps

In L2-M3.5 (End-to-End Agent Evaluation Pipeline), you will combine all the evaluation techniques from this module into a complete automated pipeline with CI/CD integration, regression detection, and quality dashboards.
