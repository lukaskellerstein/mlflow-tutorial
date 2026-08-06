# L2-M3.2 — Agent Configuration Optimization

**Level:** AI Agents
**Duration:** 90 min

## Overview

L2-M3.1 used `mlflow.genai.optimize_prompts()` — a real optimizer with a real algorithm. This lesson covers every knob MLflow has **no** optimizer for: which model to use, which tools and MCP servers to expose, and how to structure delegation. There is one pattern for all of them, and MLflow's role in it is to **track the search, not run it**.

## Prerequisites

- Completed: L2-M2.1.3 (Agent Quality Metrics)
- Completed: L2-M3.1 (Prompt and Instruction Optimization)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)

## Concepts

### What MLflow does and does not do here

`optimize_prompts()` takes `prompt_uris` — registered prompts, nothing else. There is no MLflow optimizer for model choice, tool budgets, skills, subagents or MCP server selection. MLflow's own documentation files all agent optimization under the prompt registry, which tells you where the supported surface ends.

That is not a dead end. The honest framing is:

> MLflow tracks the search; you run it.

A parent run per sweep, a child run per configuration, every child scored the same way. The comparison becomes auditable afterwards, which is the part that actually survives contact with a team.

### The knobs worth sweeping

| Knob | Why it matters |
|:--|:--|
| **Model** | The highest-leverage single change in practice, and the cheapest to try |
| **Tool / MCP budget** | Which tools exist at all. Fewer tools frequently beats more — every extra tool is another chance to pick wrong |
| **Delegation topology** | Subagents and skills as a search space, not a fixed design |

### Read a frontier, not a winner

Reporting one winner hides the tradeoff. A configuration is on the **Pareto frontier** if nothing beats it on *both* quality and cost. Everything on the frontier is defensible; choosing between them is a business decision, not a measurement.

### Knowing when to stop

With five test cases, one flipped answer moves accuracy by 20 points. If the spread across your configurations is smaller than one test case is worth, the sweep has found nothing — enlarge the dataset before believing a ranking. This lesson prints that comparison explicitly.

## Step-by-Step

### Step 1: Define the search space

```python
MODELS = ["gemma-26b-free", "gemma-31b-free"]
TOOL_BUDGETS = {"minimal": MINIMAL_TOOLS, "full": ALL_TOOLS}
```

The local aliases (`gemma-chat`, `gemma-judge`, `gemma-agent`) are deliberately absent, and the reason is sharper than "LMStudio might be asleep". Both carry an error fallback to OpenRouter in `infra/litellm/config.yaml`, so an unloaded model does not fail the sweep — it silently *substitutes a different model* and the run keeps going. A sweep whose independent variable can change without telling you is worse than one that crashes.

### Step 2: One nested run per configuration

```python
with mlflow.start_run(run_name="configuration_sweep"):
    for model_alias, (budget_name, tools) in space:
        with mlflow.start_run(run_name=f"{model_alias}/{budget_name}", nested=True):
            mlflow.log_params({"model": model_alias, "tool_budget": budget_name})
            mlflow.log_metrics(score_config(...))
```

The nesting **is** the search log.

### Step 3: Compute the frontier

A config is dominated if another is at least as good on accuracy *and* at least as fast, and strictly better on one of them. What survives is the frontier.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M3_agent_optimization/2_configuration_optimization
uv sync
uv run python main.py
```

## Expected Output

```text
  Part 3: Pareto frontier (quality vs. latency)

  configuration                 accuracy   latency   tools
  --------------------------------------------------------
  gemma-26b-free/minimal           100%     36.8s       2
  gemma-26b-free/full              100%     20.9s       4
  gemma-31b-free/minimal           100%     14.6s       2
  gemma-31b-free/full              100%     18.1s       4

  On the frontier (1 of 4):
    gemma-31b-free/minimal  --  100% at 14.62s

  Part 4: when to stop
    accuracy spread across configs : 0%
    one test case is worth          : 20%
    -> The spread is within a single test case. This sweep has NOT
       found a real difference; add cases before trusting a winner.
```

**This is the expected result, and it is the point.** Five easy cases do not separate four capable configurations — every one scores 100%, the frontier collapses to "whichever was fastest", and Part 4 says so out loud rather than crowning a winner. Latency spread (14.6s to 36.8s) is real and is the only axis carrying signal here.

The exercise worth doing: add harder cases until accuracy separates, and watch the frontier stop being a single point. Free-tier models are also non-deterministic even at `temperature=0.0`, so re-running shifts the latency ordering — another reason a one-run ranking is not evidence.

## Key Takeaways

- MLflow has no optimizer for model, tool budget or topology — it tracks the search.
- Nested runs turn an ad-hoc sweep into an auditable comparison.
- Fewer tools often beats more; the tool budget is a real knob, not a detail.
- Report a Pareto frontier, not a single winner.
- If the spread is smaller than one test case, you have measured noise.

## Next Steps

Continue to **L2-M3.3 — Optimizing Against Benchmarks Without Destroying Them**, which applies this same search discipline to public benchmarks, where tuning against the data you report on silently invalidates the number.
