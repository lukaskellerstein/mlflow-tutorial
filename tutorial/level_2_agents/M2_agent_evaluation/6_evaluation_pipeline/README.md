# L2-M2.6 — Evaluation Pipeline: Offline Gates and Online Scoring

**Level:** AI Agents
**Duration:** 90 min

## Overview

The capstone for the Agent Evaluation module, and the lesson where the module's
central distinction finally has a name. An evaluation setup has to answer **two**
questions, and they are not the same question:

- **Offline** — *is this version good enough to ship?* A curated dataset, known
  expectations, every case scored, and **you** decide when it runs.
- **Online** — *is what shipped still good?* Production traces, no ground truth,
  **sampled** coverage because judges cost a model call each, and the **server**
  pulls the trigger on a schedule.

Parts 1–2 build the offline half (pipeline, quality gates, regression detection).
Part 3 builds the online half on a real MLflow AI Gateway model.

## Prerequisites

- Completed: L2-M2.1 (Test Generation), L2-M2.2 (Judges), L2-M2.3 (Quality Metrics), L2-M2.5 (Optimization)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- **`OPENROUTER_API_KEY` exported in your shell** — Part 3 creates a gateway
  secret from the environment. `source infra/.env` if it is not set.

## The four axes

| | offline | online |
|:--|:--|:--|
| **input** | curated dataset | production traces |
| **ground truth** | expectations | none |
| **coverage** | every case | sampled (20% here) |
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
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
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

## Part 3 — Online scoring

### The constraint that shapes everything

`scorer.start()` **refuses any judge whose model is not a gateway model**:

```text
INVALID_PARAMETER_VALUE: Scorer 'x' does not use a gateway model.
Automatic evaluation is only supported for scorers that use gateway models.
```

That is not arbitrary. Online scoring runs **server-side**, so the server needs
its own credentialed endpoint — it cannot borrow the API key from your shell.
`openai:/…` and `anthropic:/…` are client-side model URIs; only `gateway:/…`
points at something the server owns.

### Building the endpoint

Three objects, in order — secret → model definition → endpoint:

```python
store = _get_store()
secret = store.create_gateway_secret(
    secret_name="openrouter-tutorial",
    secret_value={"api_key": os.environ["OPENROUTER_API_KEY"]},  # from env, never a literal
    provider="openrouter",
)
model_def = store.create_gateway_model_definition(
    name="or-gemma-large",
    secret_id=secret.secret_id,
    provider="openrouter",
    model_name="google/gemma-4-26b-a4b-it:free",
)
store.create_gateway_endpoint(
    name="tutorial-gemma-endpoint",
    model_configs=[
        GatewayEndpointModelConfig(
            model_definition_id=model_def.model_definition_id,
            linkage_type=GatewayModelLinkageType.PRIMARY,
            weight=1,
            fallback_order=0,
        )
    ],
)
```

Then the judge, and the sampling config that turns it on:

```python
judge = mlflow.genai.make_judge(..., model="gateway:/tutorial-gemma-endpoint")
judge.register(name=...).start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
```

`sample_rate` is the entire economic argument for online scoring: judging costs a
model call **per trace**, so spend scales with *traffic*, not with dataset size.
20% is a decision, not a default.

### Three traps, each of which cost real debugging time

> [!warning]
> **1. `auth_config={"base_url": ...}` is accepted and silently ignored.** The
> obvious move is to point a `provider="openai"` secret at the LiteLLM proxy the
> rest of this module uses. The secret is created without complaint — and then
> the judge call fails with an authentication error from **`platform.openai.com`**
> about a key you never sent there. The base URL never reaches the request.
> `provider="openrouter"` is supported natively (`DEFAULT_API_BASE =
> https://openrouter.ai/api/v1`), so the server calls OpenRouter directly and no
> base URL is needed. **The gateway does not go through LiteLLM at all.**

> [!warning]
> **2. `linkage_type` wants the enum, not the string.** `"PRIMARY"` fails deep in
> proto serialisation with `'str' object has no attribute 'to_proto'`. Use
> `GatewayModelLinkageType.PRIMARY`.

> [!warning]
> **3. `start()` returns a NEW scorer object.** Reading `.status` off the
> original still shows `STOPPED` and looks like a silent failure. Read the
> returned object, or re-fetch with `get_scorer()` — which is also how you prove
> the state is genuinely persisted server-side rather than local.

### Scoring is asynchronous

The server's scheduler picks up active scorers on its own cadence (roughly every
minute), so assessments appear on sampled traces *shortly after* the script
finishes — not synchronously with it. An empty result the instant the script ends
is expected, not a failure.

The lesson calls `stop()` before exiting. A scorer left started keeps sampling
every trace in that experiment, forever, at a model call each.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/6_evaluation_pipeline
uv sync
uv run python main.py
```

> [!note]
> Part 3 needs `OPENROUTER_API_KEY` in the environment. The script exits with a
> clear message rather than half-building a gateway if it is missing.

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
