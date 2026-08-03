# L2-M4.1 -- SWE-Bench Evaluation Pipeline

**Level:** AI Agents
**Duration:** 2.5 hours

## Overview

This lesson builds an end-to-end evaluation pipeline that runs a LangChain/LangGraph coding agent against the SWE-Bench Verified dataset -- a benchmark of real GitHub issues from popular Python repositories with verified human patches. You will learn how to load the dataset, execute an agent on each instance, track results in MLflow, and compare agent configurations side by side.

## Prerequisites

- Completed: L2-M1 Agent Frameworks
- Completed: L2-M3 Agent Evaluation
- MLFlow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-26b-a4b` loaded
- Internet connection (first run downloads the SWE-Bench dataset from Hugging Face)

## Concepts

### What is SWE-Bench?

SWE-Bench is a benchmark dataset created by researchers at Princeton. Each instance is a real issue from a popular open-source Python repository (Django, Flask, scikit-learn, sympy, and others). Every instance includes:

- **repo** -- the GitHub repository where the issue was filed
- **instance_id** -- a unique identifier for the issue
- **problem_statement** -- the full issue description as written by the original reporter
- **hints_text** -- optional hints or discussion from the issue thread
- **patch** -- the verified human-written fix (ground truth)

The "Verified" variant (`SWE-bench_Verified`) contains instances where the patches have been independently verified to actually fix the issue.

### Why Evaluate Coding Agents?

Coding agents are a rapidly growing class of AI systems. Unlike simple question-answering, a coding agent must:

1. Understand a natural-language problem description
2. Reason about which files and code paths are relevant
3. Generate a correct patch in unified diff format
4. Avoid introducing regressions

SWE-Bench measures real-world software engineering capability, not just code generation. It is the standard benchmark for evaluating coding agents.

### Pipeline Architecture

The evaluation pipeline follows this structure:

```text
Load Dataset --> Sample Instances --> For each config:
                                        Create Agent (LLM + Tools)
                                        For each instance:
                                            Build prompt
                                            Invoke agent
                                            Log results to MLflow
                                     --> Compare configs
```

All runs are organized under a single parent MLflow run (`swe_bench_eval`) with nested runs per configuration and per instance.

### Comparing Agent Configurations

We compare two temperature settings to explore the precision-creativity tradeoff:

| Config     | Temperature | Hypothesis                                    |
|------------|-------------|-----------------------------------------------|
| `precise`  | 0.3         | Lower randomness leads to more reliable patches |
| `creative` | 0.7         | Higher randomness explores more solution paths  |

MLflow nested runs make it straightforward to compare aggregate metrics (latency, patch generation rate) across configurations.

## Step-by-Step

### Step 1: Load the SWE-Bench Dataset

We use the Hugging Face `datasets` library to load the SWE-Bench Verified split. For demonstration purposes we take the first 5 instances.

```python
ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
sample = [ds[i] for i in range(SAMPLE_SIZE)]
```

### Step 2: Define Agent Tools

The agent has two tools:

- **analyze_code** -- takes the problem description and returns an analysis of the root cause and suggested approach
- **generate_patch** -- takes the analysis and repository name and returns a unified diff

```python
@tool
def analyze_code(problem: str) -> str:
    """Analyze a coding problem and identify the root cause."""
    ...


@tool
def generate_patch(analysis: str, repo: str) -> str:
    """Generate a unified diff patch based on the analysis."""
    ...
```

### Step 3: Build the Agent

We use `create_agent` from LangChain v1.0+ with a `ChatOpenAI` model pointed at LMStudio.

```python
from langchain.agents import create_agent

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="google/gemma-4-26b-a4b",
    temperature=temperature,
    max_tokens=1024,
)
agent = create_agent(model=llm, tools=[analyze_code, generate_patch], system_prompt=SYSTEM_PROMPT)
```

### Step 4: Run Each Instance

For each SWE-Bench instance, we build a prompt from the problem statement and hints, invoke the agent, and log per-instance metrics to MLflow.

```python
result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
mlflow.log_metric("latency_s", round(latency, 2))
mlflow.log_metric("patch_generated", int(patch_generated))
```

### Step 5: Compare Configurations

After both configurations finish, aggregate metrics are computed and logged. A summary table is printed showing average latency, patch generation rate, and response length for each configuration.

### Step 6: Review in MLflow UI

Open <http://127.0.0.1:5555>, navigate to the `L2/M4_agent_benchmarks/1_swe_bench` experiment. You will see:

- A parent run `swe_bench_eval` containing all metadata
- Nested runs `config_precise` and `config_creative` with aggregate metrics
- Per-instance nested runs with detailed metrics and tracing (via `mlflow.langchain.autolog()`)
- CSV artifacts with full results

## Running the Lesson

```bash
cd tutorial/level_2_agents/M4_agent_benchmarks/1_swe_bench
uv sync
uv run python main.py
```

Note: the first run will download the SWE-Bench Verified dataset from Hugging Face (~100 MB). Subsequent runs use the cached version.

## Expected Output

```text
============================================================
L2-M4.1 -- SWE-Bench Evaluation Pipeline
============================================================

Step 1: Loading SWE-Bench Verified dataset ...
  Loaded 500 instances, using 5 for demo

============================================================
Config: precise  (temperature=0.3)
============================================================
  [django__django-11099] latency=8.2s patch=True
  [django__django-11133] latency=7.5s patch=True
  [django__django-11179] latency=9.1s patch=True
  [django__django-11283] latency=6.8s patch=False
  [django__django-11374] latency=7.9s patch=True

============================================================
Config: creative  (temperature=0.7)
============================================================
  [django__django-11099] latency=9.4s patch=True
  [django__django-11133] latency=8.7s patch=True
  [django__django-11179] latency=10.2s patch=True
  [django__django-11283] latency=8.1s patch=True
  [django__django-11374] latency=9.3s patch=True

============================================================
Summary Comparison
============================================================
          avg_latency  patch_rate  avg_response_len  success_count
config
creative         9.14        1.0            1245.2              5
precise          7.90        0.8            1102.8              5

============================================================
Done. View results at http://127.0.0.1:5555
============================================================
```

Actual values will vary depending on LMStudio model performance.

## Key Takeaways

- **SWE-Bench is the standard benchmark** for evaluating coding agents on real-world software engineering tasks, using actual GitHub issues with verified patches.
- **Evaluation pipelines need structure**: dataset loading, agent execution, per-instance tracking, and aggregate comparison should all be organized in MLflow with nested runs.
- **Comparing configurations is essential**: even small changes like temperature can significantly affect agent behavior -- MLflow makes this comparison systematic.
- **Autologging captures traces**: `mlflow.langchain.autolog()` automatically records the agent's reasoning steps, tool calls, and intermediate outputs.
- **Artifact logging preserves details**: saving per-instance CSV results as artifacts ensures you can drill into individual cases long after the run completes.

## Next Steps

From here you can extend the pipeline in several directions:

- **Scale up** the sample size and run against the full SWE-Bench Verified dataset
- **Add patch validation** by comparing generated patches against the ground-truth patches using diff similarity metrics
- **Build CI/CD gates** that fail a deployment if patch generation rate drops below a threshold
- **Integrate with production evaluation pipelines** using `mlflow.genai.scorers` for automated quality assessment
- **Compare across agent frameworks** (LangGraph vs Claude Agent SDK) using the same evaluation harness
