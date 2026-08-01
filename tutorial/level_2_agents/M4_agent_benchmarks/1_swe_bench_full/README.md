# L2-M4.1 -- Full SWE-Bench Evaluation Pipeline

**Level:** AI Agents
**Duration:** 2.5 hours

## Overview

This lesson builds the **real** SWE-Bench evaluation pipeline. Unlike the simplified version in `1_swe_bench/` (which only checks if the agent produces diff-like output), this lesson applies patches inside Docker/Podman containers, runs the repository's test suite, and computes a resolution score — the same metric used on the SWE-Bench leaderboard.

## Prerequisites

- Completed: L2-M1 Agent Frameworks
- Completed: L2-M3 Agent Evaluation
- MLFlow server running at http://127.0.0.1:5555
- LMStudio running with `google/gemma-4-26b-a4b` loaded
- **Podman or Docker installed and running**
- Internet connection (downloads dataset and clones repos)

## Concepts

### How SWE-Bench Scoring Really Works

The SWE-Bench leaderboard ranks coding agents by **resolution rate** — the percentage of real GitHub issues an agent can fix. Each instance in the dataset is a real issue from a popular open-source Python repository with a verified human patch.

Evaluation follows this pipeline:

```
Clone repo at exact commit → Install dependencies → Apply test patch
    → Verify bug exists (tests fail) → Apply agent's patch
    → Run tests → Score: resolved / applied / failed
```

An instance is **resolved** only if:
1. The agent edited the source files (captured via `git diff`)
2. ALL `FAIL_TO_PASS` tests now pass (the bug is fixed)
3. ALL `PASS_TO_PASS` tests still pass (no regressions)

This requires running code inside isolated containers — you can't just string-match the output.

### Dataset Fields

Each SWE-Bench instance provides:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique ID, e.g. `sympy__sympy-20590` |
| `repo` | GitHub repo, e.g. `sympy/sympy` |
| `base_commit` | The commit to check out (the state when the bug existed) |
| `problem_statement` | The issue description as written by the reporter |
| `patch` | The verified human fix (ground truth) |
| `test_patch` | Test code that verifies whether the fix works |
| `FAIL_TO_PASS` | Tests that should fail before the fix and pass after |
| `PASS_TO_PASS` | Tests that must continue passing (regression check) |

### Why Containers?

Each instance needs the repository at a specific commit with its exact dependencies. Running this on the host would:
- Pollute your Python environment with conflicting packages
- Risk version conflicts between instances
- Make results non-reproducible

Containers provide isolated, reproducible environments for each evaluation.

### Architecture

```
main.py (orchestrator)
  ├── agent.py (DeepAgents agent with container sandbox)
  │     ├── ContainerSandbox(BaseSandbox) — wraps podman exec
  │     └── create_deep_agent() — auto-generates ls, read, write, edit, grep, glob, execute tools
  └── harness.py (container lifecycle + test execution + scoring)
        └── Manages podman run / exec / cp / stop / rm
```

For each instance, the pipeline:

1. **Starts a container** from a `python:3.11-slim` base image
2. **Clones the repo** at the exact `base_commit` and installs dependencies
3. **Applies the test patch** (adds the verification tests from the dataset)
4. **Baseline check** — runs `FAIL_TO_PASS` tests to confirm they actually fail
5. **Runs the DeepAgents agent** — the agent uses sandbox tools (read, edit, grep, execute, etc.) to explore and **edit files directly** inside the container
6. **Captures the diff** via `git diff` (the agent's changes are already applied)
7. **Runs tests** — both `FAIL_TO_PASS` (should now pass) and `PASS_TO_PASS` (should still pass)
8. **Scores** — resolved, applied, or no_changes
9. **Logs everything** to MLflow with nested runs

### Why This Lesson Uses sympy

We filter for `sympy/sympy` instances because sympy is:
- **Pure Python** — no C extensions, no system dependencies
- **Fast to install** — `pip install -e .` just works
- **Quick test suite** — individual tests run in seconds

Django requires database setup, scikit-learn needs compiled extensions, and matplotlib requires system graphics libraries. Sympy keeps the demo practical.

## Step-by-Step

### Step 1: Build the Base Image

The `Dockerfile` creates a minimal evaluation environment:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential
WORKDIR /workspace
```

The harness builds this image automatically on first run.

### Step 2: Load and Filter the Dataset

```python
ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
instances = select_instances(ds, "sympy/sympy", SAMPLE_SIZE)
```

### Step 3: Agent Explores and Fixes the Bug

The agent uses DeepAgents with a `ContainerSandbox` that wraps `podman exec`. DeepAgents auto-generates all file tools (read, write, edit, grep, glob, ls, execute) from just 4 sandbox methods:

```python
from deepagents import create_deep_agent
from deepagents.backends.sandbox import BaseSandbox


class ContainerSandbox(BaseSandbox):
    def execute(self, command, *, timeout=None):
        rc, stdout, stderr = exec_in_container(self._container_id, command)
        return ExecuteResponse(output=stdout + stderr, exit_code=rc, truncated=False)

    # + id, upload_files, download_files


sandbox = ContainerSandbox(container_id)
agent = create_deep_agent(model=llm, backend=sandbox, system_prompt=SYSTEM_PROMPT)
result = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"recursion_limit": 50},
)
```

The agent **edits files directly** in the container (like a real coding agent), then we capture the diff:

```python
_, agent_diff, _ = harness.exec_in_container(container_id, "cd /workspace/repo && git diff")
```

### Step 4: Run Tests

Files are already modified by the agent — no `git apply` step needed:

```python
f2p_passed, f2p_failed, output = harness.run_tests(container_id, f2p_tests, repo)
```

### Step 5: Score and Compare

Resolution rate is computed identically to the leaderboard:

```python
resolved = f2p_passed == len(f2p_tests) and f2p_failed == 0 and p2p_failed == 0
resolution_rate = sum(r["resolved"] for r in results) / len(results)
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M4_agent_benchmarks/1_swe_bench_full
uv sync
uv run python main.py
```

**First run** builds the base Docker image (~30s) and downloads the dataset (~100 MB). Each instance takes 2-5 minutes (clone + install + agent + tests). Default 2 instances = ~15-20 minutes total.

To use Docker instead of Podman:

```bash
SWE_BENCH_RUNTIME=docker uv run python main.py
```

## Expected Output

```
============================================================
L2-M4.1 -- Full SWE-Bench Evaluation Pipeline
============================================================

Step 1: Ensuring evaluation base image ...
  Image swe-bench-eval ready.

Step 2: Loading SWE-Bench Verified dataset ...
  Loaded 500 instances, selected 2 from sympy/sympy
    sympy__sympy-20590
    sympy__sympy-21612

Step 3: Enabling MLflow autolog ...

Step 4: Running evaluation ...

============================================================
Config: precise  (temperature=0.3)
============================================================

  [1/2] Instance: sympy__sympy-20590
    Starting container ...
    Cloning sympy/sympy at abc12345 ...
    Installing dependencies ...
    Repo setup complete.
    Applying test patch ...
    Baseline check: running 2 FAIL_TO_PASS tests ...
    Baseline: 0 passed, 2 failed (expect failures)
    Running agent ...
    Agent edited files (312 bytes diff)
    Running 2 FAIL_TO_PASS tests ...
    FAIL_TO_PASS: 0/2 passed
    Running 5 PASS_TO_PASS tests ...
    PASS_TO_PASS: 5/5 passed
    Result: APPLIED  (latency=142.3s)

  [2/2] Instance: sympy__sympy-21612
    ...
    Result: APPLIED  (latency=98.7s)

============================================================
Config: creative  (temperature=0.7)
============================================================
  ...

============================================================
Summary
============================================================
  precise      0.0% resolved  (0/2)  avg_latency=120.5s
  creative     0.0% resolved  (0/2)  avg_latency=135.2s

  SWE-Bench Verified Leaderboard (for context):
  --------------------------------------------------
    Claude 3.5 Sonnet                    49.0%
    GPT-4o                               33.2%
    DeepSeek-V2.5                        27.0%
    Local gemma-4-26b (this run)          0.0% <--

  Note: With only 2 instance(s), results are not statistically
  meaningful. Increase SAMPLE_SIZE for reliable comparison.

============================================================
Done. View results at http://127.0.0.1:5555
============================================================
```

The local model will likely score 0% — SWE-Bench tasks require deep codebase understanding that's beyond most small models. The value of this lesson is seeing the **real evaluation pipeline** in action.

## Key Takeaways

- **SWE-Bench resolution rate** requires actually running tests inside containers — string-matching diffs is not real evaluation.
- **DeepAgents + BaseSandbox** gives the agent full file operations (read, write, edit, grep, execute) from just 4 methods — no need to hand-roll individual tools.
- **Direct editing** is how real coding agents (Claude Code, Devin) work — the agent modifies files in place rather than generating diffs.
- **Container isolation** ensures each instance is evaluated in a reproducible environment with the exact repo state and dependencies.
- **MLflow tracks the full pipeline**: per-instance metrics, agent diffs, gold patches, and test output are all logged as artifacts for post-hoc analysis.
- **Agent capability matters**: the gap between 0% (local small model) and 49% (Claude 3.5 Sonnet) shows that SWE-Bench tests real software engineering ability, not just code generation.

## Next Steps

- **Scale up**: increase `SAMPLE_SIZE` to 20-50 for statistically meaningful results (budget 1-3 hours)
- **Try the gold patches**: modify the code to evaluate the human patches to verify the harness scores 100%
- **Compare frameworks**: use the same harness with a LangGraph agent or Claude Agent SDK agent
- **Add patch similarity**: compare agent patches against gold patches using diff metrics
- **Production pipeline**: see L3 for CI/CD integration with automated quality gates
