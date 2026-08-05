# L2-M3.1 -- SWE-Bench Evaluation Pipeline

**Level:** AI Agents
**Duration:** 2.5 hours

## Overview

This lesson builds the **real** SWE-Bench evaluation pipeline. Rather than checking whether the agent merely produces diff-like output, it runs a Claude Agent SDK agent **inside** a Docker/Podman container where the repository is checked out. The agent edits actual files with its built-in tools, then the harness runs the repository's test suite and computes a resolution score — the same metric used on the SWE-Bench leaderboard.

## Prerequisites

- Completed: L2-M1 Agent Frameworks (especially L2-M1.3, Claude Agent SDK)
- Completed: L2-M2 Agent Evaluation
- MLFlow server running at <http://127.0.0.1:5555>
- **Podman or Docker installed and running**
- **A `CLAUDE_CODE_OAUTH_TOKEN` in your environment.** The agent's CLI process runs inside a Linux container and cannot reach your host's Keychain login. Create a long-lived token once with `claude setup-token` (browser OAuth, billed to your existing Claude subscription) and store it in your secrets manager so it lands in your shell environment — never in a file in this repo.
- Internet connection (downloads dataset and clones repos; the container needs egress to api.anthropic.com)

> **Cost note:** this lesson runs real bug-fixing sessions against Claude. Each instance is capped at `max_budget_usd=1.0`; the default 2 instances x 2 configurations tops out at $4 and typically costs $1-2.

## Concepts

### How SWE-Bench Scoring Really Works

The SWE-Bench leaderboard ranks coding agents by **resolution rate** — the percentage of real GitHub issues an agent can fix. Each instance in the dataset is a real issue from a popular open-source Python repository with a verified human patch.

Evaluation follows this pipeline:

```text
Clone repo at exact commit → Install dependencies → Apply test patch
    → Verify bug exists (tests fail) → Agent edits files in container
    → Run tests → Score: resolved / applied / no_changes
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

### Architecture — the agent runs where the code is

```text
main.py (orchestrator, host)
  ├── agent.py (Claude Agent SDK, host side = tracing + message streaming)
  │     └── cli_path = container_claude.sh
  │           └── podman exec -i -w /workspace/repo <container> claude ...
  │                 └── the CLI — and every tool it runs (Bash, Read,
  │                     Edit, Grep, Glob) — executes INSIDE the container
  └── harness.py (container lifecycle + test execution + scoring, host)
        └── Manages podman run / exec / cp / stop / rm
```

The SDK normally spawns a local `claude` binary and talks to it over stdio. `cli_path` lets the lesson swap that binary for a five-line wrapper that starts the CLI *inside* the instance's container instead — stdio flows through `podman exec -i` unchanged, so the host still streams every message into MLflow traces, while the agent itself lives next to the repo. Isolation falls out of the architecture: nothing of the agent runs on the host, so there is no host filesystem to protect with tool restrictions.

For each instance, the pipeline:

1. **Starts a container** from the eval image (python 3.11 + git + pytest + the Claude Code CLI)
2. **Clones the repo** at the exact `base_commit` and installs dependencies
3. **Applies the test patch** (adds the verification tests from the dataset)
4. **Baseline check** — runs `FAIL_TO_PASS` tests to confirm they actually fail
5. **Runs the agent** — a fresh `ClaudeSDKClient` session whose CLI process runs in the container with its standard built-in tools; it explores and **edits files directly**
6. **Captures the diff** via `git diff` (the agent's changes are already applied)
7. **Runs tests** — both `FAIL_TO_PASS` (should now pass) and `PASS_TO_PASS` (should still pass)
8. **Scores** — resolved, applied, or no_changes
9. **Logs everything** to MLflow: nested runs, hand-built traces, diffs, tool-call logs, and per-instance cost

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

The agent needs no custom tools at all — because its CLI process runs inside the container, its standard built-in tools already operate on the right filesystem. The whole integration is two options:

```python
ClaudeAgentOptions(
    tools=["Bash", "Read", "Edit", "Write", "Grep", "Glob"],
    cli_path=CLI_WRAPPER,                      # container_claude.sh
    env={"SWE_CONTAINER_ID": container_id},    # tells the wrapper which container
    ...
)
```

and the wrapper itself:

```bash
exec podman exec -i -w /workspace/repo \
    -e "CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}" \
    "${SWE_CONTAINER_ID}" claude "$@"
```

The agent edits files directly (like a real coding agent), then the harness — on the host — captures the diff:

```python
_, agent_diff, _ = harness.exec_in_container(container_id, "cd /workspace/repo && git diff")
```

### Step 4: Run Tests

Files are already modified by the agent — no `git apply` step needed. One subtlety: SWE-Bench stores **bare test function names** (`test_issue_24543`), which pytest cannot collect on its own. The harness therefore runs pytest against the test files the instance's `test_patch` touches, filtered with `-k` on the names:

```python
test_files = harness.test_files_from_patch(instance["test_patch"])
f2p_passed, f2p_failed, output = harness.run_tests(container_id, f2p_tests, test_files)
```

Two related constraints are baked into the image and instance selection: pytest is installed in the `Dockerfile` (no repo declares it), and `select_instances` picks the **newest** sympy instances — the harness runs everything on one `python:3.11` image, and pre-1.9 sympy cannot even import on 3.11 (`from collections import Mapping` was removed in 3.10). The real SWE-Bench harness builds a per-instance environment instead; this single-image version is the teachable simplification.

### Step 5: Score and Compare

Resolution rate is computed identically to the leaderboard:

```python
resolved = f2p_passed == len(f2p_tests) and f2p_failed == 0 and p2p_failed == 0
resolution_rate = sum(r["resolved"] for r in results) / len(results)
```

The two configurations compare effort levels (`low` vs `high`) at a fixed model (`claude-sonnet-5`) — the SDK has no temperature; effort is its speed/quality/cost axis. Cost per instance comes from `ResultMessage.total_cost_usd`.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M3_agent_benchmarks/1_swe_bench
uv sync
uv run python main.py
```

**First run** builds the base Docker image (~30s) and downloads the dataset (~100 MB). Each instance takes 3-10 minutes (clone + install + agent + tests). Default 2 instances x 2 configs = ~20-40 minutes total.

To use Docker instead of Podman:

```bash
SWE_BENCH_RUNTIME=docker uv run python main.py
```

## Expected Output

```text
============================================================
L2-M3.1 -- SWE-Bench Evaluation Pipeline
============================================================

Step 1: Ensuring evaluation base image ...
  Image swe-bench-eval already exists.

Step 2: Loading SWE-Bench Verified dataset ...
  Loaded 500 instances, selected 2 from sympy/sympy
    sympy__sympy-24562
    sympy__sympy-24661

Step 3: Agent: Claude Agent SDK / claude-sonnet-5
  Tracing is hand-built (@mlflow.trace + spans) -- the SDK has no autolog.

Step 4: Running evaluation ...

============================================================
Config: low_effort  (model=claude-sonnet-5, effort=low)
============================================================

  [1/2] Instance: sympy__sympy-24562

  [sympy__sympy-24562]
    Starting container ...
    Cloning sympy/sympy at b1cb676c ...
    Installing dependencies ...
    Repo setup complete.
    Applying test patch ...
    Baseline check: running 1 FAIL_TO_PASS tests ...
    Baseline: 0 passed, 1 failed (expect failures)
    Running agent (model=claude-sonnet-5, effort=low) ...
    Agent finished: 4 turns, 3 tool calls, $0.0827
    Agent edited files (1350 bytes diff)
    Running 1 FAIL_TO_PASS tests ...
    FAIL_TO_PASS: 1/1 passed
    Running 10 PASS_TO_PASS tests ...
    PASS_TO_PASS: 11 passed, 0 failed
    Result: RESOLVED  (latency=42.7s)

  [2/2] Instance: sympy__sympy-24661
    ...

============================================================
Summary
============================================================
  high_effort  100.0% resolved  (2/2)  avg_latency=58.8s  cost=$0.26
  low_effort   100.0% resolved  (2/2)  avg_latency=43.5s  cost=$0.16

  SWE-Bench Verified Leaderboard (for context):
  --------------------------------------------------
    claude-sonnet-5 (this run)          100.0% <--
    Claude 3.5 Sonnet                    49.0%
    GPT-4o                               33.2%
    DeepSeek-V2.5                        27.0%

  Note: With only 4 instance(s), results are not statistically meaningful.
  Increase SAMPLE_SIZE for reliable comparison.

============================================================
Done. View results at http://127.0.0.1:5555
============================================================
```

Two things worth noticing in real output. PASS_TO_PASS can report *more* passes than the 10 listed tests — `-k` matches by substring, so `test_mod` also selects `test_mod_inverse`; scoring is unaffected because an instance only resolves when the failed count is zero. And with 2 instances per config, one flipped outcome moves the resolution rate by 50 points — the 100% here says these two bugs are tractable, not that the model beats the leaderboard. The value of this lesson is seeing the **real evaluation pipeline** in action.

## Key Takeaways

- **SWE-Bench resolution rate** requires actually running tests inside containers — string-matching diffs is not real evaluation.
- **Run the agent where the code is**: `cli_path` + a five-line `podman exec` wrapper puts the whole agent inside the sandbox, its built-in tools included — no custom tool code at all.
- **The container boundary is the safety boundary**: nothing of the agent runs on the host, so it physically cannot touch host files — stronger than any tool allowlist.
- **Direct editing** is how real coding agents (Claude Code, Devin) work — the agent modifies files in place rather than generating diffs.
- **Container isolation** ensures each instance is evaluated in a reproducible environment with the exact repo state and dependencies.
- **MLflow tracks the full pipeline**: per-instance metrics, agent diffs, tool-call logs, gold patches, test output, and real dollar cost are all logged for post-hoc analysis.

## Next Steps

- **Scale up**: increase `SAMPLE_SIZE` to 20-50 for statistically meaningful results (budget several hours and a corresponding API spend)
- **Try the gold patches**: modify the code to evaluate the human patches to verify the harness scores 100%
- **Compare frameworks**: swap in a LangGraph or DeepAgents agent against the same harness and compare resolution rates
- **Add patch similarity**: compare agent patches against gold patches using diff metrics
- **Production pipeline**: see L3 for CI/CD integration with automated quality gates
