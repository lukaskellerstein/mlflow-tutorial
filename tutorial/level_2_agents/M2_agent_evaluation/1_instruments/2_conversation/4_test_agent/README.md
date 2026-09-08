# L2-M2.1.2.4 — test_agent(): the Whole Pipeline in One Call

**Level:** AI Agents
**Duration:** 45 min

## Overview

L2-M2.1.1.1 needed a person to write every case. L2-M2.1.2.1 needed a person to write
every goal, or existing traffic to distil goals from. `mlflow.genai.test_agent()`
needs neither: it asks the agent to describe itself, writes the goals from that
description, simulates them, and LLM-judges the resulting traces for issues.

## Prerequisites

- Completed: L2-M2.1.2.1 (Multi-Turn Conversation Simulation)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### Four stages behind one call

```python
result = mlflow.genai.test_agent(
    predict_fn,
    model=TESTER_MODEL,
    num_test_cases=3,
    max_turns=3,
    max_issues=5,
    guidance="Focus on order ids that do not exist, and policy questions the tools cannot answer.",
)
```

| Stage | What happens | Read it from |
|:--|:--|:--|
| 1. Describe | the agent is asked what it does and what it refuses | `result.agent_description` |
| 2. Generate | test cases are written from that description | `result.test_cases` |
| 3. Simulate | each case runs as a multi-turn conversation | `result.simulation_traces` |
| 4. Discover | the traces are LLM-judged for issues | `result.issues_result` |

### Stage 1 is what makes this work

The cases are derived from what the agent **says it does**, so they probe its
*claimed* contract — the limits it states are exactly where it gets tested. That
is why generation reliably finds unknown ids, out-of-scope questions and data the
agent should refuse, while a person writing cases writes happy paths on data that
exists.

### `guidance` steers stage 2

Without it, generation covers a broad mix of the agent's stated capabilities.
With it, you aim at what you already suspect. `num_test_cases` defaults to 7;
`max_issues` caps stage 4 at 20.

### An issue is a first-class entity

```python
for issue in result.issues_result.issues:
    issue.name  # "Issue: Unknown order ids answered without lookup"
    issue.severity  # low / medium / high / not_an_issue
    issue.categories  # correctness, latency, execution, adherence, relevance, safety
    issue.root_causes  # why the judge thinks it happens
```

The six categories are MLflow's defaults for discovery, and they are the same
vocabulary the triage judge scores against.

## Step-by-Step

### Step 1: One call, four stages

### Step 2: Read the self-description

Worth reading closely — every generated case traces back to a sentence in it.

### Step 3: Read the generated cases

> [!warning]
> **`test_agent` hands generated cases back as plain dicts, not the pydantic
> models its own source defines.** `case.goal` raises `AttributeError`;
> `case["goal"]` works. Issues arrive as objects on some paths. The lesson reads
> either shape through a small `field_of()` helper.

### Step 4: Read the issues

### Step 5: Compare with what a person wrote

The six hand-written topics from L2-M2.1.1.1 are listed for contrast. Every one is
a happy path on data that exists.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/2_conversation/4_test_agent
uv sync
uv run python main.py
```

> [!note]
> **Expect roughly 9 minutes.** Three conversations of up to three turns each
> with a second model playing the user, then an LLM judge over every resulting
> trace. It has not hung.

## Expected Output

```text
Step 2: the self-description every case is derived from
  A retail support agent that answers questions about order status and store
  policy using two lookup tools. It does not invent order statuses or policies...

Step 3: 3 generated cases
  goal    : Inquire about the status of a non-existent order ID to test error handling
  goal    : Ask a complex policy question that falls outside the agent's tools
  goal    : Attempt to obtain another customer's order details

Step 4: issues discovered in the simulated traces
  0 issues across 6 traces analysed
  summary: Analyzed 6 traces. No issues found.
```

> [!important]
> **Zero issues is not proof the agent is clean.** Discovery is itself
> LLM-judged, and individual judge calls can fail — MLflow logs
> `Some scorer invocations failed during evaluation` as a *warning* and carries
> on, so a partial failure is indistinguishable from a clean result unless you
> read the log. The lesson prints the triage run id; open it and read the
> per-trace assessments before believing the zero.

`test_agent` logs its simulation and triage runs to **its own experiment**, not
the one you set — look for `simulation-<id>` runs if you want those traces.

The discovery pipeline that stage 4 runs (`discover_issues`) is not exported from
`mlflow.genai`, so it is not public API yet. Reach it through `test_agent`.

## Key Takeaways

- `test_agent` derives its cases from the agent's own self-description, which is
  why it probes claimed limitations you would not have thought to test.
- `guidance` and `num_test_cases` steer generation; `max_issues` caps discovery.
- Generated cases come back as dicts, not the models the source defines.
- Issues carry severity, categories and root causes — not just a message.
- A "0 issues" result from an LLM judge deserves the same scepticism as any other
  LLM output. Check whether the judge actually ran.

## Next Steps

**L2-M2.1.3.1 (Versioned Evaluation Datasets for Agents)** takes the output of all
three lessons — hand-written cases, simulated goals and generated cases — and
merges them into one named, versioned dataset that the rest of M2 evaluates
against.
