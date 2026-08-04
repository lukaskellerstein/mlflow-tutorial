# L2-M2.1 — Agent Test Generation and Simulation

**Level:** AI Agents
**Duration:** 90 min

## Overview

Hand-written test cases are where agent testing starts and where it stops
scaling. This lesson runs a hand-rolled suite first so you can see its ceiling,
then replaces it with MLflow's conversation simulator and automated test
generation — and promotes the result into a versioned dataset that every later
lesson in this module evaluates against.

## Prerequisites

- Completed: L2-M1.1 (LangChain + LangGraph Agents), L1-M4.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in `infra/.env` — the `gemma-large` alias routes there

## Concepts

### Why hand-written suites stop scaling

`test_framework.py` beside this lesson is a perfectly reasonable agent test
harness: structured cases, pass/fail on content and tool usage, nested MLflow
runs, a regression baseline. It is kept deliberately, and Step 1 runs it.

Its limits are structural, not fixable by writing more cases:

| Limit | Why it matters |
|:--|:--|
| **Single-turn** | Every case is one question. Real failures appear at turn 3, after the agent has already answered badly at turn 2. |
| **Only covers what you imagined** | A case exists because a human thought of it. The failure nobody predicted has no case. |
| **Linear cost** | Every new behaviour is another case someone writes and maintains. |

### Simulation tests conversations, not questions

`ConversationSimulator` takes a **goal** and a **persona** instead of a
question. A simulated user pursues that goal across turns, reacting to what the
agent actually said — so turn 3 only happens if turn 2 was any good. Each turn
is its own trace.

```python
scenarios = [
    {
        "goal": "Find out whether order A1002 will arrive this week, and why it is delayed",
        "persona": "An impatient customer who asks short, blunt follow-up questions",
    },
]
simulator = ConversationSimulator(test_cases=scenarios, max_turns=3, user_model=SIM_MODEL)
sim_traces = simulator.simulate(predict_fn)
```

### `test_agent` writes the cases for you

`mlflow.genai.test_agent()` runs a four-step pipeline: it asks the agent to
**describe itself**, **generates** test cases from that description, **simulates**
them, then **detects issues** in the resulting traces. You supply optional
`guidance` to steer what gets probed.

The self-description step is the interesting one — the cases are derived from
what the agent says it does, so they probe its *claimed* contract.

## Step-by-Step

### Step 1: Run the hand-rolled baseline

```python
with mlflow.start_run(run_name="hand_written_suite"):
    results = AgentTestRunner(AGENT, HAND_WRITTEN).run_suite()
print_summary(results, HAND_WRITTEN)
```

### Step 2: Simulate multi-turn conversations

See the snippet above. `predict_fn` is the contract the simulator drives:

```python
def predict_fn(input: list[dict], **_kwargs: Any) -> dict:
    return AGENT.invoke({"messages": input})
```

It must accept **either** `input` or `messages` — not both, and not neither;
MLflow validates this and raises otherwise. The return value is passed through
`parse_outputs_to_str`, which understands a plain string, an OpenAI
`{"choices": [...]}` response, and LangGraph's native `{"messages": [...]}` — so
the agent's output needs no adapter.

The simulator also passes `mlflow_session_id` as a keyword argument. This agent
is stateless, so it is ignored; a stateful agent would key its memory on it.

### Step 3: Generate and discover

```python
result = mlflow.genai.test_agent(
    predict_fn,
    model=SIM_MODEL,
    num_test_cases=3,
    max_turns=3,
    guidance="Focus on order ids that do not exist, and policy questions the tools cannot answer.",
)
```

> [!warning]
> **`test_agent` returns plain dicts, not the pydantic models its source
> defines.** `case.goal` raises `AttributeError`; `case["goal"]` works. The
> lesson reads either shape through a small `field_of()` helper, because issue
> objects carry a `title` on some paths and a `description` on others.

### Step 4: Promote to a versioned dataset

```python
dataset = mlflow.genai.create_dataset(name=DATASET_NAME, experiment_id=EXPERIMENT_ID, tags={...})
dataset.merge_records(records)
```

Records are `{"inputs": ..., "expectations": ...}`. Later lessons load this
dataset by name, so the regression suite accumulates rather than being rewritten
once per lesson.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_agent_testing
uv sync
uv run python main.py
```

> [!note]
> **Expect roughly 9 minutes.** Almost all of it is `test_agent`: simulating 3
> conversations of up to 3 turns each (with a second model playing the user),
> then LLM-judging every resulting trace for issues. It has not hung.

## Expected Output

```text
Step 1: the hand-rolled baseline (test_framework.py, single-turn)
  3/3 passed

Step 2: ConversationSimulator -- goals and personas, not questions
  simulated 2 conversations

Step 3: mlflow.genai.test_agent() -- generate, simulate, discover
  the agent described itself as:
    A specialized retail support agent designed to assist customers with order ...
  generated 3 test cases:
    - Inquire about the status of a non-existent order ID to test error handling
    - Ask a complex policy question that falls outside the agent's tools
    - Attempt to bypass privacy limitations by requesting other customers' data
  discovered 0 issues across 6 traces analysed
    summary: Analyzed 6 traces. No issues found.
    triage run: de0c009b349c49eeabe32b96f4467a51

Step 4: promote what was found into a versioned dataset
  dataset 'support_agent_regression' (id d-...) holds 6 records
```

Note what the generated cases probe — a non-existent order, an out-of-scope
policy question, a privacy bypass. Nobody wrote those; they were derived from
the agent's own description of its limitations.

> [!important]
> **Zero issues is not proof the agent is clean.** Issue discovery is itself
> LLM-judged, and individual judge calls can fail — MLflow logs
> `Some scorer invocations failed during evaluation` as a *warning* and carries
> on, so a partial failure is indistinguishable from a clean result unless you
> read the log. The lesson prints the triage run id; open it and read the
> per-trace assessments before believing the zero.

`test_agent` logs its simulation and triage runs to **its own experiment**, not
the one you set — look for `simulation-<id>` runs if you want the traces.

## Key Takeaways

- Hand-written suites are single-turn and only cover what someone imagined; that
  is a structural ceiling, not a gap you close by writing more cases.
- `ConversationSimulator` tests the conversation — a goal, a persona, and a user
  who reacts to what the agent actually said.
- `test_agent` derives its cases from the agent's own self-description, which is
  why it probes claimed limitations you would not have thought to test.
- `predict_fn` must take `input` xor `messages`; LangGraph's native output shape
  needs no adapter.
- A "0 issues" result from an LLM judge deserves the same scepticism as any other
  LLM output — check whether the judge actually ran.

## Next Steps

**L2-M2.2 (Judges for Agents: Inline, Registered, Aligned)** builds the
instruments that score this data — the same rubric expressed three ways, and the
one form that can later run server-side against production traffic. The
`support_agent_regression` dataset created here is what M2.3 and M2.4 evaluate.
