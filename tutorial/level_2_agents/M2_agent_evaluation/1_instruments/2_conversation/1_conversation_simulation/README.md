# L2-M2.1.2.1 — Conversation Simulation: the Engine

**Level:** AI Agents
**Duration:** 45 min

## Overview

A hand-written case is one question. A real user has a **goal** and pursues it
over several turns, reacting to what the agent actually said. This lesson puts a
second model in the user's seat with `ConversationSimulator`, and runs the engine
in the only direction it has: **a goal you write becomes a conversation**.

Where the goal comes from when you do *not* write it is the next two lessons.

## Prerequisites

- Completed: L2-M2.1.1.1 (Hand-Written Agent Test Suites)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>

## Concepts

### A goal is not a question

| Hand-written case | Simulated scenario |
|:--|:--|
| `"Is order A1002 on its way?"` | goal: *find out whether A1002 arrives this week, and why it is delayed* |
| one turn | up to `max_turns`, and turn 3 depends on turn 2 |
| the answer is fixed in advance | the conversation is generated, so no two runs are identical |

Three fields shape the simulated user:

- **`goal`** (required) — what the user is trying to achieve.
- **`persona`** — how they behave. *"An impatient customer who asks short, blunt
  follow-up questions"* produces a different conversation from a polite one.
- **`simulation_guidelines`** — constraints on the **user**, not the agent. The
  lesson uses *"Do not say the word 'warranty' unless the agent says it first"*,
  which forces the agent to surface the policy itself instead of being handed
  the keyword. That is the behaviour actually under test.

### The `predict_fn` contract

MLflow validates the signature before the first turn runs, so a mistake here
fails immediately rather than halfway through a simulation.

```python
def predict_fn(input: list[dict], **_kwargs: Any) -> dict:
    return AGENT.invoke({"messages": input})
```

| Rule | Detail |
|:--|:--|
| `input` **xor** `messages` | the conversation so far, as message dicts. Both, or neither, raises. |
| return a readable shape | a plain string, an OpenAI `{"choices": [...]}` response, or LangGraph's `{"messages": [...]}`. The third is what `create_agent` already returns, so no adapter is needed. |
| `mlflow_session_id` arrives as a kwarg | the same value for every turn of one conversation. A stateless agent ignores it; a thread-based agent keys its memory on it. |

### One conversation is one session

Each turn is its own trace. The session id is what ties them together, and it is
the unit a session-level scorer reads (L2-M2.1.1.3):

```python
sessions = mlflow.search_sessions(locations=[EXPERIMENT_ID], max_results=2)
for session in sessions:
    print(session.id, len(session))  # -> a session behaves like a list of traces
```

### What the simulator does NOT do

It grades nothing. It asks an LLM "has the goal been achieved?" only to decide
when to **stop** a conversation — that answer never becomes a score and never
reaches a run.

A simulator test case holds a `goal`, not a right answer, so there is nothing to
compare against. Judging what came out is a separate step, and a separate
lesson.

> [!note]
> You will see warnings reading `Could not parse response for goal achievement
> check`. Read one: the JSON is valid, and the model wrapped it in a ```json
> fence that MLflow's parser does not strip. Because that check only decides
> when to stop, a failed parse costs extra turns and nothing else.

## Step-by-Step

### Step 1–2: Write two scenarios and simulate them

```python
simulator = ConversationSimulator(test_cases=scenarios, max_turns=3, user_model=SIM_MODEL)
sim_traces = simulator.simulate(predict_fn)  # -> list[list[Trace]], one inner list per scenario
```

### Step 3: Look at the sessions

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/2_conversation/1_conversation_simulation
uv sync
uv run python main.py
```

> [!note]
> **Expect roughly 6 minutes.** Every turn is two model calls — one for the
> simulated user, one for the agent — plus tool calls. It has not hung.

## Expected Output

```text
Step 2: simulate, up to 3 turns each
  simulated 2 conversations

  goal: Find out whether order A1002 will arrive this week, and w
    turn 1  user  : Where's order A1002?
            agent : Order A1002 is held at the warehouse because payment is not...
    turn 2  user  : So when does it ship?
            agent : It will not ship until payment is confirmed...

Step 3: one conversation = one session = several traces
  session 8f2c1b0a3d4e5f6a   3 traces
  session 1a7d9e2b4c6f8a0b   3 traces

  conversations simulated : 2
  turns traced            : 6
  sessions                : 2
```

> [!important]
> **Simulation is non-deterministic by design.** The simulated user is an LLM,
> so no two runs produce the same conversation, and a suite of simulations is
> not a pass/fail gate on its own. It is a *trace generator* — the pass/fail
> comes from scoring those traces — turn scorers in
> [`../../1_turn/`](../../1_turn/), session scorers in
> [`../3_judging_conversations`](../3_judging_conversations/).

## Key Takeaways

- `ConversationSimulator` tests the conversation. A goal, a persona, and a user
  who reacts to what the agent actually said.
- `simulation_guidelines` constrain the simulated user, which is how you stop a
  test from handing the agent the answer.
- `predict_fn` takes `input` xor `messages`; LangGraph's native output shape
  needs no adapter, and `mlflow_session_id` is how a stateful agent keeps its
  thread.
- Every turn is a trace; the session ties them together and is the unit
  session-level scorers read.
- **`persona` is ONE user.** The simulator has exactly two sides, that user and
  your agent. There is no multi-party mode.
- **A simulator test case is a scenario, not an assertion.** `goal` is the only
  required key, and nothing here can pass or fail.

## Next Steps

**[`2_goals_from_real_sessions`](../2_goals_from_real_sessions/)** runs the same
engine, but the goal is distilled out of conversations that already happened —
so your users write the suite instead of you.
