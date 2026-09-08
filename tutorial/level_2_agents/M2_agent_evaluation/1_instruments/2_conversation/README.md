# L2-M2.1.2 — Conversation-Level Instruments

**Many turns, one session.** One conversation is one session and several traces
— one trace per turn, tied together by a `session_id`.

## One engine, three sources, two verdicts

There is only ever **one** direction of execution. A goal becomes a
conversation. What changes between lessons is where the goal came from.

```text
    WHERE THE GOAL COMES FROM              THE ENGINE            WHAT YOU DO WITH IT
 ┌──────────────────────────────────┐  ┌───────────────┐  ┌──────────────────────────┐
 │ 1  you write it          [L1]    │  │               │  │ discover_issues()        │
 │                                  │  │ Conversation  │  │    -> issues, in words   │
 │ 2  generate_test_cases() [L2]    │─▶│  Simulator    │─▶├──────────────────────────┤
 │      from real traffic           │  │  .simulate()  │  │ session scorers    [L3]  │
 │                                  │  │               │  │    -> scores, in numbers │
 │ 3  the agent describes   [L4]    │  └───────────────┘  └──────────────────────────┘
 │      itself                      │          │
 └──────────────────────────────────┘   traces / sessions
```

`test_agent()` (lesson 4) is **source 3 + the engine + `discover_issues()`** in
one call. It reuses lesson 1's simulator and lesson 3's issue discovery. It
never touches lesson 2.

## The four lessons

| # | Lesson | You give it | You get back | Does it judge? |
|:--|:--|:--|:--|:--|
| 1 | [`1_conversation_simulation`](1_conversation_simulation/) | a goal, and optionally one persona | conversations | No |
| 2 | [`2_goals_from_real_sessions`](2_goals_from_real_sessions/) | existing conversations | goals and personas | No |
| 3 | [`3_judging_conversations`](3_judging_conversations/) | conversations | issues in words, scores in numbers | Yes |
| 4 | [`4_test_agent`](4_test_agent/) | only the agent | conversations and issues | Issues only, never scores |

Read the "you give it" column down: you write the goal, then old traffic writes
it, then nobody writes it.

## Two corrections worth reading before lesson 1

> **`persona` is one person.** The simulator has exactly two sides — one
> simulated user and your agent. `persona: str` describes that single user. It
> is not a participant list, and there is no multi-party mode.

> **A simulator test case cannot pass or fail.** It holds a `goal`, not a right
> answer. `goal` is the only required key; `persona`, `context`, `expectations`
> and `simulation_guidelines` are optional. The simulator does ask an LLM
> "has the goal been achieved?" — but only to decide when to **stop**, and that
> answer never becomes a score.

## What only conversation scope can ask

- Was the task ever finished? — `ConversationCompleteness`
- Did turn 4 remember turn 1? — `KnowledgeRetention`
- Is the user getting annoyed? — `UserFrustration` (good at **0.0**, not 1.0)
- Was the same tool called again and again across turns? — `ConversationalToolCallEfficiency`
- Did the agent stay in role? — `ConversationalRoleAdherence`
- Was the whole thread safe? — `ConversationalSafety`
- Did it follow your written rules? — `ConversationalGuidelines`

All seven take `model=`, so all seven are judges. A `@scorer` whose parameter is
named `session` becomes session-level too, and it needs no LLM at all.

## Two traps that bite silently

1. **Every trace in a session needs a `session_id`.** Collecting traces is not
   enough — the scorer raises `All traces in 'session' must have a session_id`.
   Wrap each turn so you own the root span, then call
   `mlflow.update_current_trace(session_id=...)`.
2. **These judges answer with strings.** `"yes"`, `"no"`, `"none"`. Coercing
   with `bool(value)` scores every failure as a pass, because `bool("no")` is
   `True`. And the polarity is not uniform — never sum them into one score.

## Next

**[`../3_dataset_store/`](../3_dataset_store/)** — keeping what these lessons
produce.
