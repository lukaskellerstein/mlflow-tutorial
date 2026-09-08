# L2-M1 — Agent Frameworks

Three frameworks, built and traced with MLflow. Six lessons in two groups, and
the same three frameworks appear in both.

Before any of them, one picture. **Scope** is how much of the interaction one
run covers, and it decides what the agent is allowed to know.

```text
   1_turn/                          2_conversation/

   ┌───────────┐  ┌───────────┐     ┌───────────────────────────────┐
   │ task 1    │  │ task 2    │     │ turn 1  "what is 15 * 23?"    │
   │  answer   │  │  answer   │     │ turn 2  "now double that"  <──┼── needs turn 1
   └───────────┘  └───────────┘     │ turn 3  "reverse 'MLflow'"    │
                                    │ turn 4  "what did I ask 1st?" │
   nothing carried between          └───────────────────────────────┘
                                       ONE session, state carried
```

## The two groups

| Group | What one run covers | The question it answers |
|:--|:--|:--|
| [`1_turn/`](1_turn/) | one task, start to finish | What does this framework give me, and how does MLflow see it? |
| [`2_conversation/`](2_conversation/) | many turns, one session | How does state survive a turn boundary, and how do I see the whole discussion? |

Do the turn lessons first. The conversation lessons are deliberately short: they
teach only what changes, and assume you have already met the framework.

## The two keys

Every conversation lesson turns on one idea, so it is stated once here rather
than three times below.

A conversation needs two separate things, owned by different systems:

| Key | Owner | What it decides |
|:--|:--|:--|
| the thread key | the agent framework | what the **agent** remembers |
| `session_id` | MLflow | which traces belong to **one conversation** |

They are independent. Set only the first and the agent answers correctly, but
MLflow shows you four unrelated traces. Set only the second and MLflow groups
them neatly, while the agent has amnesia.

Set both, to the same value, and you can follow one conversation from the user's
browser through the agent and into the trace store. That is the pattern every
lesson in `2_conversation/` uses.

## The frameworks

| Lesson | Framework | Memory mechanism | Who picks the session key |
|:--|:--|:--|:--|
| `1_langchain_langgraph` | LangChain v1 + LangGraph | checkpointer, keyed by `thread_id` | you |
| `2_deepagents` | DeepAgents | the same checkpointer — and it carries todos and files too | you |
| `3_claude_agent_sdk` | Claude Agent SDK | the open `ClaudeSDKClient`, plus `resume` | **the SDK** |

The last row is why the third lesson exists in both groups. Every other
framework here makes you invent a key and push it down. The Claude Agent SDK
hands you one after the fact, so the code has to run the other way round.

## What you need running

| Lesson | Needs |
|:--|:--|
| `1_langchain_langgraph`, `2_deepagents` | MLflow, the MLflow AI Gateway, Unsloth Studio |
| `3_claude_agent_sdk` | MLflow, and a working Claude Code CLI login. **No gateway, and it spends real usage.** |

MLflow must be 3.15 or later for the `2_conversation/` lessons —
`mlflow.search_sessions()` and `mlflow.tracing.context()` are both newer than
3.0.

## Next

**L2-M2 — Agent Evaluation** judges what you have traced here, and splits by the
same axis: `1_turn/` scores one answer, `2_conversation/` scores a whole session
and reads it with the `mlflow.search_sessions()` you meet in `2_conversation/`.
