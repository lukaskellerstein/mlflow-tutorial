# L2-M1.1 — Turn-Level Agent Frameworks

**One task in, one final answer out.** Each lesson here runs several tasks, but
nothing is carried from one to the next: every task starts from an empty message
list. Meet the framework first, then add memory in
[`2_conversation/`](../2_conversation/).

## The three lessons

| Lesson | Framework | What it is about |
|:--|:--|:--|
| [`1_langchain_langgraph`](1_langchain_langgraph/) | LangChain v1 + LangGraph | the same ReAct agent built twice — `create_agent`, then by hand as a `StateGraph` |
| [`2_deepagents`](2_deepagents/) | DeepAgents | the built-in toolkit, sub-agent delegation, and what a "file" is under each backend |
| [`3_claude_agent_sdk`](3_claude_agent_sdk/) | Claude Agent SDK | tracing a framework MLflow does not instrument, and two MCP transports |

## How MLflow sees each one

| Framework | Instrumentation | Cost of it |
|:--|:--|:--|
| LangChain, LangGraph | `mlflow.langchain.autolog()` | one line |
| DeepAgents | the same call — it is LangGraph underneath | one line |
| Claude Agent SDK | none exists; you build it | `@mlflow.trace` plus a manual span per tool call |

The third row is the reason the third lesson is worth its time. Every framework
you adopt that MLflow has not integrated looks like that, and the pattern
transfers unchanged.

## Do these in order

`2_deepagents` builds on the agent from `1_langchain_langgraph`, and
`3_claude_agent_sdk` assumes you have seen what autolog does before it asks you
to replace it by hand.
