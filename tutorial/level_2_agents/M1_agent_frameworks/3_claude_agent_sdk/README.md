# L2-M1.3 — Claude Agent SDK

**Level:** AI Agents
**Duration:** 90 min

## Overview

The first two lessons in this module got tracing for free: LangChain, LangGraph
and DeepAgents all run on one runtime, and one `mlflow.langchain.autolog()` call
covers them. The Claude Agent SDK has no such hook. This lesson builds the
integration by hand — `@mlflow.trace` for the root span, `mlflow.start_span()`
per event — which is the pattern for **any** framework MLflow does not ship
support for.

It also shows both ways to give the SDK tools, because they trace differently:
an in-process MCP server built with `@tool`, and an external FastMCP process over
STDIO.

## Prerequisites

- Completed: L2-M1.1, L2-M1.2, and L1-M2.1 (auto vs manual tracing)
- MLflow server running at <http://127.0.0.1:5555>
- Claude Code CLI installed and logged in (`claude --version`)
- **This lesson makes real Anthropic API calls and costs real money** — roughly
  $2 for a full run at the time of writing. It is the only lesson in this module
  that does; the other two go through the local LiteLLM gateway.

## Concepts

### Why this one is different

The SDK drives the Claude Code CLI, which speaks Anthropic's Messages API. The
LiteLLM gateway from `infra/` speaks the OpenAI API. They do not meet, so this
lesson does **not** use `gemma-large` — it uses whatever your Claude Code CLI is
authenticated as. That mismatch is the lesson: not every agent framework fits
behind one endpoint, and the tracing you write has to survive that.

### The manual tracing pattern

```text
@mlflow.trace            -> the root span for one agent run
mlflow.start_span()      -> a child span per tool call in the message stream
nested mlflow runs       -> per-query params and metrics
the SDK's own numbers    -> turns, duration_ms, total_cost_usd
```

The last line matters: `ResultMessage` carries cost and turn counts that nothing
else in your process knows. Manual instrumentation is not only about spans, it is
about capturing what the framework alone can tell you.

### Two MCP transports

| | in-process | external |
|:--|:--|:--|
| Built with | `@tool` + `create_sdk_mcp_server()` | a FastMCP script (`server.py`) |
| Runs in | this Python process | a subprocess over STDIO |
| Tool name | `mcp__inproc__word_stats` | `mcp__tools__calculator` |
| Startup | immediate | **lazy — see the gotchas below** |

### Two gotchas that fail silently

Both cost real money to discover, so they are called out rather than left as
exercises. Neither raises an exception; both simply produce an agent with fewer
tools than you configured, and a run that exits 0.

**1. `strict_mcp_config=True` is required.** By default the CLI *merges* the
servers you pass with any `.mcp.json` in the working directory. This repo has
one. Those project servers arrive unapproved, and your external server then never
becomes available — the model answers "there is no calculator tool" and computes
the answer itself.

**2. The external server connects lazily.** `async with ClaudeSDKClient(...)`
returns before the subprocess has registered its tools. Query immediately and
roughly half the time the model sees only the in-process tools. The SDK exposes
no readiness signal — `get_mcp_status()` returns an empty list under strict
config — so the lesson waits a fixed `MCP_SETTLE_SECONDS = 3.0` after connecting.

## Step-by-Step

### Step 1: Define an in-process tool

```python
@tool("word_stats", "Count words and characters in a piece of text", {"text": str})
async def word_stats(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    stats = {"words": len(text.split()), "characters": len(text)}
    return {"content": [{"type": "text", "text": json.dumps(stats)}]}


INPROC_SERVER = create_sdk_mcp_server(name="inproc", version="1.0.0", tools=[word_stats])
```

### Step 2: Wire both servers, with strict config

```python
ClaudeAgentOptions(
    mcp_servers={
        "inproc": INPROC_SERVER,
        "tools": {"type": "stdio", "command": sys.executable, "args": [SERVER_PATH]},
    },
    allowed_tools=[
        "mcp__inproc__word_stats",
        "mcp__tools__calculator",
        "mcp__tools__knowledge_lookup",
    ],
    permission_mode="bypassPermissions",
    strict_mcp_config=True,
)
```

### Step 3: Trace the run by hand

```python
@mlflow.trace(name="claude_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> AgentResult:
    async with ClaudeSDKClient(options=options) as client:
        await anyio.sleep(MCP_SETTLE_SECONDS)
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes({"tool_name": block.name, ...})
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M1_agent_frameworks/3_claude_agent_sdk
uv sync
uv run python main.py
```

## Expected Output

Each query exercises a different tool path:

```text
  Query 1: Use the calculator tool to compute 4217 * 1793 + 88.
    Tool: mcp__tools__calculator({"expression": "4217 * 1793 + 88"})
  Query 2: Use the knowledge_lookup tool to look up 'mlflow' and report exactly what it returns.
    Tool: mcp__tools__knowledge_lookup({"topic": "mlflow"})
  Query 3: Use the word_stats tool on this sentence: 'Agent evaluation needs real traces'.
    Tool: mcp__inproc__word_stats({"text": "Agent evaluation needs real traces"})
  Query 4: Explain why testing AI agents is important.
```

The queries name their tool explicitly on purpose. Without that, Claude answers
arithmetic and general questions from its own knowledge, the external server is
never touched, and the transport comparison has nothing to compare.

Trace analysis prints the hand-built span hierarchy:

```text
  Trace tr-2ff14e77a6a7d... | TraceStatus.OK | 8010ms
    - claude_agent.run (8010ms)
    - tool_call.mcp__inproc__word_stats (0ms)
```

Tool spans show ~0ms because the SDK reports the *decision* to call a tool, not
its execution — the tool itself runs inside the CLI. The span records the name
and arguments, which is what you have to work with.

In the MLflow UI under **L2/M1_agent_frameworks/3_claude_agent_sdk**: one parent
run with four nested query runs carrying `cost_usd` and `num_turns`, a
`results.json` artifact, and the traces above.

## Key Takeaways

- No autolog is not a blocker: `@mlflow.trace` plus `mlflow.start_span()`
  reproduces the essentials for any framework.
- Log what only the framework knows — turns, duration, cost.
- In-process and external MCP servers behave differently at startup, and the
  external one's failure is silent.
- `strict_mcp_config=True` whenever you supply MCP servers programmatically,
  or the developer's `.mcp.json` changes your agent's tool set.
- A framework that does not fit your gateway is normal; plan the instrumentation
  around it.

## Next Steps

**L2-M2 — Agent Evaluation** turns from tracing agents to scoring them: test
generation, quality metrics, architecture comparison, optimization, and a full
evaluation pipeline.
