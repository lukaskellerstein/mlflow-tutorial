# L2-M2.1 — Claude Agent SDK + MLflow Integration

**Level:** AI Agents
**Duration:** 90 min

## Overview

This lesson demonstrates how to build a custom MLflow tracing integration for the Claude Agent SDK. The SDK has no native MLflow support, so we wrap agent execution with `@mlflow.trace` and manual spans to capture the full lifecycle — tool calls, responses, cost, and duration.

The lesson uses an external FastMCP server (STDIO transport) for custom tools and the real Claude Agent SDK for agent execution.

## Prerequisites

- Completed: L1-M2.1 (Auto-Tracing and Manual Tracing)
- MLflow server running at <http://127.0.0.1:5555>
- Claude Code CLI installed (`claude` command available)
- Anthropic API key configured (via Claude Code subscription or `ANTHROPIC_API_KEY`)
- **Note**: This lesson makes real API calls and incurs costs

## Concepts

### Claude Agent SDK

The Claude Agent SDK is Anthropic's framework for building autonomous agents. It wraps the Claude Code CLI as a subprocess and provides:
- `ClaudeSDKClient` — stateful, multi-turn agent sessions
- `query()` — simple, stateless one-shot queries
- Custom tools via MCP (Model Context Protocol) servers
- Hooks for intercepting tool execution

### External MCP Server

Instead of in-process tools, this lesson uses an external FastMCP server (`server.py`) that runs as a separate STDIO process. The Claude Agent SDK spawns it automatically via the `mcp_servers` config. This matches production patterns where tools live in separate services.

### Why Custom Integrations?

MLflow provides autologging for LangChain, OpenAI, and Anthropic, but many agent frameworks lack native support. The pattern here — `@mlflow.trace` + `mlflow.start_span()` — works for any framework.

## Step-by-Step

### Step 1: MCP Server (server.py)

The FastMCP server defines two tools with simple return types:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tutorial_tools")


@mcp.tool()
def calculator(expression: str) -> dict:
    result = eval(expression)
    return {"expression": expression, "result": result}


@mcp.tool()
def knowledge_lookup(topic: str) -> dict:
    return {"topic": topic, "fact": facts.get(topic, "Not found")}
```

### Step 2: SDK Configuration

Connect to the MCP server via STDIO and configure the agent:

```python
options = ClaudeAgentOptions(
    tools=[],  # no built-in tools (Bash, Read, etc.)
    mcp_servers={
        "tools": {
            "command": sys.executable,
            "args": [SERVER_PATH],
        },
    },
    allowed_tools=[
        "mcp__tools__calculator",
        "mcp__tools__knowledge_lookup",
    ],
    max_turns=3,
    permission_mode="dontAsk",
)
```

### Step 3: Traced Agent Execution

Wrap the agent call with `@mlflow.trace` and create child spans for tool calls:

```python
@mlflow.trace(name="claude_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> AgentResult:
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                    elif isinstance(block, TextBlock):
                        response_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                duration_ms = message.duration_ms
                cost_usd = message.total_cost_usd
```

### Step 4: Nested MLflow Runs

Each query gets its own nested run with metrics:

```python
with mlflow.start_run(run_name="claude_sdk_integration"):
    for query in queries:
        with mlflow.start_run(run_name=f"query_{i}", nested=True):
            result = await run_agent(query, options)
            mlflow.log_metrics({"duration_ms": result.duration_ms, ...})
```

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_custom_integrations/1_claude_agent_sdk
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
L2-M2.1 — Claude Agent SDK + MLflow Integration
============================================================

Part 1: MCP server + SDK config
  MCP server: /path/to/server.py
  Tools: calculator, knowledge_lookup
  Transport: STDIO (external FastMCP process)

Part 2: Traced agent execution
  Response: The result of 123 * 456 is 56088.
  Tools used: ['calculator']
  Duration: 3200ms, Model: claude-sonnet-4-5-20250514

Part 3: Running example queries

  Query 1: What is 42 * 17 + 3?
    Tool: calculator({"expression": "42 * 17 + 3"})
    Response: The result is 717...
    Duration: 2800ms, Turns: 2

  Query 2: Tell me about MLflow.
    Tool: knowledge_lookup({"topic": "mlflow"})
    Response: MLflow is an open-source MLOps platform...
    Duration: 3100ms, Turns: 2

  Query 3: Explain why testing AI agents is important.
    Response: Testing AI agents is important because...
    Duration: 2500ms, Turns: 1

Part 4: Trace analysis
  Found 4 traces
  Trace a1b2c3d4e5f6... | OK | 3200ms
    - claude_agent.run (3200ms)
    - tool_call.calculator (5ms)
```

In the MLflow UI:
- **Experiment**: L2/M2_custom_integrations/1_claude_agent_sdk
- **Runs**: Parent run with 3 nested child runs
- **Traces tab**: Span hierarchy with tool call details
- **Artifacts**: `claude_agent_results.json` summary

## Key Takeaways

- **Any agent SDK can be integrated** with MLflow using `@mlflow.trace` and `mlflow.start_span()`
- **External MCP servers** (FastMCP + STDIO) provide tools as separate processes, matching production patterns
- **ClaudeSDKClient** is required for MCP tool use — `query()` does not support custom tools
- **Capture structured data**: use `set_inputs()`, `set_outputs()`, `set_attributes()` on spans
- **Log aggregate metrics**: span-level detail for debugging, run-level metrics for comparison

## Next Steps

Continue to **L2-M2.2 (DeepAgents + MLflow)** to see the same integration pattern applied to a multi-agent orchestration framework.
