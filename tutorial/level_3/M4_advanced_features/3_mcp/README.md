# L3-4.3 — MLflow MCP (Model Context Protocol) Integration

**Level:** Expert
**Duration:** 1 hour

## Overview

Model Context Protocol (MCP) is a standard that lets AI applications discover and use external tools through a unified interface. MLflow implements MCP natively, exposing trace, experiment, run, and scorer operations as MCP tools that AI assistants can call. This lesson builds an MCP-style tool server and client from scratch, traces every interaction with MLflow, and analyzes the overhead of the MCP protocol pattern.

## Prerequisites

- Completed: L1-M5 (Tracing), L2-M4 (Advanced Tracing)
- MLflow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### What is MCP?

Model Context Protocol (MCP) standardizes how AI models connect to external tools and data sources. Instead of hard-coding tool integrations, an MCP client can:

1. **Discover** tools from any MCP-compatible server (`tools/list`)
2. **Understand** each tool via its JSON Schema description
3. **Call** tools with validated arguments (`tools/call`)
4. **Receive** structured results

This decouples the AI application from the tools it uses — you can swap, add, or remove tools without changing the client code.

### MLflow's MCP Server

MLflow ships a built-in MCP server (`mlflow mcp run`) that exposes MLflow operations as tools. Categories include:

- **traces** — search and retrieve traces
- **scorers** — list and manage scorers
- **experiments** — create, search, and manage experiments
- **runs** — search and manage runs
- **models** — serve and manage models
- **deployments** — manage deployment endpoints

AI assistants (e.g., Claude Desktop) can connect to this server and query MLflow data directly during conversations.

### MCP in the Tracing Context

Every MCP interaction — discovery, tool selection, tool execution — is a distinct step that can be traced. This lesson captures the full MCP flow as MLflow spans, showing how protocol overhead maps to observable latency.

## Step-by-Step

### Step 1: Define MCP-Style Tools

Each tool follows the MCP convention: a name, description, JSON input schema, and a handler function. This mirrors how MLflow's own MCP server converts Click commands to MCP tools.

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]
```

Three tools are registered: `calculator`, `weather_lookup`, and `knowledge_base_search`.

### Step 2: Build the Tool Server

The `ToolServer` class exposes two MCP-style operations, both decorated with `@mlflow.trace`:

- `list_tools()` — returns tool manifests (name, description, schema)
- `call_tool(name, arguments)` — invokes a tool and returns the result

```python
@mlflow.trace(name="mcp_server.list_tools")
def list_tools(self) -> list[dict[str, Any]]:
    ...

@mlflow.trace(name="mcp_server.call_tool")
def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    ...
```

### Step 3: Build the MCP Client

The `MCPClient` discovers tools from the server, uses an LLM (ChatOllama) to select the right tool for a natural-language query, and calls it:

1. `discover_tools()` — fetches the tool manifests from the server
2. `select_tool(query)` — asks the LLM to return `{"tool": ..., "arguments": ...}`
3. `execute_query(query)` — full flow: discover, select, call, with timing

### Step 4: Run Traced Interactions

Four queries demonstrate different tool combinations. Each query creates a nested MLflow run with:

- Tool selected and arguments used
- Per-phase timing (discovery, selection, call)
- Success/failure status

### Step 5: Analyze MCP Overhead

The lesson compares MCP total latency (discovery + LLM selection + call) against the raw tool call time, quantifying the protocol overhead. The overhead is dominated by LLM tool selection — the value-add being that the client automatically routes to the right tool without hard-coded logic.

## Running the Lesson

```bash
cd tutorial/level_3/M4_advanced_features/3_mcp
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-4.3 — MLflow MCP (Model Context Protocol) Integration
============================================================

Part 1: MCP Concept Overview
  [explanation of MCP protocol]

Part 2: MCP-Style Tool Server
  Registered 3 tools:
    - calculator: Evaluate a mathematical expression ...
    - weather_lookup: Get current weather conditions ...
    - knowledge_base_search: Search an internal knowledge base ...

Part 3: MCP-Style Client with LLM Tool Selection
  Client discovered 3 tools from server

Part 4: MCP Interactions (fully traced in MLflow)
  Query 1: What is 145 * 23 + 17?
    Tool:   calculator
    Result: {"result": 3352, "expression": "145 * 23 + 17"}
    Timing: 1523ms total (discover=0ms, select=1520ms, call=1ms)

  ...

Part 5: MCP Metrics and Analysis
  Success rate:         100%
  Avg total latency:    1450.2ms
  Avg discovery time:   0.1ms
  Avg LLM selection:    1445.0ms
  Avg tool call:        0.5ms
  Average MCP overhead: 1449.7ms
```

In the MLflow UI you will see:
- A parent run `mcp_interactions` with aggregate metrics
- Nested child runs for each query with per-query timing
- Traces showing the full span tree: `execute_query` > `discover_tools` > `list_tools` > `select_tool` > `call_tool`

## Key Takeaways

- MCP standardizes tool discovery and invocation so AI clients can use tools without hard-coded integrations
- MLflow ships a built-in MCP server (`mlflow mcp run`) exposing traces, experiments, runs, and scorers as tools
- Every step of the MCP flow (discover, select, call) maps to an MLflow span, making the protocol fully observable
- MCP overhead is dominated by LLM tool selection; the raw protocol cost (discovery + call routing) is negligible
- Tracing MCP interactions in MLflow lets you measure and optimize tool routing latency, success rates, and tool usage patterns

## Next Steps

Continue to **L3-4.4 — Advanced Data Management** to learn about dataset versioning, data lineage, and feature store integration patterns at production scale.
