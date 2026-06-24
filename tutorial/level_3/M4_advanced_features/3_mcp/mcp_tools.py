"""
MCP-style tool server and client implementation.

Provides ToolSchema, ToolServer, and MCPClient classes that simulate
the Model Context Protocol pattern with full MLflow tracing.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------------
# Tool schema and handler definitions
# ---------------------------------------------------------------------------
@dataclass
class ToolSchema:
    """Standardized tool definition following MCP conventions."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]


def calculator_handler(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return json.dumps({"error": f"Invalid characters in: {expression}"})
    try:
        result = eval(expression)
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e)})


def weather_lookup_handler(city: str) -> str:
    """Simulated weather lookup for demonstration."""
    weather_data = {
        "new york": {"temp_f": 72, "condition": "Partly cloudy", "humidity": 55},
        "london": {"temp_f": 61, "condition": "Overcast", "humidity": 78},
        "tokyo": {"temp_f": 82, "condition": "Sunny", "humidity": 60},
        "paris": {"temp_f": 68, "condition": "Light rain", "humidity": 85},
    }
    key = city.lower().strip()
    if key in weather_data:
        return json.dumps({"city": city, **weather_data[key]})
    return json.dumps({"city": city, "error": "City not found"})


def knowledge_base_search_handler(query: str) -> str:
    """Simulated knowledge base search."""
    kb = {
        "mlflow": "MLflow is an open-source platform for the ML lifecycle, "
                  "including experiment tracking, model registry, and deployment.",
        "mcp": "Model Context Protocol (MCP) is a standard for connecting AI "
               "models to external tools and data sources via a unified interface.",
        "tracing": "Distributed tracing captures the full execution path of a "
                   "request across services, recording latency and metadata.",
    }
    results = []
    for topic, content in kb.items():
        if topic in query.lower():
            results.append({"topic": topic, "content": content, "score": 0.95})
    if not results:
        results.append({"topic": "general", "content": "No specific match.", "score": 0.1})
    return json.dumps({"query": query, "results": results})


# ---------------------------------------------------------------------------
# MCP-style tool server
# ---------------------------------------------------------------------------
class ToolServer:
    """Simulates an MCP tool server exposing tools via a standardized interface.

    In a real MCP setup this would be a separate process communicating over
    stdio or HTTP. Here we simulate the protocol in-process.
    """

    def __init__(self, name: str = "mcp-tool-server"):
        self.name = name
        self._tools: dict[str, ToolSchema] = {}

    def register_tool(self, tool: ToolSchema) -> None:
        self._tools[tool.name] = tool

    @mlflow.trace(name="mcp_server.list_tools")
    def list_tools(self) -> list[dict[str, Any]]:
        """MCP tools/list -- return tool manifests (no handlers)."""
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in self._tools.values()
        ]

    @mlflow.trace(name="mcp_server.call_tool")
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """MCP tools/call -- invoke a tool and return the result."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        try:
            raw = self._tools[name].handler(**arguments)
            return {"tool": name, "result": json.loads(raw)}
        except Exception as e:
            return {"tool": name, "error": str(e)}


def build_server() -> ToolServer:
    """Create and populate the MCP tool server with three tools."""
    server = ToolServer()
    server.register_tool(ToolSchema(
        name="calculator",
        description="Evaluate a mathematical expression and return the numeric result.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"],
        },
        handler=calculator_handler,
    ))
    server.register_tool(ToolSchema(
        name="weather_lookup",
        description="Get current weather conditions for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        handler=weather_lookup_handler,
    ))
    server.register_tool(ToolSchema(
        name="knowledge_base_search",
        description="Search an internal knowledge base for information on a topic.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
        handler=knowledge_base_search_handler,
    ))
    return server


# ---------------------------------------------------------------------------
# MCP-style client
# ---------------------------------------------------------------------------
class MCPClient:
    """MCP client that discovers tools from a server, uses an LLM to select
    the right tool for a query, calls it, and returns the answer."""

    def __init__(self, server: ToolServer):
        self.server = server
        self.llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
        self._tool_manifests: list[dict[str, Any]] = []

    @mlflow.trace(name="mcp_client.discover_tools")
    def discover_tools(self) -> list[dict[str, Any]]:
        """Step 1: discover available tools from the server."""
        self._tool_manifests = self.server.list_tools()
        return self._tool_manifests

    @mlflow.trace(name="mcp_client.select_tool")
    def select_tool(self, query: str) -> dict[str, Any]:
        """Use the LLM to choose a tool and build arguments."""
        tool_list = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self._tool_manifests
        )
        system_prompt = (
            "You are a tool-selection agent. Given a user query and available tools, "
            'respond with ONLY a JSON object: {"tool": "<name>", "arguments": {<args>}}.\n'
            "Do not include any other text.\n\n"
            f"Available tools:\n{tool_list}"
        )
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ])
        text = response.content.strip()
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"tool": "knowledge_base_search", "arguments": {"query": query}}

    @mlflow.trace(name="mcp_client.execute_query")
    def execute_query(self, query: str) -> dict[str, Any]:
        """Full MCP flow: discover -> select -> call -> return with timing."""
        start = time.time()

        disc_start = time.time()
        self.discover_tools()
        disc_time = time.time() - disc_start

        sel_start = time.time()
        selection = self.select_tool(query)
        sel_time = time.time() - sel_start

        tool_name = selection.get("tool", "unknown")
        arguments = selection.get("arguments", {})

        call_start = time.time()
        result = self.server.call_tool(tool_name, arguments)
        call_time = time.time() - call_start

        return {
            "query": query,
            "tool_selected": tool_name,
            "arguments": arguments,
            "result": result,
            "timing": {
                "discovery_ms": round(disc_time * 1000, 1),
                "selection_ms": round(sel_time * 1000, 1),
                "call_ms": round(call_time * 1000, 1),
                "total_ms": round((time.time() - start) * 1000, 1),
            },
        }
