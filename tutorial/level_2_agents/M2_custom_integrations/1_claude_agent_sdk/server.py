"""
MCP Server — Tutorial Tools

Provides calculator and knowledge_lookup tools via FastMCP (STDIO transport).
The Claude Agent SDK connects to this server as an external MCP process.

Run standalone:  uv run python server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tutorial_tools")


@mcp.tool()
def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression and return the numeric result.

    Args:
        expression: A math expression using digits and +-*/.() operators.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return {"error": f"Invalid expression: {expression}"}
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def knowledge_lookup(topic: str) -> dict:
    """Look up a fact from the knowledge base by topic keyword.

    Args:
        topic: A topic keyword (e.g. 'python', 'mlflow', 'claude', 'langchain').
    """
    facts = {
        "python": "Python was created by Guido van Rossum in 1991. It emphasizes readability and supports multiple programming paradigms.",
        "mlflow": "MLflow is an open-source MLOps platform by Databricks for managing the ML lifecycle: tracking, models, registry, evaluation, and deployment.",
        "claude": "Claude is an AI assistant by Anthropic. The Claude Agent SDK enables building autonomous agents with tool use, hooks, and multi-turn conversations.",
        "langchain": "LangChain is a framework for building LLM-powered applications with composable components for chains, agents, and retrieval.",
    }
    key = topic.lower().strip()
    for k, v in facts.items():
        if k in key:
            return {"topic": topic, "fact": v}
    return {"topic": topic, "fact": f"No entry found for: {topic}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
