"""
L2-M2.1 — Claude Agent SDK + MLflow Integration

Demonstrates building a custom MLflow tracing integration for the Claude
Agent SDK. The SDK has no native MLflow support, so we wrap agent execution
with @mlflow.trace and manual spans to capture the full lifecycle.

An external FastMCP server (server.py) provides the tools. The SDK
connects to it via STDIO transport.

Parts:
  1. MCP server config and SDK setup
  2. Traced agent execution with @mlflow.trace
  3. Comparison runs with nested MLflow runs
  4. Trace analysis — query spans and hierarchy
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, cast

import anyio
import mlflow
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from mlflow.entities import Trace

# ── Part 1: MCP server config and SDK setup ─────────────────────

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

SYSTEM_PROMPT = "You are a helpful assistant. Use available tools when they help answer the question. Be concise."


def build_options(max_turns: int = 3) -> ClaudeAgentOptions:
    """Create ClaudeAgentOptions that connect to our FastMCP server."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        tools=[],
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
        max_turns=max_turns,
        permission_mode="dontAsk",
    )


# ── Part 2: Traced agent execution ──────────────────────────────


@dataclass
class AgentResult:
    """Collected data from an agent execution."""

    query: str
    response: str
    thinking: str
    tool_calls: list[dict[str, Any]]
    duration_ms: int
    num_turns: int
    cost_usd: float | None
    model: str


@mlflow.trace(name="claude_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> AgentResult:
    """Execute a Claude agent query with full MLflow tracing.

    Creates a root span via @mlflow.trace and child spans for each
    tool call encountered in the message stream.
    """
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    model_name = ""
    duration_ms = 0
    num_turns = 0
    cost_usd = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                model_name = message.model
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.thinking)
                    elif isinstance(block, TextBlock):
                        response_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes(
                                {
                                    "tool_name": block.name,
                                    "tool_use_id": block.id,
                                }
                            )
                        tool_calls.append(
                            {
                                "name": block.name,
                                "input": block.input,
                            }
                        )
            elif isinstance(message, ResultMessage):
                duration_ms = message.duration_ms
                num_turns = message.num_turns
                cost_usd = message.total_cost_usd

    return AgentResult(
        query=prompt,
        response="".join(response_parts),
        thinking="".join(thinking_parts),
        tool_calls=tool_calls,
        duration_ms=duration_ms,
        num_turns=num_turns,
        cost_usd=cost_usd,
        model=model_name,
    )


# ── Part 3: Comparison runs with nested MLflow runs ─────────────

EXAMPLE_QUERIES = [
    "What is 42 * 17 + 3?",
    "Tell me about MLflow.",
    "Explain why testing AI agents is important.",
]


async def run_examples(options: ClaudeAgentOptions) -> list[AgentResult]:
    """Run the traced agent on example queries with nested MLflow runs."""
    results: list[AgentResult] = []

    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n  Query {i}: {q}")
        with mlflow.start_run(run_name=f"query_{i}", nested=True):
            result = await run_agent(q, options)
            results.append(result)

            mlflow.log_params({"query": q[:250], "model": result.model})
            mlflow.log_metrics(
                {
                    "duration_ms": result.duration_ms,
                    "num_turns": result.num_turns,
                    "tool_calls": len(result.tool_calls),
                    "response_length": len(result.response),
                }
            )
            if result.cost_usd is not None:
                mlflow.log_metric("cost_usd", result.cost_usd)
            mlflow.set_tags(
                {
                    "has_tool_call": str(len(result.tool_calls) > 0),
                    "tools_used": json.dumps([tc["name"] for tc in result.tool_calls]),
                }
            )

            print(f"    Response: {result.response[:100]}...")
            if result.tool_calls:
                for tc in result.tool_calls:
                    print(f"    Tool: {tc['name']}({json.dumps(tc['input'])})")
            print(f"    Duration: {result.duration_ms}ms, Turns: {result.num_turns}")
            if result.cost_usd:
                print(f"    Cost: ${result.cost_usd:.4f}")

    return results


# ── Part 4: Trace analysis ──────────────────────────────────────


def analyze_traces() -> None:
    """Query traces from MLflow and display the span hierarchy."""
    experiment = mlflow.get_experiment_by_name("L2/M2_custom_integrations/1_claude_agent_sdk")
    if experiment is None:
        print("  Experiment not found.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(
            locations=[experiment.experiment_id],
            return_type="list",
            flush=True,
        ),
    )
    print(f"  Found {len(traces)} traces")

    for trace in traces[:3]:
        req_id = trace.info.request_id
        status = trace.info.status
        dur_ms = trace.info.execution_time_ms
        print(f"\n  Trace {req_id[:12]}... | {status} | {dur_ms}ms")

        for span in trace.data.spans:
            span_dur = ((span.end_time_ns or 0) - (span.start_time_ns or 0)) / 1e6
            print(f"    - {span.name} ({span_dur:.0f}ms)")

    if traces:
        durations = [t.info.execution_time_ms for t in traces if t.info.execution_time_ms is not None]
        if durations:
            print(f"\n  Average trace duration: {sum(durations) / len(durations):.0f}ms")
            print(f"  Total traces: {len(traces)}")


# ── Main ─────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("L2-M2.1 — Claude Agent SDK + MLflow Integration")
    print("=" * 60)

    # Part 1: MCP server config
    print("\n" + "=" * 60)
    print("Part 1: MCP server + SDK config")
    print("=" * 60)
    print(f"  MCP server: {SERVER_PATH}")
    print("  Tools: calculator, knowledge_lookup")
    print("  Transport: STDIO (external FastMCP process)")

    # Part 2: Single traced query
    print("\n" + "=" * 60)
    print("Part 2: Traced agent execution")
    print("=" * 60)
    options = build_options()
    result = await run_agent("What is 123 * 456?", options)
    print(f"  Response: {result.response[:150]}")
    print(f"  Tools used: {[tc['name'] for tc in result.tool_calls]}")
    print(f"  Duration: {result.duration_ms}ms, Model: {result.model}")
    if result.cost_usd:
        print(f"  Cost: ${result.cost_usd:.4f}")

    # Part 3: Comparison runs
    print("\n" + "=" * 60)
    print("Part 3: Running example queries")
    print("=" * 60)
    with mlflow.start_run(run_name="claude_sdk_integration") as parent_run:
        mlflow.log_params(
            {
                "agent_type": "claude_agent_sdk",
                "num_queries": len(EXAMPLE_QUERIES),
                "tools": json.dumps(["calculator", "knowledge_lookup"]),
            }
        )
        mlflow.set_tags(
            {
                "framework": "claude_agent_sdk",
                "integration_type": "custom_tracing",
            }
        )

        results = await run_examples(options)

        total_dur = sum(r.duration_ms for r in results)
        total_tools = sum(len(r.tool_calls) for r in results)
        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        mlflow.log_metrics(
            {
                "total_duration_ms": total_dur,
                "avg_duration_ms": total_dur // max(len(results), 1),
                "total_tool_calls": total_tools,
            }
        )
        if costs:
            mlflow.log_metric("total_cost_usd", sum(costs))

        summary = {
            "results": [
                {
                    "query": r.query,
                    "response": r.response[:200],
                    "tools": [tc["name"] for tc in r.tool_calls],
                    "duration_ms": r.duration_ms,
                    "cost_usd": r.cost_usd,
                }
                for r in results
            ],
        }
        path = "/tmp/claude_agent_results.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(path)

        parent_id = parent_run.info.run_id

    # Part 4: Trace analysis
    print("\n" + "=" * 60)
    print("Part 4: Trace analysis")
    print("=" * 60)
    analyze_traces()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  Integration pattern for ANY third-party agent SDK:")
    print("  1. External MCP server (FastMCP) for custom tools")
    print("  2. ClaudeSDKClient with STDIO MCP server config")
    print("  3. @mlflow.trace on the entry-point function")
    print("  4. mlflow.start_span() for tool call events")
    print("  5. Log metrics/artifacts to MLflow runs")
    print(f"\n  Run ID: {parent_id}")
    print("  MLflow UI: http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M2_custom_integrations/1_claude_agent_sdk")
    anyio.run(main)
