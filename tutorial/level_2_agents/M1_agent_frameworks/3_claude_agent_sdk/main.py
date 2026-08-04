"""
L2-M1.3 — Claude Agent SDK with MLflow

The Claude Agent SDK has no `mlflow.*.autolog()` — unlike LangChain and
DeepAgents, nothing instruments it for you. This lesson builds that integration
by hand, which is the pattern for ANY agent framework MLflow does not ship
support for: wrap the entry point in `@mlflow.trace`, and open a manual span for
each interesting event in the message stream.

Both ways of giving the agent tools are shown, because they trace differently:

  in-process MCP  — @tool + create_sdk_mcp_server, runs inside this process
  external MCP    — server.py as a separate FastMCP process over STDIO

Parts:
  1. Tools — in-process and external MCP servers side by side
  2. Traced agent execution — @mlflow.trace plus manual tool spans
  3. Comparison runs with nested MLflow runs
  4. Trace analysis — walk the span hierarchy that was captured
"""

import json
import os
import sys
from dataclasses import dataclass, field
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
    create_sdk_mcp_server,
    tool,
)
from mlflow.entities import Trace

EXPERIMENT = "L2/M1_agent_frameworks/3_claude_agent_sdk"

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

# Seconds to let the external MCP subprocess finish connecting before querying.
# See the comment in run_agent — without it the external tools are invisible
# roughly half the time, silently.
MCP_SETTLE_SECONDS = 3.0

SYSTEM_PROMPT = "You are a helpful assistant. Use available tools when they help answer the question. Be concise."

# The SDK drives the Claude Code CLI, so it authenticates the same way your CLI
# does — no api_key here, and no LiteLLM gateway either. The gateway speaks the
# OpenAI API; this SDK speaks Anthropic's Messages API through the CLI. That is
# exactly why this lesson exists: not every framework fits one endpoint.


# ── Part 1: tools — two MCP transports ────────────────────────────


@tool("word_stats", "Count words and characters in a piece of text", {"text": str})
async def word_stats(args: dict[str, Any]) -> dict[str, Any]:
    """An in-process tool: no subprocess, no serialization boundary."""
    text = str(args.get("text", ""))
    stats = {"words": len(text.split()), "characters": len(text)}
    return {"content": [{"type": "text", "text": json.dumps(stats)}]}


# create_sdk_mcp_server builds an MCP server that lives in THIS process. The
# external alternative (server.py) is a real subprocess — same protocol, but the
# tool runs somewhere this program cannot see, which is why the manual span
# below is the only record of it.
INPROC_SERVER = create_sdk_mcp_server(name="inproc", version="1.0.0", tools=[word_stats])


def build_options(max_turns: int = 3) -> ClaudeAgentOptions:
    """Wire up both MCP servers and allow their tools."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        mcp_servers={
            # in-process: the @tool functions above
            "inproc": INPROC_SERVER,
            # external: a separate FastMCP process over STDIO
            "tools": {"type": "stdio", "command": sys.executable, "args": [SERVER_PATH]},
        },
        allowed_tools=[
            "mcp__inproc__word_stats",
            "mcp__tools__calculator",
            "mcp__tools__knowledge_lookup",
        ],
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        # REQUIRED, and the failure without it is silent. The SDK drives the
        # Claude Code CLI, which by default MERGES the servers below with any
        # .mcp.json in the working directory. Those project servers arrive
        # unapproved, and the external server here then never becomes available:
        # the model simply reports "there is no calculator tool" and answers
        # from its own knowledge. Nothing errors and the run still exits 0.
        # strict_mcp_config makes the CLI use ONLY the servers passed here.
        strict_mcp_config=True,
    )


# ── Part 2: traced agent execution ────────────────────────────────


@dataclass
class AgentResult:
    """Collected data from one agent execution."""

    query: str
    response: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    num_turns: int = 0
    cost_usd: float | None = None
    model: str = ""


@mlflow.trace(name="claude_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> AgentResult:
    """Execute one query with full MLflow tracing.

    @mlflow.trace opens the root span. Each ToolUseBlock in the stream gets a
    child span — that is the hand-built equivalent of what autolog would do.
    """
    result = AgentResult(query=prompt)
    response_parts: list[str] = []
    thinking_parts: list[str] = []

    async with ClaudeSDKClient(options=options) as client:
        # The external MCP server is a subprocess and connects LAZILY — the
        # context manager returns before its tools are registered. Query too soon
        # and the model sees only the in-process tools and answers "there is no
        # calculator tool" without erroring. The SDK exposes no readiness signal
        # for this (get_mcp_status returns an empty list under strict config), so
        # a settle delay is the available fix. In-process tools need none.
        await anyio.sleep(MCP_SETTLE_SECONDS)

        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                result.model = message.model
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.thinking)
                    elif isinstance(block, TextBlock):
                        response_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        # The SDK reports the call, not the result, so this span
                        # records the decision to use a tool and its arguments.
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes(
                                {
                                    "tool_name": block.name,
                                    "tool_use_id": block.id,
                                    "transport": "in-process" if "__inproc__" in block.name else "external-stdio",
                                }
                            )
                        result.tool_calls.append({"name": block.name, "input": block.input})
            elif isinstance(message, ResultMessage):
                result.duration_ms = message.duration_ms
                result.num_turns = message.num_turns
                result.cost_usd = message.total_cost_usd

    result.response = "".join(response_parts)
    result.thinking = "".join(thinking_parts)
    return result


# ── Part 3: comparison runs ───────────────────────────────────────

# Chosen to exercise each tool path: external calculator, external knowledge
# lookup, in-process word_stats, and one question needing no tool at all.
#
# The first three name their tool explicitly. Without that, Claude answers
# arithmetic and general questions from its own knowledge and the external MCP
# server is never touched — which would leave the transport comparison with
# nothing to compare.
EXAMPLE_QUERIES = [
    "Use the calculator tool to compute 4217 * 1793 + 88.",
    "Use the knowledge_lookup tool to look up 'mlflow' and report exactly what it returns.",
    "Use the word_stats tool on this sentence: 'Agent evaluation needs real traces'.",
    "Explain why testing AI agents is important.",
]


async def run_examples(options: ClaudeAgentOptions) -> list[AgentResult]:
    """Run every query in a nested MLflow run of its own."""
    results: list[AgentResult] = []

    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n  Query {i}: {query}")
        with mlflow.start_run(run_name=f"query_{i}", nested=True):
            result = await run_agent(query, options)
            results.append(result)

            mlflow.log_params({"query": query[:250], "model": result.model})
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
                    "has_tool_call": str(bool(result.tool_calls)),
                    "tools_used": json.dumps([tc["name"] for tc in result.tool_calls]),
                }
            )

            print(f"    Response: {result.response[:100]}")
            for tc in result.tool_calls:
                print(f"    Tool: {tc['name']}({json.dumps(tc['input'])[:70]})")
            print(f"    Duration: {result.duration_ms}ms, Turns: {result.num_turns}")
            if result.cost_usd:
                print(f"    Cost: ${result.cost_usd:.4f}")

    return results


# ── Part 4: trace analysis ────────────────────────────────────────


def analyze_traces() -> None:
    """Query the traces this lesson produced and show the span hierarchy."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  Experiment not found.")
        return

    traces = cast(
        list[Trace],
        mlflow.search_traces(locations=[experiment.experiment_id], return_type="list", flush=True),
    )
    print(f"  Found {len(traces)} traces")

    for trace in traces[:3]:
        print(f"\n  Trace {trace.info.trace_id[:16]}... | {trace.info.status} | {trace.info.execution_time_ms}ms")
        for span in trace.data.spans:
            duration_ms = ((span.end_time_ns or 0) - (span.start_time_ns or 0)) / 1e6
            print(f"    - {span.name} ({duration_ms:.0f}ms)")

    durations = [t.info.execution_time_ms for t in traces if t.info.execution_time_ms is not None]
    if durations:
        print(f"\n  Average trace duration: {sum(durations) / len(durations):.0f}ms across {len(traces)} traces")


# ── Main ──────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("L2-M1.3 — Claude Agent SDK with MLflow")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Part 1: Tools — two MCP transports")
    print("=" * 60)
    print("  in-process : mcp__inproc__word_stats (@tool, this process)")
    print(f"  external   : mcp__tools__calculator, mcp__tools__knowledge_lookup ({SERVER_PATH})")

    print("\n" + "=" * 60)
    print("Part 2: Traced agent execution")
    print("=" * 60)
    options = build_options()
    result = await run_agent("Use the calculator tool to compute 123 * 456.", options)
    print(f"  Response  : {result.response[:150]}")
    print(f"  Tools used: {[tc['name'] for tc in result.tool_calls]}")
    print(f"  Duration  : {result.duration_ms}ms, Model: {result.model}")
    if result.cost_usd:
        print(f"  Cost      : ${result.cost_usd:.4f}")

    print("\n" + "=" * 60)
    print("Part 3: Running example queries")
    print("=" * 60)
    with mlflow.start_run(run_name="claude_sdk_integration") as parent_run:
        mlflow.log_params(
            {
                "agent_type": "claude_agent_sdk",
                "num_queries": len(EXAMPLE_QUERIES),
                "inproc_tools": json.dumps(["word_stats"]),
                "external_tools": json.dumps(["calculator", "knowledge_lookup"]),
            }
        )
        mlflow.set_tags({"framework": "claude_agent_sdk", "integration_type": "custom_tracing"})

        results = await run_examples(options)

        total_duration = sum(r.duration_ms for r in results)
        costs = [r.cost_usd for r in results if r.cost_usd is not None]
        mlflow.log_metrics(
            {
                "total_duration_ms": total_duration,
                "avg_duration_ms": total_duration // max(len(results), 1),
                "total_tool_calls": sum(len(r.tool_calls) for r in results),
            }
        )
        if costs:
            mlflow.log_metric("total_cost_usd", sum(costs))

        mlflow.log_dict(
            {
                "results": [
                    {
                        "query": r.query,
                        "response": r.response[:200],
                        "tools": [tc["name"] for tc in r.tool_calls],
                        "duration_ms": r.duration_ms,
                        "cost_usd": r.cost_usd,
                    }
                    for r in results
                ]
            },
            "results.json",
        )
        parent_id = parent_run.info.run_id

    print("\n" + "=" * 60)
    print("Part 4: Trace analysis")
    print("=" * 60)
    analyze_traces()

    print("\n" + "=" * 60)
    print("Summary — the integration pattern for ANY unsupported agent SDK")
    print("=" * 60)
    print("  1. @mlflow.trace on the entry-point function -> the root span")
    print("  2. mlflow.start_span() per interesting event  -> the child spans")
    print("  3. Nested runs per query                      -> comparable metrics")
    print("  4. Log the SDK's own numbers (turns, cost)    -> what it alone knows")
    print(f"\n  Parent run: {parent_id}")
    print(f"  MLflow UI : http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT)
    anyio.run(main)
