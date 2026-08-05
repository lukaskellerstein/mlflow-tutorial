"""L2-M2.2.4 -- GAIA General Assistant Benchmark.

Runs a Claude Agent SDK agent against GAIA Level 1 tasks (multi-step
reasoning + tool use) and tracks per-task results in MLflow.
Two agent configurations (effort low vs high) are compared.

Tracing is hand-built (the L2-M1.3 pattern): @mlflow.trace opens the
root span per task and every ToolUseBlock gets a child span.
"""

import json
import time
from typing import Any

import anyio
import datasets
import mlflow
import pandas as pd
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    EffortLevel,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

SAMPLE_SIZE = 5
MODEL = "claude-sonnet-5"

# -- Tools for the assistant agent (in-process MCP) ----------------------------


@tool("calculator", "Evaluate a mathematical expression and return the result", {"expression": str})
async def calculator(args: dict[str, Any]) -> dict[str, Any]:
    """Safe arithmetic evaluator for GAIA's numeric sub-steps."""
    expression = str(args.get("expression", ""))
    allowed = set("0123456789+-*/.() ")
    try:
        if all(c in allowed for c in expression):
            text = str(eval(expression))
        else:
            text = f"Cannot evaluate: {expression}"
    except Exception as e:
        text = f"Error: {e}"
    return {"content": [{"type": "text", "text": text}]}


@tool("knowledge_lookup", "Look up factual information to answer a knowledge question", {"query": str})
async def knowledge_lookup(args: dict[str, Any]) -> dict[str, Any]:
    """Mock knowledge base -- the lesson is the harness, not retrieval."""
    query = str(args.get("query", ""))
    text = (
        f"Knowledge lookup for: {query[:200]}. "
        "Based on available information, the relevant facts are: "
        "This requires multi-step reasoning to combine known facts "
        "and arrive at the final answer."
    )
    return {"content": [{"type": "text", "text": text}]}


@tool("text_analyzer", "Analyze text content -- extract key facts, count items, or summarize", {"text": str})
async def text_analyzer(args: dict[str, Any]) -> dict[str, Any]:
    """Word-level text statistics for GAIA's document-shaped tasks."""
    text_in = str(args.get("text", ""))
    words = text_in.split()
    text = (
        f"Text analysis: {len(words)} words. "
        f"Key content: {' '.join(words[:30])}... "
        "Analysis complete -- use the extracted facts to form your answer."
    )
    return {"content": [{"type": "text", "text": text}]}


BENCH_SERVER = create_sdk_mcp_server(name="bench", version="1.0.0", tools=[calculator, knowledge_lookup, text_analyzer])

SYSTEM_PROMPT = (
    "You are a general-purpose AI assistant solving benchmark tasks. "
    "Use the available tools to reason step-by-step. "
    "Give a short, precise final answer on the last line of your response."
)


def build_options(effort: EffortLevel) -> ClaudeAgentOptions:
    """Agent options for one configuration -- see L2-M2.2.3 for why the
    isolation flags (tools=[], strict_mcp_config, setting_sources) matter."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        effort=effort,
        tools=[],
        mcp_servers={"bench": BENCH_SERVER},
        allowed_tools=[
            "mcp__bench__calculator",
            "mcp__bench__knowledge_lookup",
            "mcp__bench__text_analyzer",
        ],
        max_turns=8,
        max_budget_usd=0.25,
        permission_mode="bypassPermissions",
        strict_mcp_config=True,
        setting_sources=[],
    )


# -- Agent execution -----------------------------------------------------------


@mlflow.trace(name="gaia_agent.run")
async def run_agent(prompt: str, options: ClaudeAgentOptions) -> dict:
    """Execute one query and capture response, tool calls, and SDK metrics."""
    response_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    num_turns = 0
    cost_usd: float | None = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes({"tool_name": block.name, "tool_use_id": block.id})
                        tool_calls.append({"name": block.name, "input": block.input})
            elif isinstance(message, ResultMessage):
                num_turns = message.num_turns
                cost_usd = message.total_cost_usd

    return {
        "response": "".join(response_parts),
        "tool_calls": tool_calls,
        "num_turns": num_turns,
        "cost_usd": cost_usd,
    }


async def run_instance(
    options: ClaudeAgentOptions, question: str, expected: str, level: int, task_id: str, config_name: str
) -> dict:
    """Run the agent on one GAIA instance and log to MLflow."""
    with mlflow.start_run(run_name=f"{config_name}_{task_id[:8]}", nested=True):
        mlflow.log_params(
            {
                "task_id": task_id,
                "level": level,
                "config": config_name,
            }
        )
        start = time.perf_counter()
        try:
            agent_result = await run_agent(f"## Question\n{question}", options)
            latency = time.perf_counter() - start
            answer = agent_result["response"]
            answer_short = answer.strip().split("\n")[-1].strip()
            exact_match = expected.lower().strip() in answer_short.lower()
            mlflow.log_metrics(
                {
                    "latency_s": round(latency, 2),
                    "exact_match": int(exact_match),
                    "response_length": len(answer),
                    "num_turns": agent_result["num_turns"],
                    "num_tool_calls": len(agent_result["tool_calls"]),
                }
            )
            if agent_result["cost_usd"] is not None:
                mlflow.log_metric("cost_usd", agent_result["cost_usd"])
            mlflow.set_tag("status", "success")
            mlflow.set_tag("tools_used", json.dumps([tc["name"] for tc in agent_result["tool_calls"]]))
            print(f"  [{task_id[:8]}] L{level} match={exact_match} latency={latency:.1f}s")
            record = {
                "task_id": task_id,
                "level": level,
                "config": config_name,
                "latency_s": round(latency, 2),
                "exact_match": exact_match,
                "response_length": len(answer),
                "cost_usd": agent_result["cost_usd"] or 0.0,
                "status": "success",
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            mlflow.log_metrics({"latency_s": round(latency, 2), "exact_match": 0})
            mlflow.set_tag("status", "error")
            mlflow.set_tag("error", str(exc)[:200])
            print(f"  [{task_id[:8]}] ERROR: {exc}")
            record = {
                "task_id": task_id,
                "level": level,
                "config": config_name,
                "latency_s": round(latency, 2),
                "exact_match": False,
                "response_length": 0,
                "cost_usd": 0.0,
                "status": f"error: {exc}",
            }
    return record


async def run_config(name: str, effort: EffortLevel, instances: list[dict]) -> list[dict]:
    """Run the agent across all GAIA instances for a given config."""
    print(f"\n{'=' * 60}")
    print(f"Config: {name}  (model={MODEL}, effort={effort})")
    print("=" * 60)

    options = build_options(effort)
    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{name}", nested=True):
        mlflow.log_params(
            {
                "model": MODEL,
                "effort": effort,
                "sample_size": len(instances),
            }
        )
        for inst in instances:
            results.append(
                await run_instance(
                    options,
                    question=inst["Question"],
                    expected=inst.get("Final answer", ""),
                    level=inst.get("Level", 1),
                    task_id=inst.get("task_id", "unknown"),
                    config_name=name,
                )
            )

        df = pd.DataFrame(results)
        mlflow.log_metrics(
            {
                "accuracy": round(float(df["exact_match"].mean()), 3),
                "avg_latency_s": round(float(df["latency_s"].mean()), 2),
                "success_rate": round((df["status"] == "success").mean(), 3),
                "total_cost_usd": round(float(df["cost_usd"].sum()), 4),
            }
        )
        csv_path = f"/tmp/gaia_{name}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print a comparison table across configurations."""
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    df = pd.DataFrame(all_results)
    summary = df.groupby("config").agg(
        accuracy=("exact_match", "mean"),
        avg_latency=("latency_s", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        success_rate=("status", lambda s: (s == "success").mean()),
    )
    print(summary.to_string())

    if df["level"].nunique() > 1:
        print("\nAccuracy by Level:")
        by_level = df.groupby(["config", "level"])["exact_match"].mean().unstack()
        print(by_level.to_string())
    print()


async def main() -> None:
    print("=" * 60)
    print("L2-M2.2.4 -- GAIA General Assistant Benchmark")
    print("=" * 60)

    print("\nStep 1: Loading GAIA dataset ...")
    ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    sample = [ds[i] for i in range(min(SAMPLE_SIZE, len(ds)))]
    print(f"  Loaded {len(ds)} Level-1 instances, using {len(sample)} for demo")

    print("\nStep 2: Running benchmark ...")
    configs: list[tuple[str, EffortLevel]] = [("low_effort", "low"), ("high_effort", "high")]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="gaia_benchmark"):
        mlflow.log_params({"dataset": "GAIA", "level": 1, "sample_size": len(sample), "model": MODEL})
        mlflow.set_tag("task", "gaia_benchmark")
        mlflow.set_tag("framework", "claude_agent_sdk")
        for name, effort in configs:
            all_results.extend(await run_config(name, effort, sample))
        combined_csv = "/tmp/gaia_combined.csv"
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    print_summary(all_results)
    print("=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M2_agent_evaluation/2_offline/4_gaia")
    anyio.run(main)
