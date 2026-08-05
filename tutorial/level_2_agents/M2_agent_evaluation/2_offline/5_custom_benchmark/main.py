"""L2-M2.2.5 -- Building a Custom Domain-Specific Agent Benchmark.

Demonstrates how to design, build, and run a domain-specific
agent benchmark. Uses a "customer support agent" scenario as the
example domain, with task taxonomy, evaluation dataset, harness,
and statistical analysis -- all tracked in MLflow.

The agent runs on the Claude Agent SDK with hand-built tracing
(the L2-M1.3 pattern): @mlflow.trace per task, a child span per
tool call.
"""

import json
import statistics
import tempfile
import time
from typing import Any

import anyio
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

MODEL = "claude-sonnet-5"

# -- Step 1: Define the task taxonomy -----------------------------------------

TASK_TAXONOMY = {
    "order_lookup": {
        "description": "Look up order status and details",
        "difficulty": 1,
        "required_tools": ["check_order"],
    },
    "refund_request": {
        "description": "Process a refund based on policy rules",
        "difficulty": 2,
        "required_tools": ["check_order", "check_refund_policy"],
    },
    "complex_complaint": {
        "description": "Handle multi-step complaints requiring escalation logic",
        "difficulty": 3,
        "required_tools": ["check_order", "check_refund_policy", "escalate_ticket"],
    },
}

# -- Step 2: Create the evaluation dataset ------------------------------------

BENCHMARK_DATASET = [
    {
        "task_id": "OL-001",
        "category": "order_lookup",
        "difficulty": 1,
        "input": "What is the status of order #12345?",
        "expected_answer": "shipped",
        "expected_tools": ["check_order"],
    },
    {
        "task_id": "OL-002",
        "category": "order_lookup",
        "difficulty": 1,
        "input": "When will order #67890 arrive?",
        "expected_answer": "2024-03-15",
        "expected_tools": ["check_order"],
    },
    {
        "task_id": "RR-001",
        "category": "refund_request",
        "difficulty": 2,
        "input": "I want a refund for order #12345. It arrived damaged.",
        "expected_answer": "approved",
        "expected_tools": ["check_order", "check_refund_policy"],
    },
    {
        "task_id": "RR-002",
        "category": "refund_request",
        "difficulty": 2,
        "input": "Can I return order #67890? I changed my mind.",
        "expected_answer": "within policy",
        "expected_tools": ["check_order", "check_refund_policy"],
    },
    {
        "task_id": "CC-001",
        "category": "complex_complaint",
        "difficulty": 3,
        "input": (
            "Order #12345 arrived damaged AND late. I've contacted support "
            "three times with no resolution. I want a full refund and compensation."
        ),
        "expected_answer": "escalated",
        "expected_tools": ["check_order", "check_refund_policy", "escalate_ticket"],
    },
    {
        "task_id": "CC-002",
        "category": "complex_complaint",
        "difficulty": 3,
        "input": (
            "I received the wrong item for order #67890. The replacement was also wrong. Please fix this immediately."
        ),
        "expected_answer": "escalated",
        "expected_tools": ["check_order", "escalate_ticket"],
    },
]

# -- Step 3: Build domain-specific tools (in-process MCP) ----------------------

ORDERS_DB = {
    "#12345": {"status": "shipped", "delivery": "2024-03-10", "total": 89.99, "item": "Headphones"},
    "#67890": {
        "status": "processing",
        "delivery": "2024-03-15",
        "total": 149.99,
        "item": "Keyboard",
    },
}


@tool("check_order", "Look up an order by ID and return its status, delivery date, and details", {"order_id": str})
async def check_order(args: dict[str, Any]) -> dict[str, Any]:
    order_id = str(args.get("order_id", ""))
    clean_id = order_id if order_id.startswith("#") else f"#{order_id}"
    order = ORDERS_DB.get(clean_id)
    if not order:
        text = f"Order {clean_id} not found."
    else:
        text = json.dumps({"order_id": clean_id, **order})
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "check_refund_policy",
    "Check if a refund is approved based on the reason and order total",
    {"reason": str, "order_total": float},
)
async def check_refund_policy(args: dict[str, Any]) -> dict[str, Any]:
    reason = str(args.get("reason", ""))
    order_total = float(args.get("order_total", 0.0))
    auto_approve = ["damaged", "wrong item", "defective", "not as described"]
    if any(r in reason.lower() for r in auto_approve):
        text = json.dumps({"approved": True, "type": "full_refund", "reason": reason})
    elif order_total < 50:
        text = json.dumps({"approved": True, "type": "courtesy_refund", "reason": reason})
    else:
        text = json.dumps({"approved": False, "type": "review_needed", "reason": reason})
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "escalate_ticket",
    "Escalate a support ticket to a human supervisor",
    {"order_id": str, "summary": str, "priority": str},
)
async def escalate_ticket(args: dict[str, Any]) -> dict[str, Any]:
    order_id = str(args.get("order_id", ""))
    summary = str(args.get("summary", ""))
    priority = str(args.get("priority", "high"))
    text = json.dumps(
        {
            "ticket_id": f"ESC-{order_id.replace('#', '')}",
            "status": "escalated",
            "priority": priority,
            "summary": summary[:200],
        }
    )
    return {"content": [{"type": "text", "text": text}]}


BENCH_SERVER = create_sdk_mcp_server(
    name="support", version="1.0.0", tools=[check_order, check_refund_policy, escalate_ticket]
)

SYSTEM_PROMPT = (
    "You are a customer support agent. Answer the customer's question "
    "using the available tools. Be concise and helpful. "
    "State the final outcome (e.g. shipped, approved, escalated) explicitly in your answer."
)


def build_options(effort: EffortLevel) -> ClaudeAgentOptions:
    """Agent options for one configuration -- see L2-M2.2.3 for why the
    isolation flags (tools=[], strict_mcp_config, setting_sources) matter."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        effort=effort,
        tools=[],
        mcp_servers={"support": BENCH_SERVER},
        allowed_tools=[
            "mcp__support__check_order",
            "mcp__support__check_refund_policy",
            "mcp__support__escalate_ticket",
        ],
        max_turns=8,
        max_budget_usd=0.25,
        permission_mode="bypassPermissions",
        strict_mcp_config=True,
        setting_sources=[],
    )


# -- Step 4: Build the benchmark harness --------------------------------------


@mlflow.trace(name="support_agent.run")
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


def evaluate_response(response: str, task: dict) -> dict:
    """Score a single agent response against the expected answer."""
    answer_match = task["expected_answer"].lower() in response.lower()
    return {"answer_correct": answer_match}


def evaluate_tool_selection(tool_calls: list[dict], task: dict) -> float:
    """Fraction of expected tools the agent actually called."""
    called = {tc["name"].removeprefix("mcp__support__") for tc in tool_calls}
    expected = set(task["expected_tools"])
    if not expected:
        return 1.0
    return len(called & expected) / len(expected)


async def run_task(options: ClaudeAgentOptions, task: dict, config_name: str) -> dict:
    """Run the agent on one benchmark task and log to MLflow."""
    with mlflow.start_run(run_name=f"{config_name}_{task['task_id']}", nested=True):
        mlflow.log_params(
            {
                "task_id": task["task_id"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "config": config_name,
            }
        )
        start = time.perf_counter()
        try:
            agent_result = await run_agent(task["input"], options)
            latency = time.perf_counter() - start
            answer = agent_result["response"]
            scores = evaluate_response(answer, task)
            tool_recall = evaluate_tool_selection(agent_result["tool_calls"], task)
            mlflow.log_metrics(
                {
                    "latency_s": round(latency, 2),
                    "answer_correct": int(scores["answer_correct"]),
                    "tool_recall": round(tool_recall, 2),
                    "num_tool_calls": len(agent_result["tool_calls"]),
                    "num_turns": agent_result["num_turns"],
                    "response_length": len(answer),
                }
            )
            if agent_result["cost_usd"] is not None:
                mlflow.log_metric("cost_usd", agent_result["cost_usd"])
            mlflow.set_tag("status", "success")
            print(
                f"  [{task['task_id']}] D{task['difficulty']} correct={scores['answer_correct']} "
                f"tool_recall={tool_recall:.2f} latency={latency:.1f}s"
            )
            record = {
                **task,
                "config": config_name,
                "latency_s": round(latency, 2),
                "answer_correct": scores["answer_correct"],
                "tool_recall": tool_recall,
                "num_tool_calls": len(agent_result["tool_calls"]),
                "cost_usd": agent_result["cost_usd"] or 0.0,
                "status": "success",
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            mlflow.log_metrics({"latency_s": round(latency, 2), "answer_correct": 0})
            mlflow.set_tag("status", "error")
            print(f"  [{task['task_id']}] ERROR: {exc}")
            record = {
                **task,
                "config": config_name,
                "latency_s": round(latency, 2),
                "answer_correct": False,
                "tool_recall": 0.0,
                "num_tool_calls": 0,
                "cost_usd": 0.0,
                "status": f"error: {exc}",
            }
    return record


async def run_benchmark(config_name: str, effort: EffortLevel) -> list[dict]:
    """Run the full benchmark suite for one configuration."""
    print(f"\n{'=' * 60}")
    print(f"Config: {config_name}  (model={MODEL}, effort={effort})")
    print("=" * 60)

    options = build_options(effort)
    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{config_name}", nested=True):
        mlflow.log_params(
            {
                "model": MODEL,
                "effort": effort,
                "num_tasks": len(BENCHMARK_DATASET),
            }
        )
        for task in BENCHMARK_DATASET:
            results.append(await run_task(options, task, config_name))

        df = pd.DataFrame(results)
        mlflow.log_metrics(
            {
                "overall_accuracy": round(float(df["answer_correct"].mean()), 3),
                "avg_tool_recall": round(float(df["tool_recall"].mean()), 3),
                "avg_latency_s": round(float(df["latency_s"].mean()), 2),
                "total_cost_usd": round(float(df["cost_usd"].sum()), 4),
            }
        )
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            mlflow.log_metric(f"accuracy_{cat}", round(float(cat_df["answer_correct"].mean()), 3))

        csv_path = f"/tmp/benchmark_{config_name}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

    return results


# -- Step 5: Statistical analysis ---------------------------------------------


def analyze_results(all_results: list[dict]) -> None:
    """Print statistical analysis of benchmark results."""
    print("\n" + "=" * 60)
    print("Step 5: Benchmark Analysis")
    print("=" * 60)

    df = pd.DataFrame(all_results)

    print("\n--- Overall Results ---")
    summary = df.groupby("config").agg(
        accuracy=("answer_correct", "mean"),
        tool_recall=("tool_recall", "mean"),
        avg_latency=("latency_s", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    )
    print(summary.to_string())

    print("\n--- Accuracy by Category ---")
    by_cat = df.groupby(["config", "category"])["answer_correct"].mean().unstack()
    print(by_cat.to_string())

    print("\n--- Accuracy by Difficulty ---")
    by_diff = df.groupby(["config", "difficulty"])["answer_correct"].mean().unstack()
    print(by_diff.to_string())

    for config in df["config"].unique():
        latencies = df[df["config"] == config]["latency_s"].tolist()
        if len(latencies) >= 2:
            print(f"\n--- Latency Stats ({config}) ---")
            print(
                f"  mean={statistics.mean(latencies):.2f}s "
                f"stdev={statistics.stdev(latencies):.2f}s "
                f"min={min(latencies):.2f}s max={max(latencies):.2f}s"
            )


# -- Main ---------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("L2-M2.2.5 -- Custom Domain-Specific Agent Benchmark")
    print("=" * 60)

    print("\nStep 1: Task Taxonomy")
    for name, info in TASK_TAXONOMY.items():
        print(f"  {name} (difficulty={info['difficulty']}): {info['description']}")

    print(f"\nStep 2: Dataset — {len(BENCHMARK_DATASET)} tasks")
    for task in BENCHMARK_DATASET:
        print(f"  [{task['task_id']}] {task['category']} D{task['difficulty']}")

    print("\nStep 3-4: Running benchmark ...")
    configs: list[tuple[str, EffortLevel]] = [("low_effort", "low"), ("high_effort", "high")]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="custom_benchmark"):
        mlflow.log_params(
            {
                "domain": "customer_support",
                "model": MODEL,
                "num_tasks": len(BENCHMARK_DATASET),
                "num_categories": len(TASK_TAXONOMY),
            }
        )
        mlflow.set_tag("framework", "claude_agent_sdk")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"taxonomy": TASK_TAXONOMY, "dataset": BENCHMARK_DATASET}, f, indent=2)
            mlflow.log_artifact(f.name, "benchmark_definition")

        for name, effort in configs:
            all_results.extend(await run_benchmark(name, effort))

        combined_csv = "/tmp/custom_benchmark_combined.csv"
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    analyze_results(all_results)
    print("\n" + "=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M2_agent_evaluation/2_offline/5_custom_benchmark")
    anyio.run(main)
