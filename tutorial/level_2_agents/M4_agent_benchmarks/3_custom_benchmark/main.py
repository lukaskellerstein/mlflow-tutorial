"""L2-M4.3 -- Building a Custom Domain-Specific Agent Benchmark.

Demonstrates how to design, build, and run a domain-specific
agent benchmark. Uses a "customer support agent" scenario as the
example domain, with task taxonomy, evaluation dataset, harness,
and statistical analysis — all tracked in MLflow.
"""

import json
import statistics
import tempfile
import time

import mlflow
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

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
            "I received the wrong item for order #67890. The replacement "
            "was also wrong. Please fix this immediately."
        ),
        "expected_answer": "escalated",
        "expected_tools": ["check_order", "escalate_ticket"],
    },
]

# -- Step 3: Build domain-specific tools -------------------------------------

ORDERS_DB = {
    "#12345": {"status": "shipped", "delivery": "2024-03-10", "total": 89.99, "item": "Headphones"},
    "#67890": {"status": "processing", "delivery": "2024-03-15", "total": 149.99, "item": "Keyboard"},
}


@tool
def check_order(order_id: str) -> str:
    """Look up an order by ID and return its status, delivery date, and details."""
    clean_id = order_id if order_id.startswith("#") else f"#{order_id}"
    order = ORDERS_DB.get(clean_id)
    if not order:
        return f"Order {clean_id} not found."
    return json.dumps({"order_id": clean_id, **order})


@tool
def check_refund_policy(reason: str, order_total: float) -> str:
    """Check if a refund is approved based on the reason and order total."""
    auto_approve = ["damaged", "wrong item", "defective", "not as described"]
    if any(r in reason.lower() for r in auto_approve):
        return json.dumps({"approved": True, "type": "full_refund", "reason": reason})
    if order_total < 50:
        return json.dumps({"approved": True, "type": "courtesy_refund", "reason": reason})
    return json.dumps({"approved": False, "type": "review_needed", "reason": reason})


@tool
def escalate_ticket(order_id: str, summary: str, priority: str = "high") -> str:
    """Escalate a support ticket to a human supervisor."""
    return json.dumps({
        "ticket_id": f"ESC-{order_id.replace('#', '')}",
        "status": "escalated",
        "priority": priority,
        "summary": summary[:200],
    })


TOOLS = [check_order, check_refund_policy, escalate_ticket]

SYSTEM_PROMPT = (
    "You are a customer support agent. Answer the customer's question "
    "using the available tools. Be concise and helpful."
)

# -- Step 4: Build the benchmark harness --------------------------------------


def evaluate_response(response: str, task: dict) -> dict:
    """Score a single agent response against the expected answer."""
    answer_match = task["expected_answer"].lower() in response.lower()
    return {"answer_correct": answer_match}


def run_task(agent, task: dict, config_name: str) -> dict:
    """Run the agent on one benchmark task and log to MLflow."""
    with mlflow.start_run(run_name=f"{config_name}_{task['task_id']}", nested=True):
        mlflow.log_params({
            "task_id": task["task_id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "config": config_name,
        })
        start = time.perf_counter()
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": task["input"]}]})
            latency = time.perf_counter() - start
            answer = result["messages"][-1].content
            scores = evaluate_response(answer, task)
            tool_calls = [
                m.name for m in result["messages"]
                if hasattr(m, "name") and m.name
            ]
            mlflow.log_metrics({
                "latency_s": round(latency, 2),
                "answer_correct": int(scores["answer_correct"]),
                "num_tool_calls": len(tool_calls),
                "response_length": len(answer),
            })
            mlflow.set_tag("status", "success")
            print(f"  [{task['task_id']}] D{task['difficulty']} "
                  f"correct={scores['answer_correct']} latency={latency:.1f}s")
            record = {
                **task, "config": config_name, "latency_s": round(latency, 2),
                "answer_correct": scores["answer_correct"],
                "num_tool_calls": len(tool_calls), "status": "success",
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            mlflow.log_metrics({"latency_s": round(latency, 2), "answer_correct": 0})
            mlflow.set_tag("status", "error")
            print(f"  [{task['task_id']}] ERROR: {exc}")
            record = {
                **task, "config": config_name, "latency_s": round(latency, 2),
                "answer_correct": False, "num_tool_calls": 0,
                "status": f"error: {exc}",
            }
    return record


def run_benchmark(config_name: str, temperature: float) -> list[dict]:
    """Run the full benchmark suite for one configuration."""
    print(f"\n{'=' * 60}")
    print(f"Config: {config_name}  (temperature={temperature})")
    print("=" * 60)

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1", api_key="lm-studio",
        model="google/gemma-4-26b-a4b", temperature=temperature, max_tokens=1024,  # pyright: ignore[reportCallIssue]  # pydantic field alias; valid at runtime
    )
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{config_name}", nested=True):
        mlflow.log_params({
            "temperature": temperature, "model": "google/gemma-4-26b-a4b",
            "num_tasks": len(BENCHMARK_DATASET),
        })
        for task in BENCHMARK_DATASET:
            results.append(run_task(agent, task, config_name))

        df = pd.DataFrame(results)
        mlflow.log_metrics({
            "overall_accuracy": round(float(df["answer_correct"].mean()), 3),
            "avg_latency_s": round(float(df["latency_s"].mean()), 2),
        })
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
        avg_latency=("latency_s", "mean"),
        avg_tools=("num_tool_calls", "mean"),
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
            print(f"  mean={statistics.mean(latencies):.2f}s "
                  f"stdev={statistics.stdev(latencies):.2f}s "
                  f"min={min(latencies):.2f}s max={max(latencies):.2f}s")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("L2-M4.3 -- Custom Domain-Specific Agent Benchmark")
    print("=" * 60)

    mlflow.langchain.autolog()

    print("\nStep 1: Task Taxonomy")
    for name, info in TASK_TAXONOMY.items():
        print(f"  {name} (difficulty={info['difficulty']}): {info['description']}")

    print(f"\nStep 2: Dataset — {len(BENCHMARK_DATASET)} tasks")
    for task in BENCHMARK_DATASET:
        print(f"  [{task['task_id']}] {task['category']} D{task['difficulty']}")

    print("\nStep 3-4: Running benchmark ...")
    configs = [("conservative", 0.2), ("balanced", 0.5)]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="custom_benchmark"):
        mlflow.log_params({
            "domain": "customer_support",
            "num_tasks": len(BENCHMARK_DATASET),
            "num_categories": len(TASK_TAXONOMY),
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"taxonomy": TASK_TAXONOMY, "dataset": BENCHMARK_DATASET}, f, indent=2)
            mlflow.log_artifact(f.name, "benchmark_definition")

        for name, temp in configs:
            all_results.extend(run_benchmark(name, temp))

        combined_csv = "/tmp/custom_benchmark_combined.csv"
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    analyze_results(all_results)
    print("\n" + "=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M4_agent_benchmarks/3_custom_benchmark")
    main()
