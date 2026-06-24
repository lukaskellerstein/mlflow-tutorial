"""
L3-1.5 — End-to-End Agent Evaluation Pipeline

Complete automated evaluation pipeline for AI agents:
  Part 1: Build and run the full pipeline (dataset -> agent -> score -> gates -> report)
  Part 2: Regression detection against a previous baseline

Builds on L3-M1.1 (Agent Testing), L3-M1.2 (Quality Metrics),
L3-M1.4 (Agent Optimization), and L2-M3.1 (Custom Metrics).
"""

import json
import re
import time
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent as create_react_agent

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L3/M1_agent_evaluation/5_evaluation_pipeline")

# ---------------------------------------------------------------------------
# Tools for the ReAct agent
# ---------------------------------------------------------------------------
KNOWLEDGE = {
    "python": "Python is a high-level programming language known for readability. "
    "It supports multiple paradigms including OOP, functional, and procedural.",
    "mlflow": "MLflow is an open-source platform for the ML lifecycle. "
    "It provides tracking, model registry, evaluation, and deployment.",
    "langgraph": "LangGraph builds stateful multi-actor LLM applications "
    "using graph-based workflows with nodes, edges, and state.",
    "docker": "Docker is a containerization platform that packages apps "
    "and dependencies into portable containers for consistent deployment.",
    "kubernetes": "Kubernetes is a container orchestration system for "
    "automating deployment, scaling, and management of containerized apps.",
    "react": "ReAct (Reason + Act) is an agent pattern where the LLM "
    "alternates between reasoning about a task and taking actions.",
}


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information on a topic."""
    q = query.lower()
    results = [v for k, v in KNOWLEDGE.items() if k in q]
    return results[0] if results else f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3' or '10 * 5'."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))  # noqa: S307
        return "Invalid expression — only basic arithmetic is supported."
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------
def build_agent():
    """Create a ReAct agent with tools."""
    llm = ChatOllama(model="gemma4:e2b", temperature=0.0)
    return create_react_agent(llm, [search_knowledge, calculate])


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
def create_eval_dataset() -> pd.DataFrame:
    """Create the evaluation dataset with inputs, expected outputs, and metadata."""
    return pd.DataFrame([
        {"input": "What is Python?", "expected": "high-level programming language",
         "category": "knowledge", "needs_tool": "search_knowledge"},
        {"input": "What is MLflow used for?", "expected": "ML lifecycle",
         "category": "knowledge", "needs_tool": "search_knowledge"},
        {"input": "Explain LangGraph.", "expected": "stateful",
         "category": "knowledge", "needs_tool": "search_knowledge"},
        {"input": "What is 25 * 4?", "expected": "100",
         "category": "math", "needs_tool": "calculate"},
        {"input": "Calculate 144 / 12.", "expected": "12",
         "category": "math", "needs_tool": "calculate"},
        {"input": "What is Docker?", "expected": "containerization",
         "category": "knowledge", "needs_tool": "search_knowledge"},
    ])


# ---------------------------------------------------------------------------
# Quality gate definition
# ---------------------------------------------------------------------------
@dataclass
class QualityGates:
    min_accuracy: float = 0.6
    min_tool_accuracy: float = 0.5
    max_avg_latency_s: float = 120.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class AgentEvaluationPipeline:
    """End-to-end evaluation pipeline for an AI agent."""

    agent: object = None
    dataset: pd.DataFrame = field(default_factory=pd.DataFrame)
    results: list[dict] = field(default_factory=list)
    quality_gates: QualityGates = field(default_factory=QualityGates)
    metrics: dict = field(default_factory=dict)
    gate_results: dict = field(default_factory=dict)

    # -- Step 1 ------------------------------------------------------------ #
    def load_dataset(self) -> pd.DataFrame:
        """Load or create the evaluation dataset."""
        print("\n  [Step 1/5] Loading evaluation dataset...")
        self.dataset = create_eval_dataset()
        print(f"    Loaded {len(self.dataset)} test cases "
              f"({self.dataset['category'].nunique()} categories)")
        return self.dataset

    # -- Step 2 ------------------------------------------------------------ #
    def run_agent(self) -> list[dict]:
        """Execute the agent on each input and collect results."""
        print("\n  [Step 2/5] Running agent on test cases...")
        self.results = []
        for idx, row in self.dataset.iterrows():
            start = time.time()
            try:
                response = self.agent.invoke(
                    {"messages": [HumanMessage(content=row["input"])]}
                )
                latency = time.time() - start
                answer = response["messages"][-1].content

                # Detect which tools were called
                tools_used = [
                    m.name for m in response["messages"]
                    if hasattr(m, "name") and m.name
                ]

                self.results.append({
                    "input": row["input"],
                    "expected": row["expected"],
                    "output": answer,
                    "category": row["category"],
                    "needs_tool": row["needs_tool"],
                    "tools_used": tools_used,
                    "latency_s": round(latency, 2),
                    "error": None,
                })
            except Exception as e:
                self.results.append({
                    "input": row["input"],
                    "expected": row["expected"],
                    "output": "",
                    "category": row["category"],
                    "needs_tool": row["needs_tool"],
                    "tools_used": [],
                    "latency_s": round(time.time() - start, 2),
                    "error": str(e),
                })
            status = "OK" if not self.results[-1]["error"] else "ERR"
            print(f"    [{status}] Test {idx + 1}/{len(self.dataset)}: "
                  f"{row['input'][:40]}... ({self.results[-1]['latency_s']}s)")
        return self.results

    # -- Step 3 ------------------------------------------------------------ #
    def score_results(self) -> dict:
        """Score results with custom deterministic metrics."""
        print("\n  [Step 3/5] Scoring results...")
        correct = 0
        tool_correct = 0
        latencies = []

        for r in self.results:
            # Accuracy: does the output contain the expected keyword?
            if r["expected"].lower() in r["output"].lower():
                correct += 1
                r["score_correct"] = True
            else:
                r["score_correct"] = False

            # Tool accuracy: did the agent use the right tool?
            expected_tool = r["needs_tool"]
            if expected_tool in r["tools_used"]:
                tool_correct += 1
                r["score_tool_correct"] = True
            else:
                r["score_tool_correct"] = False

            latencies.append(r["latency_s"])

        n = len(self.results)
        self.metrics = {
            "accuracy": round(correct / n, 3) if n else 0,
            "tool_accuracy": round(tool_correct / n, 3) if n else 0,
            "avg_latency_s": round(sum(latencies) / n, 2) if n else 0,
            "max_latency_s": round(max(latencies), 2) if latencies else 0,
            "error_rate": round(sum(1 for r in self.results if r["error"]) / n, 3) if n else 0,
            "total_tests": n,
        }

        print(f"    Accuracy:      {self.metrics['accuracy']:.1%} ({correct}/{n})")
        print(f"    Tool accuracy: {self.metrics['tool_accuracy']:.1%} ({tool_correct}/{n})")
        print(f"    Avg latency:   {self.metrics['avg_latency_s']}s")
        print(f"    Error rate:    {self.metrics['error_rate']:.1%}")
        return self.metrics

    # -- Step 4 ------------------------------------------------------------ #
    def check_quality_gates(self) -> dict:
        """Check results against quality gate thresholds."""
        print("\n  [Step 4/5] Checking quality gates...")
        gates = {
            "accuracy": {
                "threshold": self.quality_gates.min_accuracy,
                "actual": self.metrics["accuracy"],
                "passed": self.metrics["accuracy"] >= self.quality_gates.min_accuracy,
            },
            "tool_accuracy": {
                "threshold": self.quality_gates.min_tool_accuracy,
                "actual": self.metrics["tool_accuracy"],
                "passed": self.metrics["tool_accuracy"] >= self.quality_gates.min_tool_accuracy,
            },
            "avg_latency": {
                "threshold": self.quality_gates.max_avg_latency_s,
                "actual": self.metrics["avg_latency_s"],
                "passed": self.metrics["avg_latency_s"] <= self.quality_gates.max_avg_latency_s,
            },
        }

        all_passed = all(g["passed"] for g in gates.values())
        self.gate_results = {"gates": gates, "all_passed": all_passed}

        for name, g in gates.items():
            status = "PASS" if g["passed"] else "FAIL"
            print(f"    [{status}] {name}: {g['actual']:.3f} "
                  f"(threshold: {g['threshold']})")

        print(f"\n    Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
        return self.gate_results

    # -- Step 5 ------------------------------------------------------------ #
    def generate_report(self) -> str:
        """Generate a text evaluation report."""
        print("\n  [Step 5/5] Generating evaluation report...")
        lines = [
            "=" * 60,
            "  AGENT EVALUATION REPORT",
            "=" * 60,
            f"  Total test cases: {self.metrics['total_tests']}",
            f"  Accuracy:         {self.metrics['accuracy']:.1%}",
            f"  Tool accuracy:    {self.metrics['tool_accuracy']:.1%}",
            f"  Avg latency:      {self.metrics['avg_latency_s']}s",
            f"  Error rate:       {self.metrics['error_rate']:.1%}",
            "",
            "  Quality Gates:",
        ]
        for name, g in self.gate_results.get("gates", {}).items():
            status = "PASS" if g["passed"] else "FAIL"
            lines.append(f"    [{status}] {name}: {g['actual']:.3f} >= {g['threshold']}")
        overall = self.gate_results.get("all_passed", False)
        lines.append(f"\n  Verdict: {'PASSED' if overall else 'FAILED'}")
        lines.append("")
        lines.append("  Per-test breakdown:")
        for i, r in enumerate(self.results, 1):
            c = "Y" if r.get("score_correct") else "N"
            t = "Y" if r.get("score_tool_correct") else "N"
            lines.append(f"    {i}. correct={c} tool={t} "
                         f"latency={r['latency_s']}s | {r['input'][:45]}")
        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        return report

    # -- Orchestrator ------------------------------------------------------ #
    def run_pipeline(self) -> dict:
        """Orchestrate the full evaluation pipeline, logging to MLflow."""
        print("=" * 60)
        print("  Agent Evaluation Pipeline — Starting")
        print("=" * 60)

        with mlflow.start_run(run_name="evaluation_pipeline") as parent_run:
            # Step 1: load dataset
            self.load_dataset()
            dataset_path = "/tmp/eval_dataset.csv"
            self.dataset.to_csv(dataset_path, index=False)
            mlflow.log_artifact(dataset_path, "datasets")

            # Step 2: run agent (log each test as nested run)
            self.run_agent()
            for i, r in enumerate(self.results):
                with mlflow.start_run(run_name=f"test_{i + 1}", nested=True):
                    mlflow.log_params({
                        "input": r["input"][:250],
                        "category": r["category"],
                        "needs_tool": r["needs_tool"],
                    })
                    mlflow.log_metrics({
                        "latency_s": r["latency_s"],
                    })
                    mlflow.set_tags({
                        "error": str(r["error"]) if r["error"] else "none",
                        "tools_used": ",".join(r["tools_used"]) if r["tools_used"] else "none",
                    })

            # Step 3: score
            self.score_results()
            mlflow.log_metrics(self.metrics)

            # Step 4: quality gates
            self.check_quality_gates()
            all_passed = self.gate_results["all_passed"]
            mlflow.set_tag("quality_gates_passed", str(all_passed))
            for name, g in self.gate_results["gates"].items():
                mlflow.set_tag(f"gate_{name}", "PASS" if g["passed"] else "FAIL")

            # Step 5: report
            report = self.generate_report()
            report_path = "/tmp/eval_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, "reports")

            # Log results table as artifact
            results_df = pd.DataFrame(self.results)
            results_path = "/tmp/eval_results.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, "results")

            run_id = parent_run.info.run_id

        print(f"\n  Pipeline complete. MLflow run ID: {run_id}")
        return {
            "run_id": run_id,
            "metrics": self.metrics,
            "gate_results": self.gate_results,
        }


# ---------------------------------------------------------------------------
# Part 2: Regression detection
# ---------------------------------------------------------------------------
def detect_regressions(current: dict, baseline: dict,
                       threshold: float = 0.1) -> list[dict]:
    """Compare current metrics to a baseline and detect regressions.

    A regression is detected when a metric drops by more than *threshold*
    (absolute) compared to the baseline.
    """
    regressions = []
    for metric in ["accuracy", "tool_accuracy"]:
        cur_val = current.get(metric, 0)
        base_val = baseline.get(metric, 0)
        delta = cur_val - base_val
        if delta < -threshold:
            regressions.append({
                "metric": metric,
                "baseline": base_val,
                "current": cur_val,
                "delta": round(delta, 3),
            })
    # Latency regression: increase beyond threshold (in seconds)
    cur_lat = current.get("avg_latency_s", 0)
    base_lat = baseline.get("avg_latency_s", 0)
    if base_lat > 0 and cur_lat > base_lat * 1.5:
        regressions.append({
            "metric": "avg_latency_s",
            "baseline": base_lat,
            "current": cur_lat,
            "delta": round(cur_lat - base_lat, 2),
        })
    return regressions


def run_regression_check(current_metrics: dict) -> None:
    """Simulate regression detection against a previous baseline."""
    print("\n" + "=" * 60)
    print("  Part 2: Regression Detection")
    print("=" * 60)

    # Simulate a "previous" baseline with slightly better scores
    baseline = {
        "accuracy": min(current_metrics.get("accuracy", 0.5) + 0.2, 1.0),
        "tool_accuracy": min(current_metrics.get("tool_accuracy", 0.5) + 0.15, 1.0),
        "avg_latency_s": max(current_metrics.get("avg_latency_s", 5.0) - 1.0, 1.0),
    }
    print(f"\n  Simulated baseline (previous run):")
    for k, v in baseline.items():
        print(f"    {k}: {v:.3f}")

    print(f"\n  Current results:")
    for k in baseline:
        v = current_metrics.get(k, 0)
        print(f"    {k}: {v:.3f}")

    regressions = detect_regressions(current_metrics, baseline)

    with mlflow.start_run(run_name="regression_check"):
        mlflow.log_params({
            f"baseline_{k}": v for k, v in baseline.items()
        })
        mlflow.log_params({
            f"current_{k}": current_metrics.get(k, 0) for k in baseline
        })
        mlflow.set_tag("regression_detected", str(len(regressions) > 0))
        mlflow.log_metric("regressions_found", len(regressions))

        if regressions:
            print(f"\n  REGRESSIONS DETECTED ({len(regressions)}):")
            for reg in regressions:
                print(f"    {reg['metric']}: {reg['baseline']:.3f} -> "
                      f"{reg['current']:.3f} (delta: {reg['delta']})")
                mlflow.set_tag(f"regression_{reg['metric']}", str(reg["delta"]))
        else:
            print("\n  No regressions detected.")

        # Log regression report
        report = json.dumps({"baseline": baseline, "current": current_metrics,
                             "regressions": regressions}, indent=2)
        report_path = "/tmp/regression_report.json"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")

    print("  Regression check logged to MLflow.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Build the agent
    agent = build_agent()

    # Part 1: Run full evaluation pipeline
    pipeline = AgentEvaluationPipeline(agent=agent)
    result = pipeline.run_pipeline()

    # Part 2: Regression detection
    run_regression_check(result["metrics"])

    print("\n" + "=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5000")
    print("  Experiment: L3/M1_agent_evaluation/5_evaluation_pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
