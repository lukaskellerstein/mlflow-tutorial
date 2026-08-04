"""
L2-M2.6 — Evaluation Pipeline: Offline Gates and Online Scoring

The two questions an evaluation setup has to answer, and they are not the same:

  OFFLINE -- "is this version good enough to ship?"
    Part 1: the full pipeline (dataset -> agent -> score -> quality gates -> report)
    Part 2: regression detection against a stored baseline
  ONLINE  -- "is what shipped still good?"
    Part 3: a registered judge on a gateway model, sampling live traces, run by
            the server rather than by you

Builds on L2-M2.1 (Test Generation), L2-M2.2 (Judges), L2-M2.3 (Quality Metrics)
and L2-M2.5 (Optimization).
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# The LiteLLM gateway from infra/, not a provider directly. "gemma-large" is an
# alias defined in infra/litellm/config.yaml -- swapping model or provider is a
# change there, never here. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-large"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L2/M2_agent_evaluation/5_evaluation_pipeline")

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
            return str(eval(expression))  # nosec: reached only for whitelisted arithmetic chars
        return "Invalid expression — only basic arithmetic is supported."
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------
def build_agent():
    """Create a ReAct agent with tools."""
    llm = ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )
    return create_agent(llm, [search_knowledge, calculate])


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
def create_eval_dataset() -> pd.DataFrame:
    """Create the evaluation dataset with inputs, expected outputs, and metadata."""
    return pd.DataFrame(
        [
            {
                "input": "What is Python?",
                "expected": "high-level programming language",
                "category": "knowledge",
                "needs_tool": "search_knowledge",
            },
            {
                "input": "What is MLflow used for?",
                "expected": "ML lifecycle",
                "category": "knowledge",
                "needs_tool": "search_knowledge",
            },
            {
                "input": "Explain LangGraph.",
                "expected": "stateful",
                "category": "knowledge",
                "needs_tool": "search_knowledge",
            },
            {
                "input": "What is 25 * 4?",
                "expected": "100",
                "category": "math",
                "needs_tool": "calculate",
            },
            {
                "input": "Calculate 144 / 12.",
                "expected": "12",
                "category": "math",
                "needs_tool": "calculate",
            },
            {
                "input": "What is Docker?",
                "expected": "containerization",
                "category": "knowledge",
                "needs_tool": "search_knowledge",
            },
        ]
    )


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

    agent: Any = None
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
        print(f"    Loaded {len(self.dataset)} test cases ({self.dataset['category'].nunique()} categories)")
        return self.dataset

    # -- Step 2 ------------------------------------------------------------ #
    def run_agent(self) -> list[dict]:
        """Execute the agent on each input and collect results."""
        print("\n  [Step 2/5] Running agent on test cases...")
        self.results = []
        for idx, (_, row) in enumerate(self.dataset.iterrows()):
            start = time.time()
            try:
                response = self.agent.invoke({"messages": [{"role": "user", "content": row["input"]}]})
                latency = time.time() - start
                answer = response["messages"][-1].content

                # Detect which tools were called
                tools_used = [m.name for m in response["messages"] if hasattr(m, "name") and m.name]

                self.results.append(
                    {
                        "input": row["input"],
                        "expected": row["expected"],
                        "output": answer,
                        "category": row["category"],
                        "needs_tool": row["needs_tool"],
                        "tools_used": tools_used,
                        "latency_s": round(latency, 2),
                        "error": None,
                    }
                )
            except Exception as e:
                self.results.append(
                    {
                        "input": row["input"],
                        "expected": row["expected"],
                        "output": "",
                        "category": row["category"],
                        "needs_tool": row["needs_tool"],
                        "tools_used": [],
                        "latency_s": round(time.time() - start, 2),
                        "error": str(e),
                    }
                )
            status = "OK" if not self.results[-1]["error"] else "ERR"
            print(
                f"    [{status}] Test {idx + 1}/{len(self.dataset)}: "
                f"{row['input'][:40]}... ({self.results[-1]['latency_s']}s)"
            )
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
            print(f"    [{status}] {name}: {g['actual']:.3f} (threshold: {g['threshold']})")

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
            lines.append(f"    {i}. correct={c} tool={t} latency={r['latency_s']}s | {r['input'][:45]}")
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
                    mlflow.log_params(
                        {
                            "input": r["input"][:250],
                            "category": r["category"],
                            "needs_tool": r["needs_tool"],
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "latency_s": r["latency_s"],
                        }
                    )
                    mlflow.set_tags(
                        {
                            "error": str(r["error"]) if r["error"] else "none",
                            "tools_used": ",".join(r["tools_used"]) if r["tools_used"] else "none",
                        }
                    )

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


# ── Part 2: Regression detection ──────────────────────────────────
def detect_regressions(current: dict, baseline: dict, threshold: float = 0.1) -> list[dict]:
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
            regressions.append(
                {
                    "metric": metric,
                    "baseline": base_val,
                    "current": cur_val,
                    "delta": round(delta, 3),
                }
            )
    # Latency regression: increase beyond threshold (in seconds)
    cur_lat = current.get("avg_latency_s", 0)
    base_lat = baseline.get("avg_latency_s", 0)
    if base_lat > 0 and cur_lat > base_lat * 1.5:
        regressions.append(
            {
                "metric": "avg_latency_s",
                "baseline": base_lat,
                "current": cur_lat,
                "delta": round(cur_lat - base_lat, 2),
            }
        )
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
    print("\n  Simulated baseline (previous run):")
    for k, v in baseline.items():
        print(f"    {k}: {v:.3f}")

    print("\n  Current results:")
    for k in baseline:
        v = current_metrics.get(k, 0)
        print(f"    {k}: {v:.3f}")

    regressions = detect_regressions(current_metrics, baseline)

    with mlflow.start_run(run_name="regression_check"):
        mlflow.log_params({f"baseline_{k}": v for k, v in baseline.items()})
        mlflow.log_params({f"current_{k}": current_metrics.get(k, 0) for k in baseline})
        mlflow.set_tag("regression_detected", str(len(regressions) > 0))
        mlflow.log_metric("regressions_found", len(regressions))

        if regressions:
            print(f"\n  REGRESSIONS DETECTED ({len(regressions)}):")
            for reg in regressions:
                print(f"    {reg['metric']}: {reg['baseline']:.3f} -> {reg['current']:.3f} (delta: {reg['delta']})")
                mlflow.set_tag(f"regression_{reg['metric']}", str(reg["delta"]))
        else:
            print("\n  No regressions detected.")

        # Log regression report
        report = json.dumps({"baseline": baseline, "current": current_metrics, "regressions": regressions}, indent=2)
        report_path = "/tmp/regression_report.json"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")

    print("  Regression check logged to MLflow.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Part 3: online scoring
#
# Everything above is OFFLINE evaluation: a curated dataset, known expectations,
# every case scored, and you decide when it runs. It answers "is this version
# good enough to ship?"
#
# It cannot answer "is what shipped still good?" -- for that the traces come from
# production, there are no expectations, coverage is sampled because judges cost
# money per call, and the SERVER pulls the trigger on a schedule rather than you.
#
# The mechanism is a registered judge with a sampling config. One catch decides
# the whole design: `start()` refuses any judge whose model is not a GATEWAY
# model, because the scoring runs server-side and the server needs its own
# credentialed endpoint -- it cannot borrow the API key from your shell.
# ---------------------------------------------------------------------------
GATEWAY_SECRET_NAME = "openrouter-tutorial"
GATEWAY_MODEL_NAME = "or-gemma-large"
GATEWAY_ENDPOINT_NAME = "tutorial-gemma-endpoint"
# The gateway calls OpenRouter directly, so this is the upstream model id, not
# the LiteLLM alias the rest of the module uses.
UPSTREAM_MODEL = "google/gemma-4-26b-a4b-it:free"
ONLINE_JUDGE_NAME = "production_answer_quality"


def ensure_gateway_endpoint() -> str:
    """Build (or reuse) secret -> model definition -> endpoint. Returns its name.

    Why not point this at the LiteLLM proxy like every other lesson? Because it
    does not work: `create_gateway_secret(auth_config={"base_url": ...})` accepts
    the value and silently ignores it -- the request still goes to the provider's
    own API, and the first symptom is an authentication error from OpenAI about a
    key you never sent it. `provider="openrouter"` is supported natively, so the
    server talks to OpenRouter directly and no base URL is needed.
    """
    from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
    from mlflow.tracking._tracking_service.utils import _get_store

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. The gateway secret is created from the\n"
            "environment -- never hardcode it. `source infra/.env` and re-run."
        )

    store = _get_store()

    existing = {e.name: e for e in store.list_gateway_endpoints()}
    if GATEWAY_ENDPOINT_NAME in existing:
        print(f"  reusing gateway endpoint '{GATEWAY_ENDPOINT_NAME}'")
        return GATEWAY_ENDPOINT_NAME

    secrets = {s.secret_name: s for s in store.list_secret_infos()}
    secret = secrets.get(GATEWAY_SECRET_NAME) or store.create_gateway_secret(
        secret_name=GATEWAY_SECRET_NAME,
        secret_value={"api_key": api_key},
        provider="openrouter",
    )

    defs = {d.name: d for d in store.list_gateway_model_definitions()}
    model_def = defs.get(GATEWAY_MODEL_NAME) or store.create_gateway_model_definition(
        name=GATEWAY_MODEL_NAME,
        secret_id=secret.secret_id,
        provider="openrouter",
        model_name=UPSTREAM_MODEL,
    )

    store.create_gateway_endpoint(
        name=GATEWAY_ENDPOINT_NAME,
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                # The ENUM, not the string "PRIMARY" -- a string fails deep in
                # proto serialisation with 'str' object has no attribute 'to_proto'.
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1,
                fallback_order=0,
            )
        ],
    )
    print(f"  created gateway endpoint '{GATEWAY_ENDPOINT_NAME}' -> openrouter/{UPSTREAM_MODEL}")
    return GATEWAY_ENDPOINT_NAME


def run_online_scoring(agent: Any) -> None:
    """Register a judge against the gateway, sample live traffic, then stop."""
    from mlflow.genai.scorers import ScorerSamplingConfig

    print("\n" + "=" * 60)
    print("  Part 3: Online scoring (the other half of the question)")
    print("=" * 60)

    endpoint = ensure_gateway_endpoint()

    judge = mlflow.genai.make_judge(
        name=ONLINE_JUDGE_NAME,
        instructions=(
            "You are reviewing a live support answer.\n"
            "The request is in {{ inputs }} and the agent's reply is in {{ outputs }}.\n"
            "Is the reply accurate and genuinely useful to the user? Answer true or false."
        ),
        model=f"gateway:/{endpoint}",
        feedback_value_type=bool,
    )
    registered = judge.register(name=ONLINE_JUDGE_NAME)
    print(f"  registered judge '{registered.name}' on model gateway:/{endpoint}")

    # sample_rate is the whole economic argument for online scoring: judging is a
    # model call per trace, so cost scales with TRAFFIC, not with dataset size.
    # 20% is a deliberate choice, not a default.
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
    print(f"  started: status={started.status} sample_rate={started.sample_rate}")

    print("\n  sending live traffic (these traces are what the server will sample)...")
    for question in ["What is 25 * 4?", "What does the knowledge base say about mlflow?"]:
        agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"    -> {question}")
    mlflow.flush_trace_async_logging()

    fresh = mlflow.genai.get_scorer(name=ONLINE_JUDGE_NAME)
    print(f"\n  server-side state: status={fresh.status} sample_rate={fresh.sample_rate}")
    print("  The scheduler picks active scorers up on its own cadence, so assessments")
    print("  appear on sampled traces shortly -- not synchronously with this script.")

    stopped = fresh.stop()
    print(f"  stopped: status={stopped.status}  (left running, it would score forever)")

    print("\n  offline vs online, on four axes:")
    print(f"    {'':<12}{'offline':<26}{'online'}")
    print(f"    {'input':<12}{'curated dataset':<26}{'production traces'}")
    print(f"    {'truth':<12}{'expectations':<26}{'none'}")
    print(f"    {'coverage':<12}{'every case':<26}{'sampled (20% here)'}")
    print(f"    {'trigger':<12}{'you, in CI':<26}{'the server, on a schedule'}")


def main() -> None:
    # Build the agent
    agent = build_agent()

    # Part 1: Run full evaluation pipeline
    pipeline = AgentEvaluationPipeline(agent=agent)
    result = pipeline.run_pipeline()

    # Part 2: Regression detection
    run_regression_check(result["metrics"])

    # Part 3: the other half of the question
    run_online_scoring(agent)

    print("\n" + "=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5555")
    print("  Experiment: L2/M2_agent_evaluation/5_evaluation_pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
