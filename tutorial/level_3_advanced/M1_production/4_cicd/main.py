"""
L3-3.4 — CI/CD Quality Gates for AI Applications

Automated quality gates for LLM deployments:
  1. Define configurable quality gates (accuracy, latency, consistency, error rate)
  2. Run an evaluation harness against a set of test cases
  3. Check results against gate thresholds — pass or block deployment
  4. Simulate a full CI/CD pipeline with gate enforcement
  5. Track gate history over time and detect quality trends
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI


# ── Part 1: Quality Gate Definitions ──────────────────────
@dataclass
class QualityGate:
    """Configurable thresholds that a model must pass before deployment."""

    min_accuracy: float = 0.7
    max_latency_p95_ms: float = 5000.0
    min_consistency: float = 0.6
    max_error_rate: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {
            "min_accuracy": self.min_accuracy,
            "max_latency_p95_ms": self.max_latency_p95_ms,
            "min_consistency": self.min_consistency,
            "max_error_rate": self.max_error_rate,
        }


@dataclass
class GateResult:
    """Outcome for a single quality gate check."""

    gate_name: str
    threshold: float
    actual: float
    passed: bool
    detail: str


@dataclass
class EvalMetrics:
    """Aggregated metrics from an evaluation run."""

    accuracy: float
    latency_p95_ms: float
    consistency: float
    error_rate: float
    per_case: list[dict[str, Any]] = field(default_factory=list)


# ── Part 2: Evaluation Harness ────────────────────────────
TEST_CASES: list[dict[str, str]] = [
    {"input": "What is the capital of France?", "expected": "paris"},
    {"input": "What is 2 + 2?", "expected": "4"},
    {"input": "Name a primary color.", "expected": "red|blue|yellow"},
    {"input": "Is water wet? Answer yes or no.", "expected": "yes"},
    {"input": "What planet do we live on?", "expected": "earth"},
    {"input": "How many days are in a week?", "expected": "7"},
]


class EvaluationHarness:
    """Runs test cases against an LLM and collects quality metrics."""

    def __init__(self, model_name: str = "google/gemma-4-26b-a4b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=temperature,
        )

    def _check_answer(self, response: str, expected: str) -> bool:
        """Check if expected keyword(s) appear in the response."""
        response_lower = response.lower()
        for option in expected.split("|"):
            if option.strip() in response_lower:
                return True
        return False

    def run(self, test_cases: list[dict[str, str]], runs: int = 2) -> EvalMetrics:
        """Run test cases multiple times to measure accuracy and consistency."""
        per_case: list[dict[str, Any]] = []
        all_latencies: list[float] = []
        error_count = 0
        total_checks = 0
        correct_count = 0

        for tc in test_cases:
            case_results: list[bool] = []

            for attempt in range(runs):
                total_checks += 1
                start = time.time()
                try:
                    result = self.llm.invoke([{"role": "user", "content": tc["input"]}])
                    latency_ms = (time.time() - start) * 1000
                    all_latencies.append(latency_ms)

                    output = result.content
                    correct = self._check_answer(output, tc["expected"])
                    case_results.append(correct)
                    if correct:
                        correct_count += 1
                except Exception as e:
                    latency_ms = (time.time() - start) * 1000
                    all_latencies.append(latency_ms)
                    case_results.append(False)
                    error_count += 1

            # Consistency = fraction of runs that agreed with majority
            majority = sum(case_results) > len(case_results) / 2
            agreements = sum(1 for r in case_results if r == majority)
            consistency = agreements / len(case_results)

            per_case.append({
                "input": tc["input"],
                "expected": tc["expected"],
                "pass_rate": sum(case_results) / len(case_results),
                "consistency": consistency,
                "attempts": len(case_results),
            })

        # Aggregate metrics
        accuracy = correct_count / max(total_checks, 1)
        all_latencies.sort()
        p95_idx = int(len(all_latencies) * 0.95)
        latency_p95 = all_latencies[min(p95_idx, len(all_latencies) - 1)]
        avg_consistency = sum(c["consistency"] for c in per_case) / max(len(per_case), 1)
        error_rate = error_count / max(total_checks, 1)

        return EvalMetrics(
            accuracy=round(accuracy, 4),
            latency_p95_ms=round(latency_p95, 1),
            consistency=round(avg_consistency, 4),
            error_rate=round(error_rate, 4),
            per_case=per_case,
        )


# ── Part 3: Gate Checker ──────────────────────────────────
class GateChecker:
    """Evaluates metrics against quality gate thresholds."""

    def __init__(self, gate: QualityGate):
        self.gate = gate

    def check(self, metrics: EvalMetrics) -> list[GateResult]:
        """Run all gate checks and return individual results."""
        results: list[GateResult] = []

        results.append(GateResult(
            gate_name="accuracy",
            threshold=self.gate.min_accuracy,
            actual=metrics.accuracy,
            passed=metrics.accuracy >= self.gate.min_accuracy,
            detail=f"Accuracy {metrics.accuracy:.2%} >= {self.gate.min_accuracy:.2%}",
        ))

        results.append(GateResult(
            gate_name="latency_p95",
            threshold=self.gate.max_latency_p95_ms,
            actual=metrics.latency_p95_ms,
            passed=metrics.latency_p95_ms <= self.gate.max_latency_p95_ms,
            detail=f"P95 latency {metrics.latency_p95_ms:.0f}ms <= {self.gate.max_latency_p95_ms:.0f}ms",
        ))

        results.append(GateResult(
            gate_name="consistency",
            threshold=self.gate.min_consistency,
            actual=metrics.consistency,
            passed=metrics.consistency >= self.gate.min_consistency,
            detail=f"Consistency {metrics.consistency:.2%} >= {self.gate.min_consistency:.2%}",
        ))

        results.append(GateResult(
            gate_name="error_rate",
            threshold=self.gate.max_error_rate,
            actual=metrics.error_rate,
            passed=metrics.error_rate <= self.gate.max_error_rate,
            detail=f"Error rate {metrics.error_rate:.2%} <= {self.gate.max_error_rate:.2%}",
        ))

        return results

    @staticmethod
    def print_report(results: list[GateResult]) -> None:
        """Print a formatted gate report."""
        all_passed = all(r.passed for r in results)

        print(f"\n{'=' * 60}")
        print("  Quality Gate Report")
        print(f"{'=' * 60}")
        for r in results:
            icon = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {icon} {r.gate_name:20s} | {r.detail}")
        print(f"{'-' * 60}")
        verdict = "ALL GATES PASSED" if all_passed else "DEPLOYMENT BLOCKED"
        print(f"  Verdict: {verdict}")
        print(f"{'=' * 60}")


# ── Part 4: Simulate CI/CD Pipeline ───────────────────────
def run_cicd_pipeline(
    gate: QualityGate,
    model_name: str = "google/gemma-4-26b-a4b",
    pipeline_label: str = "ci-run",
) -> tuple[EvalMetrics, list[GateResult]]:
    """Simulate a CI/CD pipeline: evaluate, gate-check, log to MLflow."""

    print(f"\n{'=' * 60}")
    print(f"  CI/CD Pipeline: {pipeline_label}")
    print(f"{'=' * 60}")

    # Step 1: Run evaluation harness
    print("\n  Step 1: Running evaluation harness ...")
    harness = EvaluationHarness(model_name=model_name, temperature=0.0)
    metrics = harness.run(TEST_CASES, runs=2)

    print(f"    Accuracy:      {metrics.accuracy:.2%}")
    print(f"    P95 latency:   {metrics.latency_p95_ms:.0f} ms")
    print(f"    Consistency:   {metrics.consistency:.2%}")
    print(f"    Error rate:    {metrics.error_rate:.2%}")

    # Step 2: Check quality gates
    print("\n  Step 2: Checking quality gates ...")
    checker = GateChecker(gate)
    gate_results = checker.check(metrics)
    checker.print_report(gate_results)

    all_passed = all(r.passed for r in gate_results)

    # Step 3/4: Deploy decision
    if all_passed:
        print("\n  Step 3: Deploy APPROVED -- all quality gates passed")
    else:
        failed = [r for r in gate_results if not r.passed]
        print(f"\n  Step 4: Deploy BLOCKED -- {len(failed)} gate(s) failed:")
        for r in failed:
            print(f"    - {r.gate_name}: {r.detail}")

    # Log everything to MLflow
    with mlflow.start_run(run_name=pipeline_label):
        mlflow.log_params({
            "model_name": model_name,
            "num_test_cases": len(TEST_CASES),
            "pipeline_label": pipeline_label,
            **{f"gate_{k}": v for k, v in gate.to_dict().items()},
        })
        mlflow.log_metrics({
            "accuracy": metrics.accuracy,
            "latency_p95_ms": metrics.latency_p95_ms,
            "consistency": metrics.consistency,
            "error_rate": metrics.error_rate,
            "gates_passed": int(all_passed),
            "gates_failed_count": sum(1 for r in gate_results if not r.passed),
        })
        mlflow.set_tags({
            "pipeline_type": "cicd_quality_gate",
            "deploy_decision": "approved" if all_passed else "blocked",
        })

        # Log per-case results as artifact
        df = pd.DataFrame(metrics.per_case)
        csv_path = "/tmp/cicd_eval_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="evaluation")

        # Log gate report as artifact
        report = {
            "gates": [
                {"gate": r.gate_name, "threshold": r.threshold,
                 "actual": r.actual, "passed": r.passed, "detail": r.detail}
                for r in gate_results
            ],
            "all_passed": all_passed,
        }
        report_path = "/tmp/cicd_gate_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="gates")

    return metrics, gate_results


# ── Part 5: Gate History and Trend Analysis ───────────────
def analyze_gate_history() -> None:
    """Query past gate runs from MLflow and show quality trends."""

    print(f"\n{'=' * 60}")
    print("  Gate History & Trend Analysis")
    print(f"{'=' * 60}")

    runs = mlflow.search_runs(
        experiment_names=["L3/M3_production/4_cicd"],
        filter_string="tags.pipeline_type = 'cicd_quality_gate'",
        order_by=["start_time ASC"],
        max_results=20,
    )

    if runs.empty:
        print("  No gate history found.")
        return

    cols = {
        "run_id": "run_id",
        "tags.deploy_decision": "decision",
        "metrics.accuracy": "accuracy",
        "metrics.latency_p95_ms": "latency_p95",
        "metrics.consistency": "consistency",
        "metrics.error_rate": "error_rate",
    }
    available = [c for c in cols if c in runs.columns]
    history = runs[available].rename(columns={k: v for k, v in cols.items() if k in available})

    print(f"\n  Found {len(history)} pipeline run(s):\n")
    for idx, row in history.iterrows():
        run_short = str(row.get("run_id", ""))[:8]
        decision = row.get("decision", "unknown")
        acc = row.get("accuracy", float("nan"))
        lat = row.get("latency_p95", float("nan"))
        icon = "[PASS]" if decision == "approved" else "[FAIL]"
        print(f"    {icon} {run_short}  acc={acc:.2%}  p95={lat:.0f}ms  -> {decision}")

    # Trend detection
    if len(history) >= 2 and "accuracy" in history.columns:
        acc_vals = history["accuracy"].dropna().tolist()
        if len(acc_vals) >= 2:
            trend = acc_vals[-1] - acc_vals[0]
            direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
            print(f"\n  Accuracy trend: {direction} ({trend:+.2%} over {len(acc_vals)} runs)")

    print()


# ── Main ──────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("L3-3.4 — CI/CD Quality Gates for AI Applications")
    print("=" * 60)

    # --- Part 1: Define quality gates ---
    print("\n--- Part 1: Quality Gate Definitions ---")
    gate = QualityGate()
    print(f"  Thresholds: {gate.to_dict()}")

    # --- Part 4a: Run pipeline with standard gates (should pass) ---
    print("\n--- Part 4a: CI/CD Pipeline - Standard Gates ---")
    run_cicd_pipeline(gate, pipeline_label="ci-standard-gates")

    # --- Part 4b: Run pipeline with strict gates (may fail) ---
    print("\n--- Part 4b: CI/CD Pipeline - Strict Gates ---")
    strict_gate = QualityGate(
        min_accuracy=0.95,
        max_latency_p95_ms=500.0,
        min_consistency=0.95,
        max_error_rate=0.0,
    )
    print(f"  Strict thresholds: {strict_gate.to_dict()}")
    run_cicd_pipeline(strict_gate, pipeline_label="ci-strict-gates")

    # --- Part 5: Gate history ---
    print("\n--- Part 5: Gate History ---")
    analyze_gate_history()

    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("Look at experiment: L3/M3_production/4_cicd")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M1_production/4_cicd")
    main()
