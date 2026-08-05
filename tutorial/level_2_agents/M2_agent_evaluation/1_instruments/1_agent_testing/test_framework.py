"""
Agent testing framework: test case definitions, runner, and reporting utilities.
"""

import json
import time
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    """A single agent test case."""

    name: str
    input: str
    expected_output: str  # substring or keyword expected in answer
    expected_tools: list[str]  # tools the agent should use
    difficulty: str  # easy / medium / hard
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a single test case."""

    test_name: str
    passed: bool
    output_correct: bool
    tool_usage_correct: bool
    agent_output: str
    expected_output: str
    tools_called: list[str]
    expected_tools: list[str]
    duration_s: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
class AgentTestRunner:
    """Runs a suite of test cases against a LangGraph agent and logs results
    to MLflow with nested runs."""

    def __init__(self, agent, suite: list[TestCase]):
        self.agent = agent
        self.suite = suite

    def _extract_tools_called(self, messages: list) -> list[str]:
        """Extract the names of tools called from the agent's message history."""
        tools_used: list[str] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") or tc.get("tool", "")
                    if name and name not in tools_used:
                        tools_used.append(name)
        return tools_used

    def _check_output(self, agent_output: str, expected: str) -> bool:
        return expected.lower() in agent_output.lower()

    def _check_tools(self, called: list[str], expected: list[str]) -> bool:
        return set(expected).issubset(set(called))

    def run_single(self, tc: TestCase) -> TestResult:
        """Execute one test case and return the result."""
        start = time.time()
        try:
            result = self.agent.invoke({"messages": [HumanMessage(content=tc.input)]})
            elapsed = time.time() - start
            agent_output = result["messages"][-1].content
            tools_called = self._extract_tools_called(result["messages"])
            output_ok = self._check_output(agent_output, tc.expected_output)
            tools_ok = self._check_tools(tools_called, tc.expected_tools)

            return TestResult(
                test_name=tc.name,
                passed=output_ok and tools_ok,
                output_correct=output_ok,
                tool_usage_correct=tools_ok,
                agent_output=agent_output[:500],
                expected_output=tc.expected_output,
                tools_called=tools_called,
                expected_tools=tc.expected_tools,
                duration_s=round(elapsed, 2),
            )
        except Exception as e:
            elapsed = time.time() - start
            return TestResult(
                test_name=tc.name,
                passed=False,
                output_correct=False,
                tool_usage_correct=False,
                agent_output="",
                expected_output=tc.expected_output,
                tools_called=[],
                expected_tools=tc.expected_tools,
                duration_s=round(elapsed, 2),
                error=str(e),
            )

    def run_suite(self) -> list[TestResult]:
        """Run all test cases and log each as a nested MLflow child run."""
        results: list[TestResult] = []
        for idx, tc in enumerate(self.suite, 1):
            print(f"  [{idx}/{len(self.suite)}] Running: {tc.name} ...", end=" ", flush=True)
            tr = self.run_single(tc)
            results.append(tr)

            with mlflow.start_run(run_name=f"test_{tc.name}", nested=True):
                mlflow.log_params(
                    {
                        "test_name": tc.name,
                        "difficulty": tc.difficulty,
                        "expected_tools": json.dumps(tc.expected_tools),
                        "expected_output": tc.expected_output[:250],
                        "input": tc.input[:250],
                    }
                )
                mlflow.log_metrics(
                    {
                        "passed": int(tr.passed),
                        "output_correct": int(tr.output_correct),
                        "tool_usage_correct": int(tr.tool_usage_correct),
                        "duration_s": tr.duration_s,
                    }
                )
                mlflow.set_tags(
                    {
                        "status": "PASS" if tr.passed else "FAIL",
                        "difficulty": tc.difficulty,
                        **tc.tags,
                    }
                )

            print(f"{'PASS' if tr.passed else 'FAIL'}  ({tr.duration_s}s)")
            if tr.error:
                print(f"         Error: {tr.error}")
        return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def build_results_dataframe(results: list[TestResult]) -> pd.DataFrame:
    """Convert test results into a pandas DataFrame."""
    return pd.DataFrame(
        [
            {
                "test_name": r.test_name,
                "passed": r.passed,
                "output_correct": r.output_correct,
                "tool_usage_correct": r.tool_usage_correct,
                "duration_s": r.duration_s,
                "tools_called": json.dumps(r.tools_called),
                "expected_tools": json.dumps(r.expected_tools),
                "agent_output": r.agent_output[:200],
                "error": r.error or "",
            }
            for r in results
        ]
    )


def print_summary(results: list[TestResult], suite: list[TestCase]) -> None:
    """Print a formatted test-suite summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    output_ok = sum(1 for r in results if r.output_correct)
    tools_ok = sum(1 for r in results if r.tool_usage_correct)
    avg_dur = sum(r.duration_s for r in results) / max(total, 1)

    print(f"\n{'=' * 60}")
    print("  Test Suite Summary")
    print(f"{'=' * 60}")
    print(f"  Total tests:           {total}")
    print(f"  Passed:                {passed}/{total}  ({100 * passed / total:.0f}%)")
    print(f"  Output correct:        {output_ok}/{total}")
    print(f"  Tool usage correct:    {tools_ok}/{total}")
    print(f"  Average duration:      {avg_dur:.2f}s")

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if any(tc.difficulty == diff and tc.name == r.test_name for tc in suite)]
        if subset:
            p = sum(1 for r in subset if r.passed)
            print(f"  {diff.capitalize():8s} pass rate:   {p}/{len(subset)}")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\n  Failed tests:")
        for f in failures:
            reason = []
            if not f.output_correct:
                reason.append("output mismatch")
            if not f.tool_usage_correct:
                reason.append(f"tool mismatch (called={f.tools_called}, expected={f.expected_tools})")
            if f.error:
                reason.append(f"error: {f.error}")
            print(f"    - {f.test_name}: {'; '.join(reason)}")
    print()


def save_baseline(df: pd.DataFrame, run_id: str) -> str:
    """Save test results as a JSON baseline artifact in the current run."""
    baseline = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "pass_rate": float(df["passed"].mean()),
        "total_tests": len(df),
        "results": df.to_dict(orient="records"),
    }
    path = "/tmp/agent_test_baseline.json"
    with open(path, "w") as fh:
        json.dump(baseline, fh, indent=2)
    mlflow.log_artifact(path, artifact_path="baselines")
    return path


def compare_to_baseline(current_df: pd.DataFrame, baseline_path: str) -> None:
    """Compare current test results against a saved baseline."""
    with open(baseline_path) as fh:
        baseline = json.load(fh)

    baseline_df = pd.DataFrame(baseline["results"])
    current_rate = float(current_df["passed"].mean())
    baseline_rate = baseline["pass_rate"]
    delta = current_rate - baseline_rate

    print(f"{'=' * 60}")
    print("  Regression Comparison")
    print(f"{'=' * 60}")
    print(f"  Baseline pass rate:  {baseline_rate:.0%}  (run {baseline['run_id'][:8]})")
    print(f"  Current pass rate:   {current_rate:.0%}")
    print(f"  Delta:               {delta:+.0%}")

    regressions, improvements = [], []
    for _, row in current_df.iterrows():
        name = row["test_name"]
        bl_row = baseline_df[baseline_df["test_name"] == name]
        if bl_row.empty:
            continue
        was_passing = bool(bl_row.iloc[0]["passed"])
        now_passing = bool(row["passed"])
        if was_passing and not now_passing:
            regressions.append(name)
        elif not was_passing and now_passing:
            improvements.append(name)

    if regressions:
        print(f"\n  REGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(f"    - {r}")
    if improvements:
        print(f"\n  IMPROVEMENTS ({len(improvements)}):")
        for i in improvements:
            print(f"    + {i}")
    if not regressions and not improvements:
        print("\n  No changes from baseline.")

    mlflow.log_metrics(
        {
            "baseline_pass_rate": baseline_rate,
            "current_pass_rate": current_rate,
            "delta_pass_rate": delta,
            "regressions_count": len(regressions),
            "improvements_count": len(improvements),
        }
    )
    print()
