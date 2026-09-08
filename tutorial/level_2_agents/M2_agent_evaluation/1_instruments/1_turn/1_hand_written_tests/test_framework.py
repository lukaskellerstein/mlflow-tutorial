"""Hand-rolled agent test framework: case definitions, runner, reporting, baselines.

Nothing here is an MLflow API for testing agents -- MLflow ships those, and the
next three lessons use them. This is what you build yourself before you know
they exist, kept whole so the ceiling it hits is a measured fact rather than a
claim.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage
from mlflow.artifacts import download_artifacts


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class TestCase:
    """A single agent test case.

    Note what a case can express: one input, one expected substring, one set of
    expected tools. That shape is the framework's ceiling, not an oversight.
    """

    name: str
    input: str
    expected_output: str  # substring expected in the answer
    expected_tools: list[str]  # tools the agent should call
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


# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #
class AgentTestRunner:
    """Runs a suite of test cases against a LangGraph agent, logging each case as
    a nested MLflow child run."""

    def __init__(self, agent: Any, suite: list[TestCase]) -> None:
        self.agent = agent
        self.suite = suite

    def _extract_tools_called(self, messages: list[Any]) -> list[str]:
        """Read the names of the tools the agent called out of its messages."""
        tools_used: list[str] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") or tc.get("tool", "")
                    if name and name not in tools_used:
                        tools_used.append(name)
        return tools_used

    def _check_output(self, agent_output: str, expected: str) -> bool:
        # Substring matching. It is cheap, deterministic and free -- and it fails
        # a correct answer that used a synonym. Lesson 5 replaces it with a judge.
        return expected.lower() in agent_output.lower()

    def _check_tools(self, called: list[str], expected: list[str]) -> bool:
        return set(expected).issubset(set(called))

    def run_single(self, tc: TestCase) -> TestResult:
        """Execute one test case and return its result."""
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
        """Run every case, logging each as a nested MLflow child run."""
        results: list[TestResult] = []
        for idx, tc in enumerate(self.suite, 1):
            print(f"  [{idx}/{len(self.suite)}] {tc.name:<22}", end=" ", flush=True)
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
                print(f"         error: {tr.error}")
        return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_results_dataframe(results: list[TestResult]) -> pd.DataFrame:
    """Turn results into a DataFrame, for `mlflow.log_table` and for comparison."""
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
    """Print a formatted suite summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    output_ok = sum(1 for r in results if r.output_correct)
    tools_ok = sum(1 for r in results if r.tool_usage_correct)
    avg_dur = sum(r.duration_s for r in results) / max(total, 1)

    print(f"\n  {'-' * 56}")
    print(f"  passed             : {passed}/{total}  ({100 * passed / total:.0f}%)")
    print(f"  output correct     : {output_ok}/{total}")
    print(f"  tool usage correct : {tools_ok}/{total}")
    print(f"  average duration   : {avg_dur:.2f}s")

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if any(tc.difficulty == diff and tc.name == r.test_name for tc in suite)]
        if subset:
            p = sum(1 for r in subset if r.passed)
            print(f"  {diff:<6} pass rate    : {p}/{len(subset)}")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\n  failed:")
        for f in failures:
            reason = []
            if not f.output_correct:
                reason.append(f"output missing '{f.expected_output}'")
            if not f.tool_usage_correct:
                reason.append(f"tools called={f.tools_called} expected={f.expected_tools}")
            if f.error:
                reason.append(f"error: {f.error}")
            print(f"    - {f.test_name}: {'; '.join(reason)}")
    print(f"  {'-' * 56}")


# --------------------------------------------------------------------------- #
# Regression baselines
# --------------------------------------------------------------------------- #
BASELINE_ARTIFACT = "baselines/agent_test_baseline.json"


def save_baseline(df: pd.DataFrame, run_id: str) -> None:
    """Log the current results as a baseline artifact on the active run.

    The baseline belongs in MLflow, not on your laptop. A file in `/tmp` is gone
    after a reboot and means nothing to CI; an artifact on a run is addressable
    by run id from anywhere that can reach the tracking server.
    """
    baseline = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "pass_rate": float(df["passed"].mean()),
        "total_tests": len(df),
        "results": df.to_dict(orient="records"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_test_baseline.json"
        path.write_text(json.dumps(baseline, indent=2))
        mlflow.log_artifact(str(path), artifact_path="baselines")


def load_baseline(run_id: str) -> dict[str, Any]:
    """Download a baseline artifact back out of the run that produced it."""
    local = download_artifacts(run_id=run_id, artifact_path=BASELINE_ARTIFACT)
    with open(local) as fh:
        return json.load(fh)


def compare_to_baseline(current_df: pd.DataFrame, baseline: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Compare current results against a baseline and log the deltas.

    Returns (regressions, improvements) by test name.
    """
    baseline_df = pd.DataFrame(baseline["results"])
    current_rate = float(current_df["passed"].mean())
    baseline_rate = float(baseline["pass_rate"])
    delta = current_rate - baseline_rate

    print(f"\n  baseline pass rate : {baseline_rate:.0%}  (run {baseline['run_id'][:8]})")
    print(f"  current pass rate  : {current_rate:.0%}")
    print(f"  delta              : {delta:+.0%}")

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
        print("\n  no change from baseline")

    mlflow.log_metrics(
        {
            "baseline_pass_rate": baseline_rate,
            "current_pass_rate": current_rate,
            "delta_pass_rate": delta,
            "regressions_count": len(regressions),
            "improvements_count": len(improvements),
        }
    )
    return regressions, improvements
