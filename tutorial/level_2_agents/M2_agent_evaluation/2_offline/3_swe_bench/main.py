"""L2-M2.2.3 -- SWE-Bench Evaluation Pipeline.

Uses a Claude Agent SDK coding agent that runs inside a container sandbox
to fix real GitHub issues from SWE-Bench Verified. The agent edits files
directly with its built-in tools (see agent.py), then we run the repo
test suite and score results using the same methodology as the SWE-Bench
leaderboard. All results are tracked in MLflow with hand-built tracing.
"""

import json
import tempfile
import time

import agent as agent_mod
import anyio
import datasets
import harness
import mlflow
import pandas as pd
from claude_agent_sdk import EffortLevel

SAMPLE_SIZE = 2
SAMPLE_REPO = "sympy/sympy"


# -- Helpers -------------------------------------------------------------------


def select_instances(ds, repo: str, count: int) -> list[dict]:
    """Filter dataset to the newest instances from a specific repo.

    Newest, not first: the harness runs every instance on one python:3.11
    image, and old sympy (pre-1.9) cannot even import on 3.11 -- e.g. the
    2016 instances use `from collections import Mapping`, removed in 3.10.
    The real SWE-Bench harness builds a per-instance environment instead.
    """
    instances = [ds[i] for i in range(len(ds)) if ds[i]["repo"] == repo]
    instances.sort(key=lambda inst: inst["created_at"])
    return instances[-count:]


def save_text_artifact(text: str, filename: str) -> None:
    """Log a text string as an MLflow artifact.

    log_artifact must run after the with block: it copies the file from disk,
    and inside the block the write may still be sitting in the buffer -- the
    artifact then uploads empty.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix=filename + "_", delete=False) as f:
        f.write(text)
        path = f.name
    mlflow.log_artifact(path)


async def run_instance(effort: EffortLevel, instance: dict, config_name: str) -> dict:
    """Run the full evaluation pipeline for one SWE-Bench instance."""
    iid = instance["instance_id"]
    repo = instance["repo"]
    f2p_tests = json.loads(instance["FAIL_TO_PASS"])
    p2p_tests = json.loads(instance["PASS_TO_PASS"])[: harness.MAX_P2P_TESTS]
    test_files = harness.test_files_from_patch(instance.get("test_patch", ""))

    with mlflow.start_run(run_name=f"{config_name}_{iid}", nested=True):
        mlflow.log_params(
            {
                "instance_id": iid,
                "repo": repo,
                "base_commit": instance["base_commit"][:8],
                "config": config_name,
                "f2p_test_count": len(f2p_tests),
                "p2p_test_count": len(p2p_tests),
            }
        )
        start = time.perf_counter()

        # -- Step A: Start container and setup repo ----------------------------
        print(f"\n  [{iid}]")
        try:
            print("    Starting container ...")
            container_id = harness.start_container(iid)
        except RuntimeError as e:
            latency = time.perf_counter() - start
            print(f"    Container start failed: {e}")
            mlflow.set_tag("status", "error")
            return _error_result(iid, repo, config_name, latency, str(e))

        try:
            if not harness.setup_repo(container_id, repo, instance["base_commit"]):
                latency = time.perf_counter() - start
                mlflow.set_tag("status", "setup_failed")
                harness.cleanup_container(container_id)
                return _error_result(iid, repo, config_name, latency, "repo setup failed")

            # -- Step B: Apply test patch and verify baseline ------------------
            if instance.get("test_patch"):
                print("    Applying test patch ...")
                harness.apply_patch(container_id, instance["test_patch"], "test.patch")

            print(f"    Baseline check: running {len(f2p_tests)} FAIL_TO_PASS tests ...")
            base_passed, base_failed, _base_out = harness.run_tests(container_id, f2p_tests, test_files)
            print(f"    Baseline: {base_passed} passed, {base_failed} failed (expect failures)")

            # -- Step C: Run the Claude Agent SDK agent to fix the bug ---------
            print(f"    Running agent (model={agent_mod.MODEL}, effort={effort}) ...")
            options = agent_mod.build_options(effort, container_id)
            try:
                agent_result = await agent_mod.run_agent(agent_mod.build_prompt(instance), options)
            except Exception as e:
                latency = time.perf_counter() - start
                print(f"    Agent failed: {e}")
                mlflow.set_tag("status", "error")
                harness.cleanup_container(container_id)
                return _error_result(iid, repo, config_name, latency, f"agent error: {e}")

            print(
                f"    Agent finished: {agent_result['num_turns']} turns, "
                f"{len(agent_result['tool_calls'])} tool calls"
                + (f", ${agent_result['cost_usd']:.4f}" if agent_result["cost_usd"] else "")
            )

            # -- Step D: Capture diff and run tests ----------------------------
            _, agent_diff, _ = harness.exec_in_container(container_id, "cd /workspace/repo && git diff")
            save_text_artifact(agent_result["response"], "agent_output")
            save_text_artifact(agent_mod.summarize_tool_calls(agent_result["tool_calls"]), "agent_tool_calls")
            save_text_artifact(agent_diff or "(no changes)", "agent_diff")
            save_text_artifact(instance.get("patch", ""), "gold_patch")
            mlflow.log_metrics(
                {
                    "num_turns": agent_result["num_turns"],
                    "num_tool_calls": len(agent_result["tool_calls"]),
                }
            )
            if agent_result["cost_usd"] is not None:
                mlflow.log_metric("cost_usd", agent_result["cost_usd"])

            if not agent_diff.strip():
                latency = time.perf_counter() - start
                print("    Agent made no file changes.")
                mlflow.log_metrics({"latency_s": round(latency, 2), "resolved": 0})
                mlflow.set_tag("status", "no_changes")
                harness.cleanup_container(container_id)
                return _result(
                    iid,
                    repo,
                    config_name,
                    latency,
                    "no_changes",
                    cost_usd=agent_result["cost_usd"] or 0.0,
                    agent_patch="",
                    test_output="Agent made no file changes",
                )

            print(f"    Agent edited files ({len(agent_diff)} bytes diff)")

            print(f"    Running {len(f2p_tests)} FAIL_TO_PASS tests ...")
            f2p_passed, f2p_failed, f2p_output = harness.run_tests(container_id, f2p_tests, test_files)
            print(f"    FAIL_TO_PASS: {f2p_passed}/{len(f2p_tests)} passed")

            print(f"    Running {len(p2p_tests)} PASS_TO_PASS tests ...")
            p2p_passed, p2p_failed, p2p_output = harness.run_tests(container_id, p2p_tests, test_files)
            # -k matches by substring, so the run can cover a superset of the
            # listed tests -- report raw counts, and score on failed == 0.
            print(f"    PASS_TO_PASS: {p2p_passed} passed, {p2p_failed} failed")

            # -- Step E: Score -------------------------------------------------
            resolved = f2p_passed == len(f2p_tests) and f2p_failed == 0 and p2p_failed == 0
            status = "resolved" if resolved else "applied"
            latency = time.perf_counter() - start

            print(f"    Result: {status.upper()}  (latency={latency:.1f}s)")

            test_output = f"=== FAIL_TO_PASS ===\n{f2p_output}\n=== PASS_TO_PASS ===\n{p2p_output}"
            save_text_artifact(test_output, "test_output")
            mlflow.log_metrics(
                {
                    "latency_s": round(latency, 2),
                    "resolved": int(resolved),
                    "f2p_passed": f2p_passed,
                    "f2p_total": len(f2p_tests),
                    "p2p_passed": p2p_passed,
                    "p2p_total": len(p2p_tests),
                }
            )
            mlflow.set_tag("status", status)

            harness.cleanup_container(container_id)
            return _result(
                iid,
                repo,
                config_name,
                latency,
                status,
                resolved=resolved,
                f2p_passed=f2p_passed,
                f2p_total=len(f2p_tests),
                p2p_passed=p2p_passed,
                p2p_total=len(p2p_tests),
                cost_usd=agent_result["cost_usd"] or 0.0,
                agent_patch=agent_diff,
                test_output=test_output,
            )

        except Exception as e:
            latency = time.perf_counter() - start
            print(f"    Unexpected error: {e}")
            mlflow.set_tag("status", "error")
            harness.cleanup_container(container_id)
            return _error_result(iid, repo, config_name, latency, str(e))

    return _error_result(iid, repo, config_name, time.perf_counter() - start, "run did not complete")


def _result(iid: str, repo: str, config: str, latency: float, status: str, **kwargs) -> dict:
    return {
        "instance_id": iid,
        "repo": repo,
        "config": config,
        "latency_s": round(latency, 2),
        "status": status,
        "resolved": kwargs.get("resolved", False),
        "f2p_passed": kwargs.get("f2p_passed", 0),
        "f2p_total": kwargs.get("f2p_total", 0),
        "p2p_passed": kwargs.get("p2p_passed", 0),
        "p2p_total": kwargs.get("p2p_total", 0),
        "cost_usd": kwargs.get("cost_usd", 0.0),
        "agent_patch": kwargs.get("agent_patch", ""),
        "test_output": kwargs.get("test_output", ""),
    }


def _error_result(iid: str, repo: str, config: str, latency: float, error: str) -> dict:
    return _result(iid, repo, config, latency, "error", test_output=error)


async def run_config(name: str, effort: EffortLevel, instances: list[dict]) -> list[dict]:
    """Run all instances for one configuration."""
    print(f"\n{'=' * 60}")
    print(f"Config: {name}  (model={agent_mod.MODEL}, effort={effort})")
    print("=" * 60)

    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{name}", nested=True):
        mlflow.log_params(
            {
                "model": agent_mod.MODEL,
                "effort": effort,
                "sample_size": len(instances),
            }
        )

        for i, inst in enumerate(instances):
            print(f"\n  [{i + 1}/{len(instances)}] Instance: {inst['instance_id']}")
            results.append(await run_instance(effort, inst, name))

        df = pd.DataFrame(results)
        resolution_rate = float(df["resolved"].mean())
        mlflow.log_metrics(
            {
                "resolution_rate": round(resolution_rate, 3),
                "avg_latency_s": round(float(df["latency_s"].mean()), 2),
                "total_cost_usd": round(float(df["cost_usd"].sum()), 4),
            }
        )

        csv_path = f"/tmp/swe_bench_{name}.csv"
        df.drop(columns=["agent_patch", "test_output"], errors="ignore").to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print results and leaderboard comparison."""
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    df = pd.DataFrame(all_results)
    summary = df.groupby("config").agg(
        resolution_rate=("resolved", "mean"),
        avg_latency=("latency_s", "mean"),
        total_cost=("cost_usd", "sum"),
        resolved_count=("resolved", "sum"),
        total=("resolved", "count"),
    )
    for config, row in summary.iterrows():
        print(
            f"  {config:<12} {row['resolution_rate']:.1%} resolved  "
            f"({int(row['resolved_count'])}/{int(row['total'])})  "
            f"avg_latency={row['avg_latency']:.1f}s  cost=${row['total_cost']:.2f}"
        )

    our_rate = df["resolved"].mean() * 100

    print("\n  SWE-Bench Verified Leaderboard (for context):")
    print("  " + "-" * 50)
    leaderboard = [
        ("Claude 3.5 Sonnet", 49.0),
        ("GPT-4o", 33.2),
        ("DeepSeek-V2.5", 27.0),
        (f"{agent_mod.MODEL} (this run)", our_rate),
    ]
    for name, rate in sorted(leaderboard, key=lambda x: -x[1]):
        marker = " <--" if "this run" in name else ""
        print(f"    {name:<35} {rate:5.1f}%{marker}")

    n = len(df)
    if n < 10:
        print(
            f"\n  Note: With only {n} instance(s), results are not "
            "statistically meaningful.\n  Increase SAMPLE_SIZE for "
            "reliable comparison."
        )


# -- Main ----------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("L2-M2.2.3 -- SWE-Bench Evaluation Pipeline")
    print("=" * 60)

    print("\nStep 1: Ensuring evaluation base image ...")
    harness.ensure_image_built()

    print("\nStep 2: Loading SWE-Bench Verified dataset ...")
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = select_instances(ds, SAMPLE_REPO, SAMPLE_SIZE)
    print(f"  Loaded {len(ds)} instances, selected {len(instances)} from {SAMPLE_REPO}")

    for inst in instances:
        print(f"    {inst['instance_id']}")

    print(f"\nStep 3: Agent: Claude Agent SDK / {agent_mod.MODEL}")
    print("  Tracing is hand-built (@mlflow.trace + spans) -- the SDK has no autolog.")

    print("\nStep 4: Running evaluation ...")
    configs: list[tuple[str, EffortLevel]] = [("low_effort", "low"), ("high_effort", "high")]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="swe_bench_eval"):
        mlflow.log_params(
            {
                "dataset": "SWE-bench_Verified",
                "sample_size": len(instances),
                "repo_filter": SAMPLE_REPO,
                "container_runtime": harness.CONTAINER_RUNTIME,
                "model": agent_mod.MODEL,
                "max_budget_usd_per_instance": agent_mod.MAX_BUDGET_USD,
            }
        )
        mlflow.set_tag("task", "swe_bench_evaluation")
        mlflow.set_tag("framework", "claude_agent_sdk")

        for name, effort in configs:
            all_results.extend(await run_config(name, effort, instances))

        combined_csv = "/tmp/swe_bench_combined.csv"
        pd.DataFrame(all_results).drop(columns=["agent_patch", "test_output"], errors="ignore").to_csv(
            combined_csv, index=False
        )
        mlflow.log_artifact(combined_csv)

    print_summary(all_results)
    print("\n" + "=" * 60)
    print("Done. View results at http://127.0.0.1:5555")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L2/M2_agent_evaluation/2_offline/3_swe_bench")
    anyio.run(main)
