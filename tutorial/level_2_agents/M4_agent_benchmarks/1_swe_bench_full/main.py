"""L2-M4.1 -- Full SWE-Bench Evaluation Pipeline.

Builds a coding agent, runs it against SWE-Bench Verified instances,
applies patches inside Docker/Podman containers, runs the repo test
suite, and scores results using the same methodology as the SWE-Bench
leaderboard. All results are tracked in MLflow.
"""

import json
import tempfile
import time

import datasets
import mlflow
import pandas as pd

import agent as agent_mod
import harness

SAMPLE_SIZE = 2
SAMPLE_REPO = "sympy/sympy"


# -- Helpers -------------------------------------------------------------------

def select_instances(ds, repo: str, count: int) -> list[dict]:
    """Filter dataset to instances from a specific repo."""
    instances = [ds[i] for i in range(len(ds)) if ds[i]["repo"] == repo]
    return instances[:count]


def save_text_artifact(text: str, filename: str) -> None:
    """Log a text string as an MLflow artifact."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=filename + "_", delete=False
    ) as f:
        f.write(text)
        mlflow.log_artifact(f.name)


def run_instance(agent, instance: dict, config_name: str) -> dict:
    """Run the full evaluation pipeline for one SWE-Bench instance."""
    iid = instance["instance_id"]
    repo = instance["repo"]
    f2p_tests = json.loads(instance["FAIL_TO_PASS"])
    p2p_tests = json.loads(instance["PASS_TO_PASS"])[:harness.MAX_P2P_TESTS]

    with mlflow.start_run(run_name=f"{config_name}_{iid}", nested=True):
        mlflow.log_params({
            "instance_id": iid,
            "repo": repo,
            "base_commit": instance["base_commit"][:8],
            "config": config_name,
            "f2p_test_count": len(f2p_tests),
            "p2p_test_count": len(p2p_tests),
        })
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
            base_passed, base_failed, base_out = harness.run_tests(
                container_id, f2p_tests, repo
            )
            print(f"    Baseline: {base_passed} passed, {base_failed} failed (expect failures)")

            # -- Step C: Run agent to generate a patch -------------------------
            print("    Running agent ...")
            agent_mod.set_active_container(container_id)
            try:
                result = agent.invoke({
                    "messages": [{"role": "user", "content": agent_mod.build_prompt(instance)}]
                })
                agent_output = result["messages"][-1].content
            except Exception as e:
                latency = time.perf_counter() - start
                print(f"    Agent failed: {e}")
                mlflow.set_tag("status", "error")
                harness.cleanup_container(container_id)
                return _error_result(iid, repo, config_name, latency, f"agent error: {e}")

            agent_patch = agent_mod.extract_patch(agent_output)
            save_text_artifact(agent_patch or "(no patch)", "agent_patch")
            save_text_artifact(instance.get("patch", ""), "gold_patch")

            if not agent_patch:
                latency = time.perf_counter() - start
                print("    No valid patch extracted from agent output.")
                mlflow.log_metrics({"latency_s": round(latency, 2), "resolved": 0})
                mlflow.set_tag("status", "patch_failed")
                harness.cleanup_container(container_id)
                return _result(
                    iid, repo, config_name, latency, "patch_failed",
                    agent_patch="", test_output="No patch extracted",
                )

            print(f"    Agent generated patch ({len(agent_patch)} bytes)")

            # -- Step D: Apply agent patch and run tests -----------------------
            print("    Applying agent patch ...")
            patch_ok = harness.apply_patch(container_id, agent_patch)
            if not patch_ok:
                latency = time.perf_counter() - start
                print("    Patch application failed (git apply rejected)")
                mlflow.log_metrics({"latency_s": round(latency, 2), "resolved": 0})
                mlflow.set_tag("status", "patch_failed")
                save_text_artifact("git apply failed", "test_output")
                harness.cleanup_container(container_id)
                return _result(
                    iid, repo, config_name, latency, "patch_failed",
                    agent_patch=agent_patch, test_output="git apply failed",
                )

            print(f"    Running {len(f2p_tests)} FAIL_TO_PASS tests ...")
            f2p_passed, f2p_failed, f2p_output = harness.run_tests(
                container_id, f2p_tests, repo
            )
            print(f"    FAIL_TO_PASS: {f2p_passed}/{len(f2p_tests)} passed")

            print(f"    Running {len(p2p_tests)} PASS_TO_PASS tests ...")
            p2p_passed, p2p_failed, p2p_output = harness.run_tests(
                container_id, p2p_tests, repo
            )
            print(f"    PASS_TO_PASS: {p2p_passed}/{len(p2p_tests)} passed")

            # -- Step E: Score -------------------------------------------------
            resolved = (
                f2p_passed == len(f2p_tests)
                and f2p_failed == 0
                and p2p_failed == 0
            )
            status = "resolved" if resolved else "applied"
            latency = time.perf_counter() - start

            print(f"    Result: {status.upper()}  (latency={latency:.1f}s)")

            test_output = f"=== FAIL_TO_PASS ===\n{f2p_output}\n=== PASS_TO_PASS ===\n{p2p_output}"
            save_text_artifact(test_output, "test_output")
            mlflow.log_metrics({
                "latency_s": round(latency, 2),
                "resolved": int(resolved),
                "f2p_passed": f2p_passed,
                "f2p_total": len(f2p_tests),
                "p2p_passed": p2p_passed,
                "p2p_total": len(p2p_tests),
            })
            mlflow.set_tag("status", status)

            harness.cleanup_container(container_id)
            return _result(
                iid, repo, config_name, latency, status,
                resolved=resolved, f2p_passed=f2p_passed, f2p_total=len(f2p_tests),
                p2p_passed=p2p_passed, p2p_total=len(p2p_tests),
                agent_patch=agent_patch, test_output=test_output,
            )

        except Exception as e:
            latency = time.perf_counter() - start
            print(f"    Unexpected error: {e}")
            mlflow.set_tag("status", "error")
            harness.cleanup_container(container_id)
            return _error_result(iid, repo, config_name, latency, str(e))


def _result(
    iid: str, repo: str, config: str, latency: float, status: str, **kwargs
) -> dict:
    return {
        "instance_id": iid, "repo": repo, "config": config,
        "latency_s": round(latency, 2), "status": status,
        "resolved": kwargs.get("resolved", False),
        "f2p_passed": kwargs.get("f2p_passed", 0),
        "f2p_total": kwargs.get("f2p_total", 0),
        "p2p_passed": kwargs.get("p2p_passed", 0),
        "p2p_total": kwargs.get("p2p_total", 0),
        "agent_patch": kwargs.get("agent_patch", ""),
        "test_output": kwargs.get("test_output", ""),
    }


def _error_result(
    iid: str, repo: str, config: str, latency: float, error: str
) -> dict:
    return _result(iid, repo, config, latency, "error", test_output=error)


def run_config(
    name: str, temperature: float, instances: list[dict]
) -> list[dict]:
    """Run all instances for one configuration."""
    print(f"\n{'=' * 60}")
    print(f"Config: {name}  (temperature={temperature})")
    print("=" * 60)

    agent = agent_mod.build_agent(temperature)
    results: list[dict] = []

    with mlflow.start_run(run_name=f"config_{name}", nested=True):
        mlflow.log_params({
            "temperature": temperature,
            "model": "google/gemma-4-26b-a4b",
            "sample_size": len(instances),
        })

        for i, inst in enumerate(instances):
            print(f"\n  [{i + 1}/{len(instances)}] Instance: {inst['instance_id']}")
            results.append(run_instance(agent, inst, name))

        df = pd.DataFrame(results)
        resolution_rate = df["resolved"].mean()
        mlflow.log_metrics({
            "resolution_rate": round(resolution_rate, 3),
            "avg_latency_s": round(df["latency_s"].mean(), 2),
        })

        csv_path = f"/tmp/swe_bench_full_{name}.csv"
        df.drop(columns=["agent_patch", "test_output"], errors="ignore").to_csv(
            csv_path, index=False
        )
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
        resolved_count=("resolved", "sum"),
        total=("resolved", "count"),
    )
    for config, row in summary.iterrows():
        print(
            f"  {config:<12} {row['resolution_rate']:.1%} resolved  "
            f"({int(row['resolved_count'])}/{int(row['total'])})  "
            f"avg_latency={row['avg_latency']:.1f}s"
        )

    our_rate = df["resolved"].mean() * 100

    print("\n  SWE-Bench Verified Leaderboard (for context):")
    print("  " + "-" * 50)
    leaderboard = [
        ("Claude 3.5 Sonnet", 49.0),
        ("GPT-4o", 33.2),
        ("DeepSeek-V2.5", 27.0),
        (f"Local gemma-4-26b (this run)", our_rate),
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

def main() -> None:
    print("=" * 60)
    print("L2-M4.1 -- Full SWE-Bench Evaluation Pipeline")
    print("=" * 60)

    print("\nStep 1: Ensuring evaluation base image ...")
    harness.ensure_image_built()

    print("\nStep 2: Loading SWE-Bench Verified dataset ...")
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = select_instances(ds, SAMPLE_REPO, SAMPLE_SIZE)
    print(f"  Loaded {len(ds)} instances, selected {len(instances)} from {SAMPLE_REPO}")

    for inst in instances:
        print(f"    {inst['instance_id']}")

    print("\nStep 3: Enabling MLflow autolog ...")
    mlflow.langchain.autolog()

    print("\nStep 4: Running evaluation ...")
    configs = [("precise", 0.3), ("creative", 0.7)]
    all_results: list[dict] = []

    with mlflow.start_run(run_name="swe_bench_full_eval"):
        mlflow.log_params({
            "dataset": "SWE-bench_Verified",
            "sample_size": len(instances),
            "repo_filter": SAMPLE_REPO,
            "container_runtime": harness.CONTAINER_RUNTIME,
        })
        mlflow.set_tag("task", "swe_bench_full_evaluation")

        for name, temp in configs:
            all_results.extend(run_config(name, temp, instances))

        combined_csv = "/tmp/swe_bench_full_combined.csv"
        pd.DataFrame(all_results).drop(
            columns=["agent_patch", "test_output"], errors="ignore"
        ).to_csv(combined_csv, index=False)
        mlflow.log_artifact(combined_csv)

    print_summary(all_results)
    print("\n" + "=" * 60)
    print("Done. View results at http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M4_agent_benchmarks/1_swe_bench_full")
    main()
