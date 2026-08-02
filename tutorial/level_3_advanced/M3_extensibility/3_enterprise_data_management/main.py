"""L3-M3.3 -- Enterprise Patterns and Data Management.

Combines enterprise MLflow patterns (multi-team organization,
model governance, cost tracking, access control) with production
data management (dataset versioning, lineage, quality checks,
drift detection).
"""

import hashlib
import json
from collections import Counter
from typing import Any

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "L3/M3_extensibility/3_enterprise_data_management"


# -- Part 1: Multi-Team Experiment Organization --------------------------------


def part1_team_organization(client: MlflowClient) -> None:
    """Create hierarchical experiments for multiple teams."""
    print("=" * 60)
    print("Part 1: Multi-Team Experiment Organization")
    print("=" * 60)

    teams = {
        "data-science": {
            "projects": ["churn-prediction", "recommendation"],
            "owner": "alice@co.com",
        },
        "nlp-team": {"projects": ["chatbot-v2", "summarization"], "owner": "bob@co.com"},
        "ml-platform": {"projects": ["model-monitoring", "feature-store"], "owner": "carol@co.com"},
    }

    for team, info in teams.items():
        for project in info["projects"]:
            exp_name = f"enterprise/{team}/{project}"
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                exp_id = client.create_experiment(
                    exp_name,
                    tags={"team": team, "owner": info["owner"], "environment": "development"},
                )
                print(f"  Created: {exp_name} (id={exp_id})")
            else:
                print(f"  Exists:  {exp_name}")

    print("\n  Searching experiments by team...")
    for team in teams:
        exps = client.search_experiments(filter_string=f"tags.team = '{team}'")
        print(f"    {team}: {[e.name for e in exps]}")
    print()


# -- Part 2: Model Governance --------------------------------------------------


class ModelGovernance:
    """Model lifecycle with approval gates: development -> staging -> champion."""

    STAGES = ("development", "staging", "champion")

    def __init__(self, client: MlflowClient) -> None:
        self.client = client

    def register_and_promote(self, model_uri: str, name: str, owner: str) -> None:
        """Full governance workflow: register, promote through stages."""
        try:
            self.client.delete_registered_model(name)
        except Exception:
            pass

        result = mlflow.register_model(model_uri, name)
        self.client.set_model_version_tag(name, result.version, "governance.owner", owner)
        self.client.set_model_version_tag(name, result.version, "governance.stage", "development")
        print(f"  Registered '{name}' v{result.version} (owner: {owner})")

        checks_pass = {"unit_tests": True, "integration_tests": True, "code_review": True}
        self._promote(name, result.version, "development", "staging", checks_pass)

        checks_fail = {"load_test": True, "security_scan": False}
        self._promote(name, result.version, "staging", "champion", checks_fail)

        checks_retry = {"load_test": True, "security_scan": True, "stakeholder_approval": True}
        self._promote(name, result.version, "staging", "champion", checks_retry)

    def _promote(self, name: str, version: str, from_stage: str, to_stage: str, checks: dict[str, bool]) -> None:
        failed = [c for c, ok in checks.items() if not ok]
        if failed:
            print(f"  REJECTED {from_stage}->{to_stage}: failed {failed}")
            return
        self.client.set_model_version_tag(name, version, "governance.stage", to_stage)
        if to_stage == "champion":
            self.client.set_registered_model_alias(name, "champion", version)
        print(f"  PROMOTED {from_stage} -> {to_stage}")


def part2_governance(client: MlflowClient) -> str:
    """Demonstrate governance workflow."""
    print("=" * 60)
    print("Part 2: Model Governance Workflow")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    governance = ModelGovernance(client)

    with mlflow.start_run(run_name="governance-demo") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_framework", "pyfunc")

        class SimpleModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                return ["prediction"] * len(model_input)

        mlflow.pyfunc.log_model(name="model", python_model=SimpleModel(), input_example={"text": "hello"})
        governance.register_and_promote(f"runs:/{run_id}/model", "enterprise-demo-model", "alice@co.com")
    print()
    return run_id


# -- Part 3: Cost Tracking ----------------------------------------------------


def part3_cost_tracking() -> None:
    """Track LLM token usage and estimate costs."""
    print("=" * 60)
    print("Part 3: LLM Cost Tracking")
    print("=" * 60)

    pricing = {
        "google/gemma-4-e4b": {"input": 0.10, "output": 0.30},
        "google/gemma-4-26b-a4b": {"input": 0.50, "output": 1.50},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    calls = [
        ("google/gemma-4-e4b", 512, 128, "classification"),
        ("google/gemma-4-e4b", 1024, 256, "summarization"),
        ("google/gemma-4-26b-a4b", 2048, 512, "agent-reasoning"),
        ("google/gemma-4-26b-a4b", 4096, 1024, "evaluation-judge"),
        ("gpt-4o", 3000, 800, "complex-analysis"),
    ]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="cost-tracking"):
        records = []
        for model, inp, out, op in calls:
            prices = pricing.get(model, {"input": 0, "output": 0})
            cost = inp * prices["input"] / 1e6 + out * prices["output"] / 1e6
            records.append(
                {
                    "model": model,
                    "operation": op,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cost_usd": round(cost, 6),
                }
            )
            print(f"  {op:<20s} | {model:<25s} | ${cost:.6f}")

        df = pd.DataFrame(records)
        summary = df.groupby("model").agg(calls=("model", "count"), total_cost=("cost_usd", "sum")).reset_index()
        print(f"\n  Total estimated cost: ${df['cost_usd'].sum():.6f}")
        mlflow.log_metric("cost/total_usd", df["cost_usd"].sum())
        mlflow.log_table(summary, artifact_file="cost_report.json")
    print()


# -- Part 4: Dataset Versioning ------------------------------------------------


class DatasetManager:
    """Versioned evaluation datasets with hash-based identity."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.versions: list[dict[str, Any]] = []

    def add_version(self, df: pd.DataFrame, description: str) -> dict[str, Any]:
        digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:12]
        meta = {
            "name": self.name,
            "version": len(self.versions) + 1,
            "digest": digest,
            "description": description,
            "num_rows": len(df),
            "columns": list(df.columns),
        }
        self.versions.append(meta)
        return meta

    def log_to_mlflow(self, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        dataset = from_pandas(df, source=f"{self.name}_v{meta['version']}")
        mlflow.log_input(dataset, context="evaluation")
        mlflow.set_tag("dataset.version", str(meta["version"]))
        mlflow.set_tag("dataset.digest", meta["digest"])


def part4_dataset_versioning() -> DatasetManager:
    """Dataset versioning with three evolving versions."""
    print("=" * 60)
    print("Part 4: Dataset Versioning and Lineage")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    manager = DatasetManager("agent_eval_dataset")

    v1 = pd.DataFrame(
        {
            "question": [
                "What is the capital of France?",
                "Summarize photosynthesis.",
                "What is 25*4?",
            ],
            "expected_answer": ["Paris", "Plants convert sunlight to energy.", "100"],
            "category": ["factual", "summarization", "math"],
        }
    )
    v1_meta = manager.add_version(v1, "Initial 3-question set")
    with mlflow.start_run(run_name="dataset_v1"):
        manager.log_to_mlflow(v1, v1_meta)
    print(f"  v1: {v1_meta['num_rows']} rows, digest={v1_meta['digest']}")

    v2 = pd.concat(
        [
            v1,
            pd.DataFrame(
                {
                    "question": ["Is the sky blue?", "Translate 'hello' to Spanish."],
                    "expected_answer": ["Yes", "Hola"],
                    "category": ["yes_no", "translation"],
                }
            ),
        ],
        ignore_index=True,
    )
    v2_meta = manager.add_version(v2, "Added edge cases")
    with mlflow.start_run(run_name="dataset_v2"):
        manager.log_to_mlflow(v2, v2_meta)
    print(f"  v2: {v2_meta['num_rows']} rows, digest={v2_meta['digest']}")

    v3 = v2.copy()
    v3.loc[0, "expected_answer"] = "The capital of France is Paris."
    v3_meta = manager.add_version(v3, "Refined expected answers")
    with mlflow.start_run(run_name="dataset_v3"):
        manager.log_to_mlflow(v3, v3_meta)
    print(f"  v3: {v3_meta['num_rows']} rows, digest={v3_meta['digest']}")

    print("\n  Lineage: linking eval runs to dataset versions...")
    for ver_idx, label in [(0, "eval_on_v1"), (2, "eval_on_v3")]:
        meta = manager.versions[ver_idx]
        with mlflow.start_run(run_name=label):
            mlflow.set_tag("lineage.dataset_version", str(meta["version"]))
            mlflow.set_tag("lineage.dataset_digest", meta["digest"])
            mlflow.log_metric("mock_accuracy", 0.85 if ver_idx == 0 else 0.91)
            print(f"    {label}: linked to v{meta['version']}")
    print()
    return manager


# -- Part 5: Data Quality and Drift Detection ----------------------------------


def part5_quality_and_drift() -> None:
    """Data quality checks and drift detection."""
    print("=" * 60)
    print("Part 5: Data Quality Checks and Drift Detection")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    good = pd.DataFrame(
        {
            "question": ["What is 2+2?", "Name a primary color?"],
            "expected_answer": ["4", "Red"],
            "category": ["math", "factual"],
        }
    )
    bad = pd.DataFrame(
        {
            "question": ["What is 2+2?", "No question mark", "What is 2+2?"],
            "expected_answer": ["4", "", "4"],
            "category": ["math", "factual", "math"],
        }
    )

    print("\n  Quality checks:")
    for label, df in [("good", good), ("bad", bad)]:
        missing = int(df.isnull().sum().sum())
        dupes = int(df.iloc[:, 0].duplicated().sum()) if len(df.columns) > 0 else 0
        answers = df["expected_answer"] if "expected_answer" in df.columns else pd.Series(dtype=str)
        empty_answers = int((answers.str.len() == 0).sum())
        passed = missing == 0 and dupes == 0 and empty_answers == 0
        with mlflow.start_run(run_name=f"quality_{label}"):
            mlflow.set_tag("quality.all_passed", str(passed))
            mlflow.log_metrics({"missing_cells": missing, "duplicates": dupes, "empty_answers": empty_answers})
        print(f"    {label}: {'PASS' if passed else 'FAIL'} (missing={missing}, dupes={dupes}, empty={empty_answers})")

    baseline = pd.DataFrame({"category": ["science", "biology", "math", "science"]})
    current = pd.DataFrame({"category": ["science", "physics", "math", "ai", "biology"]})

    base_counts = Counter(baseline["category"])
    curr_counts = Counter(current["category"])
    all_keys = set(base_counts) | set(curr_counts)
    bt, ct = max(sum(base_counts.values()), 1), max(sum(curr_counts.values()), 1)
    shift = sum(abs(base_counts.get(k, 0) / bt - curr_counts.get(k, 0) / ct) for k in all_keys)
    new_cats = sorted(set(curr_counts) - set(base_counts))

    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metric("drift.category.shift", round(shift, 4))
        mlflow.set_tag("drift.new_categories", json.dumps(new_cats))

    print("\n  Drift detection:")
    print(f"    Category shift: {shift:.4f}")
    print(f"    New categories: {new_cats}")
    print()


# -- Main ----------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("L3-M3.3 -- Enterprise Patterns and Data Management")
    print("=" * 60)

    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri="http://127.0.0.1:5555")

    part1_team_organization(client)
    part2_governance(client)
    part3_cost_tracking()
    part4_dataset_versioning()
    part5_quality_and_drift()

    print("=" * 60)
    print("Done! View results at http://127.0.0.1:5555")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("Also check: enterprise/* experiments for team organization")
    print("=" * 60)


if __name__ == "__main__":
    main()
