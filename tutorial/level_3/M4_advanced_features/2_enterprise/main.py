"""
L3-4.2 — Enterprise MLflow Patterns

Demonstrates production enterprise patterns for MLflow:
- Multi-team experiment organization with hierarchical naming
- Model governance workflow (development -> staging -> champion)
- Audit logging for compliance and traceability
- LLM cost tracking and reporting
- Access control patterns via namespace conventions and tags
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "L3/M4_advanced_features/2_enterprise"


# ── Part 2: Model Governance ─────────────────────────────────────────────── #


class ModelGovernance:
    """Manages model lifecycle with approval gates and governance tags.

    Stages: development -> staging -> champion
    Each transition requires passing validation checks before promotion.
    """

    STAGES = ("development", "staging", "champion")

    def __init__(self, client: MlflowClient) -> None:
        self.client = client
        self.audit = AuditLogger(client)

    def register_model(
        self, model_uri: str, name: str, owner: str
    ) -> Any:
        """Register a new model with governance metadata."""
        result = mlflow.register_model(model_uri, name)
        self.client.set_model_version_tag(
            name, result.version, "governance.owner", owner
        )
        self.client.set_model_version_tag(
            name, result.version, "governance.stage", "development"
        )
        self.client.set_model_version_tag(
            name, result.version, "governance.created_at",
            datetime.now(timezone.utc).isoformat(),
        )
        self.audit.log(
            action="MODEL_REGISTERED",
            model_name=name,
            version=result.version,
            actor=owner,
            details=f"Registered from {model_uri}",
        )
        print(f"    Registered model '{name}' v{result.version} (owner: {owner})")
        return result

    def promote(
        self,
        name: str,
        version: str,
        actor: str,
        checks: dict[str, bool],
    ) -> bool:
        """Promote a model version to the next stage if all checks pass."""
        mv = self.client.get_model_version(name, version)
        tags = mv.tags if isinstance(mv.tags, dict) else {}
        current_stage = tags.get("governance.stage", "development")
        idx = self.STAGES.index(current_stage) if current_stage in self.STAGES else 0
        if idx >= len(self.STAGES) - 1:
            print(f"    Model '{name}' v{version} is already at '{current_stage}'")
            return False

        # All checks must pass
        failed = [c for c, passed in checks.items() if not passed]
        if failed:
            self.audit.log(
                action="PROMOTION_REJECTED",
                model_name=name,
                version=version,
                actor=actor,
                details=f"Failed checks: {', '.join(failed)}",
            )
            print(f"    Promotion rejected for '{name}' v{version}: failed {failed}")
            return False

        next_stage = self.STAGES[idx + 1]
        self.client.set_model_version_tag(
            name, version, "governance.stage", next_stage
        )
        self.client.set_model_version_tag(
            name, version, f"governance.{next_stage}_approved_by", actor
        )
        self.client.set_model_version_tag(
            name, version, f"governance.{next_stage}_approved_at",
            datetime.now(timezone.utc).isoformat(),
        )

        # Set alias for champion
        if next_stage == "champion":
            self.client.set_registered_model_alias(name, "champion", version)

        self.audit.log(
            action="MODEL_PROMOTED",
            model_name=name,
            version=version,
            actor=actor,
            details=f"{current_stage} -> {next_stage}. Checks: {checks}",
        )
        print(f"    Promoted '{name}' v{version}: {current_stage} -> {next_stage}")
        return True


# ── Part 3: Audit Logging ────────────────────────────────────────────────── #


class AuditLogger:
    """Records all governance and operational events for compliance.

    Each entry captures who, what, when, and contextual details.
    The full trail is saved as a JSON artifact on the current MLflow run.
    """

    def __init__(self, client: MlflowClient) -> None:
        self.client = client
        self.entries: list[dict[str, Any]] = []

    def log(
        self,
        action: str,
        actor: str,
        model_name: str = "",
        version: str = "",
        details: str = "",
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "actor": actor,
            "model_name": model_name,
            "version": version,
            "details": details,
        }
        self.entries.append(entry)
        print(f"    [AUDIT] {action} by {actor}: {details[:80]}")

    def save_artifact(self, run_id: str, filename: str = "audit_trail.json") -> None:
        """Write the audit trail as a JSON artifact on the given run."""
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(self.entries, f, indent=2)
            tmp_path = f.name
        self.client.log_artifact(run_id, tmp_path, artifact_path="governance")
        os.unlink(tmp_path)
        print(f"    Saved {len(self.entries)} audit entries to '{filename}'")


# ── Part 4: Cost Tracking ────────────────────────────────────────────────── #


class CostTracker:
    """Tracks LLM token usage and estimates costs per model.

    Prices are illustrative (local Ollama models have zero real cost,
    but this demonstrates the pattern for cloud LLM deployments).
    """

    # Cost per 1M tokens (USD) — illustrative rates
    PRICING: dict[str, dict[str, float]] = {
        "gemma4:e2b": {"input": 0.10, "output": 0.30},
        "gemma4:26b": {"input": 0.50, "output": 1.50},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
    }

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "",
    ) -> dict[str, float]:
        prices = self.PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = input_tokens * prices["input"] / 1_000_000
        output_cost = output_tokens * prices["output"] / 1_000_000
        total_cost = input_cost + output_cost

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }
        self.records.append(entry)
        return entry

    def log_to_mlflow(self, step: int | None = None) -> None:
        """Log aggregate cost metrics to the active MLflow run."""
        if not self.records:
            return
        total_input = sum(r["input_tokens"] for r in self.records)
        total_output = sum(r["output_tokens"] for r in self.records)
        total_cost = sum(r["total_cost_usd"] for r in self.records)
        metrics = {
            "cost/total_input_tokens": total_input,
            "cost/total_output_tokens": total_output,
            "cost/total_tokens": total_input + total_output,
            "cost/estimated_cost_usd": total_cost,
            "cost/num_calls": len(self.records),
        }
        mlflow.log_metrics(metrics, step=step)

    def report(self) -> pd.DataFrame:
        """Generate a per-model cost summary."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        summary = (
            df.groupby("model")
            .agg(
                calls=("model", "count"),
                total_input_tokens=("input_tokens", "sum"),
                total_output_tokens=("output_tokens", "sum"),
                total_cost_usd=("total_cost_usd", "sum"),
            )
            .reset_index()
        )
        return summary


# ── Part 1: Multi-Team Experiment Organization ───────────────────────────── #


def part1_team_organization(client: MlflowClient) -> None:
    """Create hierarchical experiments for multiple teams."""
    print("=" * 60)
    print("Part 1: Multi-Team Experiment Organization")
    print("=" * 60)

    teams = {
        "data-science": {
            "projects": ["churn-prediction", "recommendation-engine"],
            "owner": "alice@company.com",
        },
        "nlp-team": {
            "projects": ["chatbot-v2", "summarization"],
            "owner": "bob@company.com",
        },
        "ml-platform": {
            "projects": ["model-monitoring", "feature-store"],
            "owner": "carol@company.com",
        },
    }

    for team, info in teams.items():
        for project in info["projects"]:
            exp_name = f"enterprise/{team}/{project}"
            experiment = client.get_experiment_by_name(exp_name)
            if experiment is None:
                exp_id = client.create_experiment(
                    exp_name,
                    tags={
                        "team": team,
                        "owner": info["owner"],
                        "environment": "development",
                        "created_by": "enterprise_setup",
                    },
                )
                print(f"  Created experiment: {exp_name} (id={exp_id})")
            else:
                exp_id = experiment.experiment_id
                print(f"  Experiment exists:  {exp_name} (id={exp_id})")

            # Create a sample run in each experiment
            with mlflow.start_run(
                experiment_id=exp_id,
                run_name=f"setup-{team}-{project}",
            ):
                mlflow.set_tag("team", team)
                mlflow.set_tag("project", project)
                mlflow.set_tag("owner", info["owner"])
                mlflow.log_param("model_type", "baseline")
                mlflow.log_metric("placeholder_metric", 0.0)

    # Demonstrate searching by team
    print("\n  Searching experiments by team tag...")
    for team in teams:
        experiments = client.search_experiments(
            filter_string=f"tags.team = '{team}'"
        )
        names = [e.name for e in experiments]
        print(f"    Team '{team}': {names}")

    print()


# ── Part 2 runner ─────────────────────────────────────────────────────────── #


def part2_model_governance(client: MlflowClient) -> str:
    """Demonstrate the full governance workflow and return the run_id."""
    print("=" * 60)
    print("Part 2: Model Governance Workflow")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    governance = ModelGovernance(client)

    with mlflow.start_run(run_name="governance-demo") as run:
        run_id = run.info.run_id

        # Log a simple sklearn-style model (pyfunc wrapper)
        mlflow.log_param("model_framework", "pyfunc")
        mlflow.log_metric("accuracy", 0.92)

        # Create a minimal pyfunc model for registration
        class SimpleModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                return ["prediction"] * len(model_input)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=SimpleModel(),
            input_example={"text": "hello"},
        )
        model_uri = f"runs:/{run_id}/model"

        # Register
        model_name = "enterprise-demo-model"
        try:
            client.delete_registered_model(model_name)
        except Exception:
            pass
        mv = governance.register_model(model_uri, model_name, owner="alice@company.com")

        # Promote development -> staging (all checks pass)
        print("\n  Attempting promotion: development -> staging")
        governance.promote(
            model_name,
            mv.version,
            actor="bob@company.com",
            checks={"unit_tests": True, "integration_tests": True, "code_review": True},
        )

        # Promote staging -> champion (one check fails first)
        print("\n  Attempting promotion with a failing check...")
        governance.promote(
            model_name,
            mv.version,
            actor="carol@company.com",
            checks={"load_test": True, "security_scan": False, "stakeholder_approval": True},
        )

        # Fix and retry
        print("\n  Retrying after fixing security scan...")
        governance.promote(
            model_name,
            mv.version,
            actor="carol@company.com",
            checks={"load_test": True, "security_scan": True, "stakeholder_approval": True},
        )

        # Save audit trail
        governance.audit.save_artifact(run_id)

    print()
    return run_id


# ── Part 4 runner ─────────────────────────────────────────────────────────── #


def part4_cost_tracking() -> None:
    """Simulate LLM calls and track costs in MLflow."""
    print("=" * 60)
    print("Part 4: LLM Cost Tracking")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)
    tracker = CostTracker()

    with mlflow.start_run(run_name="cost-tracking-demo"):
        # Simulate a series of LLM calls across models
        simulated_calls = [
            ("gemma4:e2b", 512, 128, "classification"),
            ("gemma4:e2b", 1024, 256, "summarization"),
            ("gemma4:26b", 2048, 512, "agent-reasoning"),
            ("gemma4:26b", 4096, 1024, "evaluation-judge"),
            ("gpt-4o", 3000, 800, "complex-analysis"),
            ("claude-sonnet", 2500, 600, "code-generation"),
            ("gemma4:e2b", 768, 192, "embedding-query"),
            ("gemma4:26b", 1500, 400, "rag-response"),
        ]

        for model, inp, out, op in simulated_calls:
            entry = tracker.record(model, inp, out, operation=op)
            print(
                f"  {op:<20s} | {model:<15s} | "
                f"{inp:>5d} in / {out:>5d} out | "
                f"${entry['total_cost_usd']:.6f}"
            )

        tracker.log_to_mlflow()

        # Generate and display report
        report = tracker.report()
        print("\n  Cost Summary by Model:")
        print(report.to_string(index=False))

        # Log the report as an artifact
        mlflow.log_table(report, artifact_file="cost_report.json")
        mlflow.log_metric("cost/total_usd", report["total_cost_usd"].sum())

    print()


# ── Part 5: Access Control Patterns ──────────────────────────────────────── #


def part5_access_control(client: MlflowClient) -> None:
    """Demonstrate namespace conventions and tag-based team filtering."""
    print("=" * 60)
    print("Part 5: Access Control Patterns")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Pattern 1: namespace-based isolation
    print("\n  Pattern 1: Namespace Conventions")
    namespaces = {
        "prod/models/fraud-detector": {"access": "restricted", "team": "security"},
        "staging/models/fraud-detector": {"access": "team", "team": "security"},
        "dev/models/fraud-detector": {"access": "open", "team": "security"},
    }
    for ns, tags in namespaces.items():
        exp = client.get_experiment_by_name(ns)
        if exp is None:
            exp_id = client.create_experiment(ns, tags=tags)
        else:
            exp_id = exp.experiment_id
        print(f"    {ns:<45s} access={tags['access']}")

    # Pattern 2: tag-based visibility filtering
    print("\n  Pattern 2: Tag-Based Filtering")
    with mlflow.start_run(run_name="access-control-demo"):
        mlflow.set_tag("team", "nlp-team")
        mlflow.set_tag("visibility", "team-only")
        mlflow.set_tag("data_classification", "internal")
        mlflow.set_tag("pii_involved", "false")
        mlflow.log_param("model_type", "text-classifier")
        mlflow.log_metric("f1_score", 0.89)

    # Search by visibility
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        ],
        filter_string="tags.visibility = 'team-only'",
    )
    print(f"    Found {len(runs)} run(s) with visibility='team-only'")

    # Pattern 3: environment separation
    print("\n  Pattern 3: Environment Separation Convention")
    envs = ["dev", "staging", "prod"]
    for env in envs:
        convention = f"<team>/<project>/{env}"
        access = {"dev": "all engineers", "staging": "team leads", "prod": "SRE + approval"}
        print(f"    {convention:<35s} -> {access[env]}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────── #


def main() -> None:
    print("=" * 60)
    print("  L3-4.2 — Enterprise MLflow Patterns")
    print("=" * 60)

    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    part1_team_organization(client)
    part2_model_governance(client)
    part4_cost_tracking()
    part5_access_control(client)

    print("=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5000")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print("  Also check: enterprise/* experiments for team organization")
    print("=" * 60)


if __name__ == "__main__":
    main()
