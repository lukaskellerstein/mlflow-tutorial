"""
L3-4.4 — Advanced Data Management

Production-grade data management patterns for MLflow:
- Dataset versioning with hash-based identity
- Dataset lineage tracking (dataset → evaluation run)
- Data quality checks before evaluation
- Reusable evaluation data pipelines
- Data drift detection between dataset versions
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L3/M4_advanced_features/4_data_management")


# ── Part 1: Dataset Versioning ────────────────────────────────────────────── #


class DatasetManager:
    """Manages versioned evaluation datasets with hash-based identity."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.versions: list[dict[str, Any]] = []

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a stable hash for a DataFrame's contents."""
        content = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:12]

    def add_version(
        self, df: pd.DataFrame, description: str
    ) -> dict[str, Any]:
        """Register a new dataset version and return its metadata."""
        digest = self._compute_hash(df)
        version = len(self.versions) + 1
        meta = {
            "name": self.name,
            "version": version,
            "digest": digest,
            "description": description,
            "num_rows": len(df),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat(),
        }
        self.versions.append(meta)
        return meta

    def log_to_mlflow(self, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        """Log a versioned dataset to an MLflow run."""
        dataset = mlflow.data.from_pandas(
            df,
            source=f"{self.name}_v{meta['version']}",
            name=f"{self.name}_v{meta['version']}",
        )
        mlflow.log_input(dataset, context="evaluation")
        mlflow.set_tag("dataset.name", self.name)
        mlflow.set_tag("dataset.version", str(meta["version"]))
        mlflow.set_tag("dataset.digest", meta["digest"])
        mlflow.set_tag("dataset.num_rows", str(meta["num_rows"]))
        mlflow.log_dict(meta, f"dataset_metadata_v{meta['version']}.json")


def part1_dataset_versioning() -> DatasetManager:
    """Demonstrate dataset versioning with three evolving versions."""
    print("=" * 60)
    print("Part 1: Dataset Versioning")
    print("=" * 60)

    manager = DatasetManager("agent_eval_dataset")

    # Version 1: Initial test cases
    v1_df = pd.DataFrame({
        "question": [
            "What is the capital of France?",
            "Summarize photosynthesis in one sentence.",
            "What is 25 * 4?",
        ],
        "expected_answer": [
            "Paris",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "100",
        ],
        "category": ["factual", "summarization", "math"],
    })
    v1_meta = manager.add_version(v1_df, "Initial 3-question eval set")
    with mlflow.start_run(run_name="dataset_v1"):
        manager.log_to_mlflow(v1_df, v1_meta)
    print(f"  v1: {v1_meta['num_rows']} rows, digest={v1_meta['digest']}")

    # Version 2: Added edge cases
    v2_df = pd.concat([v1_df, pd.DataFrame({
        "question": [
            "Is the sky blue? Answer yes or no.",
            "Translate 'hello' to Spanish.",
        ],
        "expected_answer": ["Yes", "Hola"],
        "category": ["yes_no", "translation"],
    })], ignore_index=True)
    v2_meta = manager.add_version(v2_df, "Added yes/no and translation cases")
    with mlflow.start_run(run_name="dataset_v2"):
        manager.log_to_mlflow(v2_df, v2_meta)
    print(f"  v2: {v2_meta['num_rows']} rows, digest={v2_meta['digest']}")

    # Version 3: Refined expected answers
    v3_df = v2_df.copy()
    v3_df.loc[0, "expected_answer"] = "The capital of France is Paris."
    v3_df.loc[2, "expected_answer"] = "25 * 4 = 100"
    v3_meta = manager.add_version(v3_df, "Refined expected answers for clarity")
    with mlflow.start_run(run_name="dataset_v3"):
        manager.log_to_mlflow(v3_df, v3_meta)
    print(f"  v3: {v3_meta['num_rows']} rows, digest={v3_meta['digest']}")

    print(f"\n  Total versions tracked: {len(manager.versions)}")
    return manager


# ── Part 2: Dataset Lineage Tracking ──────────────────────────────────────── #


def part2_lineage_tracking(manager: DatasetManager) -> None:
    """Track which datasets fed which evaluation runs."""
    print("\n" + "=" * 60)
    print("Part 2: Dataset Lineage Tracking")
    print("=" * 60)

    # Simulate two evaluation runs on different dataset versions
    for ver_idx, run_label in [(0, "eval_on_v1"), (2, "eval_on_v3")]:
        meta = manager.versions[ver_idx]
        with mlflow.start_run(run_name=run_label):
            mlflow.set_tag("lineage.dataset_name", meta["name"])
            mlflow.set_tag("lineage.dataset_version", str(meta["version"]))
            mlflow.set_tag("lineage.dataset_digest", meta["digest"])
            mlflow.set_tag("lineage.purpose", "agent_evaluation")
            mlflow.log_metric("mock_accuracy", 0.85 if ver_idx == 0 else 0.91)
            mlflow.log_metric("mock_latency_ms", 320 if ver_idx == 0 else 290)
            print(f"  {run_label}: linked to {meta['name']} v{meta['version']}")

    # Query lineage
    print("\n  Lineage query — runs using dataset v3:")
    runs = mlflow.search_runs(
        filter_string="tags.`lineage.dataset_version` = '3'",
        output_format="list",
    )
    for r in runs:
        print(f"    run_id={r.info.run_id[:8]}... name={r.info.run_name}")
    if not runs:
        print("    (no runs found — dataset v3 lineage tag set above)")


# ── Part 3: Data Quality Checks ──────────────────────────────────────────── #


class DataQualityChecker:
    """Run data quality checks before using a dataset for evaluation."""

    REQUIRED_COLUMNS = {"question", "expected_answer", "category"}

    def check_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check for missing values across all columns."""
        total_cells = df.size
        missing_cells = int(df.isnull().sum().sum())
        return {
            "check": "completeness",
            "passed": missing_cells == 0,
            "missing_cells": missing_cells,
            "completeness_ratio": round(1 - missing_cells / max(total_cells, 1), 4),
        }

    def check_format_validity(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate that questions end with '?' and answers are non-empty."""
        if "question" not in df.columns:
            return {"check": "format_validity", "passed": False, "reason": "no question column"}
        q_valid = df["question"].str.strip().str.endswith("?").sum()
        a_valid = (df["expected_answer"].str.len() > 0).sum() if "expected_answer" in df.columns else 0
        return {
            "check": "format_validity",
            "passed": int(q_valid) == len(df) and int(a_valid) == len(df),
            "valid_questions": int(q_valid),
            "valid_answers": int(a_valid),
            "total_rows": len(df),
        }

    def check_uniqueness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check for duplicate questions."""
        if "question" not in df.columns:
            return {"check": "uniqueness", "passed": False, "reason": "no question column"}
        dupes = int(df["question"].duplicated().sum())
        return {
            "check": "uniqueness",
            "passed": dupes == 0,
            "duplicate_count": dupes,
            "unique_ratio": round(1 - dupes / max(len(df), 1), 4),
        }

    def check_schema_compliance(self, df: pd.DataFrame) -> dict[str, Any]:
        """Verify the DataFrame has all required columns."""
        present = set(df.columns)
        missing = self.REQUIRED_COLUMNS - present
        return {
            "check": "schema_compliance",
            "passed": len(missing) == 0,
            "required": sorted(self.REQUIRED_COLUMNS),
            "missing": sorted(missing),
        }

    def run_all(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Run every quality check and return a list of results."""
        return [
            self.check_completeness(df),
            self.check_format_validity(df),
            self.check_uniqueness(df),
            self.check_schema_compliance(df),
        ]


def part3_quality_checks() -> None:
    """Run data quality checks and log the report to MLflow."""
    print("\n" + "=" * 60)
    print("Part 3: Data Quality Checks")
    print("=" * 60)

    # Good dataset
    good_df = pd.DataFrame({
        "question": ["What is 2+2?", "Name a primary color?"],
        "expected_answer": ["4", "Red"],
        "category": ["math", "factual"],
    })

    # Dataset with issues
    bad_df = pd.DataFrame({
        "question": ["What is 2+2?", "No question mark here", "What is 2+2?"],
        "expected_answer": ["4", "", "4"],
        "category": ["math", "factual", "math"],
    })

    checker = DataQualityChecker()

    for label, df in [("good_dataset", good_df), ("bad_dataset", bad_df)]:
        results = checker.run_all(df)
        all_passed = all(r["passed"] for r in results)

        with mlflow.start_run(run_name=f"quality_check_{label}"):
            mlflow.set_tag("quality.dataset", label)
            mlflow.set_tag("quality.all_passed", str(all_passed))
            for r in results:
                mlflow.set_tag(f"quality.{r['check']}", str(r["passed"]))
            mlflow.log_dict(
                {"checks": results, "all_passed": all_passed},
                f"quality_report_{label}.json",
            )

        status = "PASS" if all_passed else "FAIL"
        print(f"\n  {label}: {status}")
        for r in results:
            mark = "OK" if r["passed"] else "FAIL"
            print(f"    [{mark}] {r['check']}")


# ── Part 4: Evaluation Data Pipelines ────────────────────────────────────── #


def part4_data_pipelines() -> None:
    """Build a reusable pipeline: raw → cleaned → augmented → eval-ready."""
    print("\n" + "=" * 60)
    print("Part 4: Evaluation Data Pipelines")
    print("=" * 60)

    # Raw data (messy)
    raw = pd.DataFrame({
        "question": [
            "  What is gravity?  ",
            "explain DNA",
            "What is the speed of light?",
            "",
            "What is gravity?",
        ],
        "expected_answer": [
            "A fundamental force of attraction between masses.",
            "DNA is the molecule carrying genetic instructions.",
            "Approximately 3 x 10^8 m/s.",
            "N/A",
            "Gravity pulls objects toward each other.",
        ],
        "category": ["science", "biology", "physics", "", "science"],
    })
    print(f"  Raw:      {len(raw)} rows")

    with mlflow.start_run(run_name="data_pipeline"):
        mlflow.set_tag("pipeline.stage", "full")

        # Step 1: Clean
        cleaned = raw.copy()
        cleaned["question"] = cleaned["question"].str.strip()
        cleaned = cleaned[cleaned["question"].str.len() > 0]
        cleaned = cleaned[cleaned["expected_answer"] != "N/A"]
        mlflow.log_metric("pipeline.after_clean", len(cleaned))
        print(f"  Cleaned:  {len(cleaned)} rows (removed blanks/N-A)")

        # Step 2: Normalize
        cleaned["question"] = cleaned["question"].apply(
            lambda q: q if q.endswith("?") else q + "?"
        )
        cleaned["category"] = cleaned["category"].replace("", "uncategorized")

        # Step 3: Deduplicate
        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["question"], keep="first")
        mlflow.log_metric("pipeline.duplicates_removed", before_dedup - len(cleaned))
        print(f"  Deduped:  {len(cleaned)} rows (removed {before_dedup - len(cleaned)} dupes)")

        # Step 4: Augment — add difficulty estimate based on answer length
        cleaned = cleaned.copy()
        cleaned["difficulty"] = cleaned["expected_answer"].apply(
            lambda a: "hard" if len(a) > 40 else "easy"
        )
        mlflow.log_metric("pipeline.final_rows", len(cleaned))
        print(f"  Final:    {len(cleaned)} rows, columns={list(cleaned.columns)}")

        # Log the final dataset
        dataset = mlflow.data.from_pandas(
            cleaned, source="pipeline_output", name="pipeline_eval_ready"
        )
        mlflow.log_input(dataset, context="evaluation")
        mlflow.log_dict(
            {
                "stages": ["clean", "normalize", "deduplicate", "augment"],
                "input_rows": len(raw),
                "output_rows": len(cleaned),
            },
            "pipeline_summary.json",
        )


# ── Part 5: Data Drift Detection ─────────────────────────────────────────── #


def detect_drift(
    baseline: pd.DataFrame, current: pd.DataFrame, column: str
) -> dict[str, Any]:
    """Compare value distributions between two dataset versions."""
    base_counts = Counter(baseline[column].dropna())
    curr_counts = Counter(current[column].dropna())

    all_keys = set(base_counts.keys()) | set(curr_counts.keys())
    base_total = max(sum(base_counts.values()), 1)
    curr_total = max(sum(curr_counts.values()), 1)

    # Compute distribution shift (sum of absolute frequency differences)
    shift = sum(
        abs(base_counts.get(k, 0) / base_total - curr_counts.get(k, 0) / curr_total)
        for k in all_keys
    )

    new_cats = sorted(set(curr_counts.keys()) - set(base_counts.keys()))
    missing_cats = sorted(set(base_counts.keys()) - set(curr_counts.keys()))

    return {
        "column": column,
        "distribution_shift": round(shift, 4),
        "new_categories": new_cats,
        "missing_categories": missing_cats,
        "baseline_categories": len(base_counts),
        "current_categories": len(curr_counts),
    }


def part5_drift_detection() -> None:
    """Detect drift between two dataset versions."""
    print("\n" + "=" * 60)
    print("Part 5: Data Drift Detection")
    print("=" * 60)

    baseline = pd.DataFrame({
        "question": [
            "What is gravity?",
            "Explain DNA.",
            "What is 10+5?",
            "Name a planet.",
        ],
        "expected_answer": [
            "A force of attraction.",
            "Molecule carrying genetic code.",
            "15",
            "Earth",
        ],
        "category": ["science", "biology", "math", "science"],
        "difficulty": ["easy", "easy", "easy", "easy"],
    })

    current = pd.DataFrame({
        "question": [
            "What is gravity?",
            "Explain quantum entanglement.",
            "Solve x^2 - 4 = 0.",
            "What is machine learning?",
            "Describe CRISPR.",
        ],
        "expected_answer": [
            "A force of attraction.",
            "Correlated quantum states across distance.",
            "x = 2 or x = -2",
            "A subset of AI that learns from data.",
            "Gene editing technology.",
        ],
        "category": ["science", "physics", "math", "ai", "biology"],
        "difficulty": ["easy", "hard", "hard", "easy", "hard"],
    })

    with mlflow.start_run(run_name="drift_detection"):
        reports: list[dict[str, Any]] = []
        for col in ["category", "difficulty"]:
            report = detect_drift(baseline, current, col)
            reports.append(report)
            mlflow.log_metric(f"drift.{col}.shift", report["distribution_shift"])
            mlflow.set_tag(
                f"drift.{col}.new", json.dumps(report["new_categories"])
            )
            mlflow.set_tag(
                f"drift.{col}.missing", json.dumps(report["missing_categories"])
            )
            print(f"\n  Column '{col}':")
            print(f"    Distribution shift : {report['distribution_shift']}")
            print(f"    New categories     : {report['new_categories']}")
            print(f"    Missing categories : {report['missing_categories']}")

        mlflow.log_dict({"drift_reports": reports}, "drift_report.json")

    has_drift = any(r["distribution_shift"] > 0.3 for r in reports)
    print(f"\n  Significant drift detected: {has_drift}")


# ── Main ──────────────────────────────────────────────────────────────────── #


def main() -> None:
    print("=" * 60)
    print("  L3-4.4 — Advanced Data Management")
    print("  Dataset versioning, lineage, quality, pipelines, drift")
    print("=" * 60)

    manager = part1_dataset_versioning()
    part2_lineage_tracking(manager)
    part3_quality_checks()
    part4_data_pipelines()
    part5_drift_detection()

    print("\n" + "=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5000")
    print("  Experiment: L3/M4_advanced_features/4_data_management")
    print("=" * 60)


if __name__ == "__main__":
    main()
