# L3-4.4 — Advanced Data Management

**Level:** Expert
**Duration:** ~45 minutes

## Overview

Production ML systems live and die by their data. This lesson covers advanced data management patterns with MLflow: versioning evaluation datasets with hash-based identity, tracking dataset-to-run lineage, enforcing quality gates before evaluation, building reusable data pipelines, and detecting drift between dataset versions.

## Prerequisites

- Completed: L1-M6.1 (Dataset Logging), L1-M4.1 (LLM Evaluation Basics)
- MLFlow server running at http://127.0.0.1:5000

## Concepts

### Dataset Versioning
Evaluation datasets evolve over time as you add edge cases, refine expected answers, or remove flawed examples. Each version needs a stable identity (content hash) and metadata so you can reproduce any past evaluation exactly.

### Dataset Lineage
When an evaluation run produces surprising results, the first question is "which dataset version was used?" Lineage tracking connects every run to the exact dataset that fed it, enabling root-cause analysis.

### Data Quality Gates
Bad data produces misleading evaluations. Quality checks (completeness, format, uniqueness, schema compliance) should run automatically before any evaluation, blocking runs on data that does not meet minimum standards.

### Data Pipelines
Raw evaluation data is rarely ready for use. A pipeline that cleans, normalizes, deduplicates, and augments data produces consistent, high-quality evaluation sets and logs every transformation step for reproducibility.

### Data Drift Detection
When a new dataset version differs significantly from the baseline (new categories, shifted distributions), evaluation results may not be comparable. Drift detection quantifies these changes so you can decide whether a direct comparison is valid.

## Step-by-Step

### Step 1: Dataset Versioning (Part 1)

The `DatasetManager` class assigns a SHA-256-based digest to each DataFrame snapshot and tracks version metadata. Three versions of an evaluation dataset are created, each logged to MLflow with version tags and metadata artifacts.

```python
manager = DatasetManager("agent_eval_dataset")
meta = manager.add_version(df, "Initial 3-question eval set")
with mlflow.start_run(run_name="dataset_v1"):
    manager.log_to_mlflow(df, meta)
```

### Step 2: Lineage Tracking (Part 2)

Each evaluation run is tagged with the dataset name, version, and digest. You can then query runs by dataset version:

```python
runs = mlflow.search_runs(
    filter_string="tags.`lineage.dataset_version` = '3'",
    output_format="list",
)
```

### Step 3: Data Quality Checks (Part 3)

The `DataQualityChecker` runs four checks on a DataFrame:
- **completeness** -- no missing values
- **format_validity** -- questions end with `?`, answers are non-empty
- **uniqueness** -- no duplicate questions
- **schema_compliance** -- all required columns present

Results are logged as a JSON artifact and summary tags.

### Step 4: Data Pipelines (Part 4)

A four-stage pipeline transforms raw data into evaluation-ready form:
1. **Clean** -- remove blank rows and placeholder values
2. **Normalize** -- ensure question marks, fill empty categories
3. **Deduplicate** -- remove duplicate questions
4. **Augment** -- add computed columns (difficulty estimate)

Each stage's output size is logged as a metric for pipeline observability.

### Step 5: Drift Detection (Part 5)

The `detect_drift` function compares category distributions between a baseline and a current dataset version, reporting:
- **distribution_shift** -- sum of absolute frequency differences (0 = identical)
- **new_categories** -- values present in current but not baseline
- **missing_categories** -- values present in baseline but not current

## Running the Lesson

```bash
cd tutorial/level_3/M4_advanced_features/4_data_management
uv sync
uv run python main.py
```

## Expected Output

```
==========================================================
  L3-4.4 — Advanced Data Management
  Dataset versioning, lineage, quality, pipelines, drift
==========================================================
==========================================================
Part 1: Dataset Versioning
==========================================================
  v1: 3 rows, digest=<hash>
  v2: 5 rows, digest=<hash>
  v3: 5 rows, digest=<hash>
  Total versions tracked: 3

==========================================================
Part 2: Dataset Lineage Tracking
==========================================================
  eval_on_v1: linked to agent_eval_dataset v1
  eval_on_v3: linked to agent_eval_dataset v3
  Lineage query — runs using dataset v3:
    run_id=...  name=eval_on_v3

==========================================================
Part 3: Data Quality Checks
==========================================================
  good_dataset: PASS
    [OK] completeness
    [OK] format_validity
    [OK] uniqueness
    [OK] schema_compliance

  bad_dataset: FAIL
    [OK] completeness
    [FAIL] format_validity
    [FAIL] uniqueness
    [OK] schema_compliance

==========================================================
Part 4: Evaluation Data Pipelines
==========================================================
  Raw:      5 rows
  Cleaned:  4 rows (removed blanks/N-A)
  Deduped:  3 rows (removed 1 dupes)
  Final:    3 rows, columns=[question, expected_answer, category, difficulty]

==========================================================
Part 5: Data Drift Detection
==========================================================
  Column 'category':
    Distribution shift : 1.1
    New categories     : ['ai', 'physics']
    Missing categories : []

  Column 'difficulty':
    Distribution shift : 0.6
    New categories     : ['hard']
    Missing categories : []

  Significant drift detected: True

==========================================================
  Done! View results in MLflow UI: http://127.0.0.1:5000
  Experiment: L3/M4_advanced_features/4_data_management
==========================================================
```

In the MLflow UI you will see:
- Separate runs for each dataset version, evaluation, quality check, pipeline, and drift detection
- Dataset inputs linked to runs (visible on the run's Datasets tab)
- JSON artifacts containing quality reports, pipeline summaries, and drift reports
- Tags for lineage queries and quality check results

## Key Takeaways

- Hash-based dataset versioning gives every dataset snapshot a stable, content-derived identity for exact reproducibility.
- Lineage tags on runs let you trace any evaluation result back to the exact dataset version that produced it.
- Automated quality checks (completeness, format, uniqueness, schema) act as gates that prevent bad data from corrupting evaluations.
- Reusable data pipelines (clean, normalize, deduplicate, augment) produce consistent evaluation sets and log each transformation step.
- Drift detection quantifies distribution changes between dataset versions so you know when comparisons are no longer apples-to-apples.

## Next Steps

This completes the Advanced Features module. Continue to L3-M5 (Capstones) to build a full production agent evaluation platform that integrates all the patterns from Levels 1-3.
