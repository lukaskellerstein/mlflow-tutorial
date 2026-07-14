# L3-M3.3 — Enterprise Patterns and Data Management

**Level:** Advanced
**Duration:** 90 min

## Overview

Production MLflow deployments serve multiple teams with different access needs, require model governance workflows, cost tracking, and systematic data management. This lesson covers enterprise organization patterns, model lifecycle governance, LLM cost estimation, dataset versioning with lineage tracking, data quality checks, and drift detection.

## Prerequisites

- Completed: L1-M1 (Tracking), L1-M3 (Models and Registry)
- Completed: L1-M4.4 (Datasets and Human-in-the-Loop)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Enterprise Patterns

- **Multi-team experiment organization** — hierarchical naming (`enterprise/team/project`) with team/owner tags
- **Model governance** — staged promotion (development → staging → champion) with approval gates
- **Cost tracking** — estimating LLM costs per model and operation
- **Access control** — namespace conventions and tag-based filtering for team isolation

### Data Management

- **Dataset versioning** — hash-based identity for tracking dataset evolution
- **Lineage tracking** — linking evaluation runs to specific dataset versions
- **Quality checks** — automated validation (completeness, uniqueness, format) before evaluation
- **Drift detection** — comparing category distributions across dataset versions

## Step-by-Step

### Part 1: Multi-Team Organization
Hierarchical experiment structure with team tags and search by team.

### Part 2: Model Governance
Register a model, promote through stages with approval checks, handle rejections and retries.

### Part 3: Cost Tracking
Simulate LLM calls across models, compute estimated costs, log aggregate metrics.

### Part 4: Dataset Versioning and Lineage
Three dataset versions with hash digests, linked to evaluation runs via lineage tags.

### Part 5: Quality Checks and Drift Detection
Automated quality validation on good vs bad datasets, plus category distribution drift detection.

## Running the Lesson

```bash
cd tutorial/level_3_advanced/M3_extensibility/3_enterprise_data_management
uv sync
uv run python main.py
```

## Expected Output

```
Part 1: Multi-Team Experiment Organization
  Created: enterprise/data-science/churn-prediction ...
  Searching experiments by team...

Part 2: Model Governance Workflow
  Registered 'enterprise-demo-model' v1
  PROMOTED development -> staging
  REJECTED staging->champion: failed ['security_scan']
  PROMOTED staging -> champion

Part 3: LLM Cost Tracking
  classification    | google/gemma-4-e4b    | $0.000090
  ...
  Total estimated cost: $0.012150

Part 4: Dataset Versioning and Lineage
  v1: 3 rows, digest=a1b2c3...
  v2: 5 rows, digest=d4e5f6...
  v3: 5 rows, digest=g7h8i9...

Part 5: Data Quality Checks and Drift Detection
  good: PASS
  bad: FAIL (missing=0, dupes=1, empty=1)
  Category shift: 0.85
  New categories: ['ai', 'physics']
```

## Key Takeaways

- Hierarchical experiment naming enables multi-team isolation without infrastructure changes
- Model governance with staged promotion and approval gates prevents untested models from reaching production
- Cost tracking across models reveals optimization opportunities (local vs cloud tradeoffs)
- Dataset versioning with hash digests ensures reproducibility across evaluation runs
- Automated quality checks catch data issues before they contaminate evaluation results
- Drift detection alerts you when your evaluation data no longer represents your production distribution

## Next Steps

Continue to L3-M4.1 (Capstone: Production AI Agent Platform) to integrate these enterprise and data management patterns into a complete production system.
