# L3-4.2 — Enterprise MLflow Patterns

**Level:** Expert
**Duration:** 1 hour

## Overview

Enterprise deployments of MLflow require more than just experiment tracking. Teams need governance workflows, audit trails, cost visibility, and access control conventions. This lesson demonstrates production patterns for running MLflow across multiple teams at scale.

## Prerequisites

- Completed: L1-M1 (Tracking), L1-M2 (model registry), L1-M10 (auth)
- MLflow server running at http://127.0.0.1:5000
- Familiarity with `MlflowClient` programmatic API

## Concepts

### Multi-Team Organization

Large organizations run dozens of ML projects across multiple teams. Without structure, the MLflow UI becomes a flat list of unrelated experiments. Hierarchical naming conventions (`team/project/experiment`) create virtual namespaces that mirror the org chart.

### Model Governance

Production models need formal lifecycle management. The governance workflow enforces that every model passes through defined stages (development, staging, champion) with required approval checks at each transition. This prevents untested models from reaching production.

### Audit Logging

Regulated industries (finance, healthcare) require immutable records of who changed what and when. The audit logger captures every governance action as structured JSON and stores it as an MLflow artifact for long-term retention.

### Cost Tracking

LLM-based applications can incur significant inference costs. Tracking token usage and estimated costs per model, per operation, lets teams set budgets, detect anomalies, and optimize their model choices.

### Access Control Patterns

While MLflow's built-in auth handles user-level permissions, enterprise deployments use naming conventions and tags to create logical boundaries between teams and environments.

## Step-by-Step

### Step 1: Multi-Team Experiment Organization

We create experiments following a `team/project` hierarchy and tag them with ownership metadata. This enables filtering by team across the entire MLflow instance.

```python
exp_name = f"enterprise/{team}/{project}"
client.create_experiment(exp_name, tags={
    "team": team,
    "owner": info["owner"],
    "environment": "development",
})
```

### Step 2: Model Governance Workflow

The `ModelGovernance` class enforces a staged promotion process. Each transition requires all validation checks to pass:

```python
governance.promote(
    name, version, actor="bob@company.com",
    checks={"unit_tests": True, "integration_tests": True, "code_review": True},
)
```

If any check fails, the promotion is rejected and the rejection is logged to the audit trail.

### Step 3: Audit Logging

Every governance action (registration, promotion, rejection) is captured with actor, timestamp, and details:

```python
audit.log(
    action="MODEL_PROMOTED",
    model_name=name, version=version,
    actor=actor,
    details=f"{current_stage} -> {next_stage}",
)
```

The full trail is saved as a JSON artifact on the MLflow run.

### Step 4: Cost Tracking

The `CostTracker` records token usage per call and estimates costs based on per-model pricing tables:

```python
tracker.record("gemma4:26b", input_tokens=2048, output_tokens=512, operation="agent-reasoning")
tracker.log_to_mlflow()
report = tracker.report()  # Per-model cost summary DataFrame
```

### Step 5: Access Control Patterns

We demonstrate three complementary patterns:
- **Namespace conventions**: `prod/models/X` vs `dev/models/X` with different access levels
- **Tag-based filtering**: `visibility=team-only`, `data_classification=internal`
- **Environment separation**: `<team>/<project>/<env>` with escalating approval requirements

## Running the Lesson

```bash
cd tutorial/level_3/M4_advanced_features/2_enterprise
uv sync
uv run python main.py
```

## Expected Output

The script produces five sections of output:

1. **Team Organization** -- creates 6 experiments (`enterprise/<team>/<project>`) and lists them per team
2. **Governance Workflow** -- registers a model, promotes it through stages, shows a rejected promotion, then a successful retry
3. **Audit Trail** -- prints each audit entry as it is recorded, then saves the JSON artifact
4. **Cost Tracking** -- logs 8 simulated LLM calls with token counts and per-call costs, then prints a per-model summary table
5. **Access Control** -- prints namespace conventions, tag-based filtering results, and environment separation guidelines

In the MLflow UI you will see:
- Experiments under `enterprise/` organized by team
- The main experiment with governance, cost, and access control runs
- Model `enterprise-demo-model` in the registry with governance tags and a `champion` alias
- Cost metrics (`cost/total_usd`, `cost/total_tokens`) on the cost-tracking run
- An `audit_trail.json` artifact under the governance run

## Key Takeaways

- Hierarchical experiment naming (`team/project/experiment`) brings order to multi-team MLflow deployments
- Model governance workflows with programmatic approval gates prevent untested models from reaching production
- Audit trails stored as MLflow artifacts satisfy compliance requirements without external systems
- Cost tracking at the API-call level enables budgeting and optimization for LLM-heavy workloads
- Namespace conventions and tag-based filtering provide logical access control even without fine-grained permissions

## Next Steps

Continue to L3-4.3 (MLflow + MCP) to learn how the Model Context Protocol integrates with MLflow for standardized tool and resource access in agent workflows.
