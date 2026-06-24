# L3-3.4 — CI/CD Quality Gates for AI Applications

**Level:** Expert
**Duration:** 1.5 hours

## Overview

Deploy AI models with confidence by enforcing automated quality gates in your CI/CD pipeline. This lesson builds a production-quality evaluation harness that runs test cases against an LLM, checks results against configurable thresholds, and makes a pass/fail deploy decision -- all logged to MLflow for auditability.

## Prerequisites

- Completed: L3-3.1 (Production Tracing), L3-3.3 (Feedback Loops)
- Completed: L1-4.2 (LLM Eval Basics), L2-3.1 (Custom Metrics)
- MLFlow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled

## Concepts

### Why Quality Gates for LLMs?

Traditional software uses unit tests and code coverage to gate deployments. LLM applications need a different approach because their outputs are non-deterministic. A prompt change, model update, or temperature tweak can silently degrade quality. Quality gates provide automated guardrails: define thresholds for accuracy, latency, consistency, and error rate, then enforce them before every deployment.

### What Makes a Good Quality Gate?

A quality gate is a measurable threshold that must be met before a deployment proceeds:

- **Accuracy** -- does the model produce correct answers? Measured by running a fixed set of test cases with known expected outputs.
- **Latency (P95)** -- is the model fast enough? The 95th percentile latency catches tail-end slowness that averages hide.
- **Consistency** -- does the model give the same answer when asked twice? Low consistency signals instability.
- **Error rate** -- how often does the model fail entirely (timeouts, crashes, refusals)?

### CI/CD Integration Pattern

In a real pipeline (GitHub Actions, GitLab CI, Jenkins), the flow is:

1. Code change triggers the pipeline
2. Evaluation harness runs test cases against the candidate model
3. Gate checker compares metrics against thresholds
4. If all gates pass, the deployment proceeds (promote model, update alias)
5. If any gate fails, the pipeline blocks and reports which gates failed
6. All results are logged to MLflow for audit trail and trend analysis

This lesson simulates steps 2-6 locally. The pattern translates directly to any CI system.

## Step-by-Step

### Step 1: Define Quality Gates

The `QualityGate` class holds configurable thresholds. Start with reasonable defaults and tighten them as your model matures:

```python
@dataclass
class QualityGate:
    min_accuracy: float = 0.7        # 70% of test cases must pass
    max_latency_p95_ms: float = 5000 # P95 latency under 5 seconds
    min_consistency: float = 0.6     # 60% consistency across retries
    max_error_rate: float = 0.1      # Under 10% error rate
```

### Step 2: Run the Evaluation Harness

The `EvaluationHarness` runs each test case multiple times (default: 2 runs) to measure both accuracy and consistency. It collects per-case pass rates, latencies, and error counts.

```python
harness = EvaluationHarness(model_name="gemma4:e2b", temperature=0.0)
metrics = harness.run(TEST_CASES, runs=2)
```

### Step 3: Check Quality Gates

The `GateChecker` evaluates metrics against each threshold and returns a structured pass/fail result:

```python
checker = GateChecker(gate)
gate_results = checker.check(metrics)
```

### Step 4: Make the Deploy Decision

If all gates pass, the deployment is approved. If any gate fails, the deployment is blocked with a detailed report showing which gates failed and by how much.

### Step 5: Track Gate History

The `analyze_gate_history` function queries past pipeline runs from MLflow to detect quality trends over time -- whether accuracy is improving or declining across successive deployments.

## Running the Lesson

```bash
cd tutorial/level_3/M3_production/4_cicd
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-3.4 — CI/CD Quality Gates for AI Applications
============================================================

--- Part 1: Quality Gate Definitions ---
  Thresholds: {'min_accuracy': 0.7, 'max_latency_p95_ms': 5000.0, ...}

--- Part 4a: CI/CD Pipeline - Standard Gates ---
  Step 1: Running evaluation harness ...
    Accuracy:      83.33%
    P95 latency:   1200 ms
    Consistency:   100.00%
    Error rate:    0.00%

  Step 2: Checking quality gates ...
  ============================================================
    Quality Gate Report
  ============================================================
    [PASS] accuracy              | Accuracy 83.33% >= 70.00%
    [PASS] latency_p95           | P95 latency 1200ms <= 5000ms
    [PASS] consistency           | Consistency 100.00% >= 60.00%
    [PASS] error_rate            | Error rate 0.00% <= 10.00%
  ------------------------------------------------------------
    Verdict: ALL GATES PASSED
  ============================================================

  Step 3: Deploy APPROVED -- all quality gates passed

--- Part 4b: CI/CD Pipeline - Strict Gates ---
  (likely blocked due to strict thresholds)

--- Part 5: Gate History ---
  Found 2 pipeline run(s):
    [PASS] abc12345  acc=83.33%  p95=1200ms  -> approved
    [FAIL] def67890  acc=83.33%  p95=1200ms  -> blocked
```

Exact numbers will vary based on model performance and system latency.

## Key Takeaways

- Quality gates turn subjective "is the model good enough?" into objective, automated checks.
- Run test cases multiple times to measure consistency, not just accuracy.
- Use two gate profiles: standard (for routine deploys) and strict (for critical models).
- Log every pipeline run to MLflow for audit trail and trend detection.
- Track gate history over time to catch gradual quality degradation before it becomes a problem.

## Next Steps

Return to the MLflow UI and explore the `L3/M3_production/4_cicd` experiment. Compare the standard-gates run (likely approved) with the strict-gates run (likely blocked). In L3-4.1, you will extend MLflow with custom plugins to build reusable quality gate components.
