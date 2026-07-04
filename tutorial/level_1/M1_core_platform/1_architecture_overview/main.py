"""
L1-1.1 — What is MLflow? Architecture Overview

This lesson introduces MLflow's architecture and core concepts.
We create a simple run to verify the connection to the MLflow server
and demonstrate the basic building blocks: experiments, runs,
parameters, metrics, artifacts, and tags.
"""

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_core_platform/1_architecture_overview"


def main() -> None:
    # Connect to the MLflow tracking server
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # 1. MLflow Architecture Overview
    # ------------------------------------------------------------------
    print("=" * 60)
    print("MLflow Architecture — The 5 Pillars")
    print("=" * 60)
    pillars = {
        "1. Tracking": "Record parameters, metrics, and artifacts for every experiment run.",
        "2. Models": "Package ML/AI models in a standard format for any framework.",
        "3. Model Registry": "Centralized model store with versioning and lifecycle management.",
        "4. Evaluation": "Evaluate model quality with built-in and custom metrics.",
        "5. Deployment": "Serve models as REST APIs or batch-process predictions.",
    }
    for pillar, description in pillars.items():
        print(f"  {pillar}")
        print(f"    {description}")
    print()

    # ------------------------------------------------------------------
    # 2. Key Concepts
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Key Concepts")
    print("=" * 60)
    concepts = [
        ("Experiment", "A named collection of runs (e.g., one per project or task)."),
        ("Run", "A single execution — stores params, metrics, artifacts, and tags."),
        ("Parameters", "Input configuration values (model name, learning rate, ...)."),
        ("Metrics", "Numeric results you want to track (accuracy, latency, ...)."),
        ("Artifacts", "Output files — models, plots, data snapshots."),
        ("Tags", "Free-form key/value metadata for organizing and filtering runs."),
    ]
    for name, description in concepts:
        print(f"  {name:12s} — {description}")
    print()

    # ------------------------------------------------------------------
    # 3. Create a Run
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Creating a run on the MLflow server")
    print("=" * 60)
    print(f"  Tracking URI : {TRACKING_URI}")
    print(f"  Experiment   : {EXPERIMENT_NAME}")
    print(f"  MLflow version: {mlflow.__version__}")
    print()

    with mlflow.start_run(run_name="architecture_overview") as run:
        # Log parameters
        mlflow.log_params(
            {
                "framework": "mlflow",
                "version": mlflow.__version__,
                "lesson": "L1-1.1",
            }
        )

        # Log a metric
        mlflow.log_metric("setup_complete", 1.0)

        # Log tags
        mlflow.set_tags(
            {
                "level": "1",
                "module": "core_platform",
                "lesson": "architecture_overview",
            }
        )

        print(f"  Run ID   : {run.info.run_id}")
        print(f"  Status   : {run.info.status}")
        print(f"  Artifact URI: {run.info.artifact_uri}")
    print()

    # ------------------------------------------------------------------
    # 4. MLflow in the Red Hat AI Ecosystem
    # ------------------------------------------------------------------
    print("=" * 60)
    print("MLflow in the Red Hat AI Ecosystem")
    print("=" * 60)
    print("  OpenShift AI includes a managed MLflow operator")
    print("  (`mlflowoperator` in the DataScienceCluster CR, GA in 3.4).")
    print()
    print("  Managed MLflow vs. Standalone:")
    print("    - Managed: deployed and lifecycle-managed by the operator")
    print("      on OpenShift. Handles HA, TLS, RBAC automatically.")
    print("    - Standalone: what this tutorial teaches. You run the")
    print("      server yourself (via Podman Compose in our case).")
    print()
    print("  The MLflow APIs and tracking code are identical in both")
    print("  modes -- only the deployment and operations differ.")
    print()

    with mlflow.start_run(run_name="architecture_overview") as run:
        mlflow.set_tags({
            "level": "1",
            "module": "core_platform",
            "lesson": "architecture_overview",
            "note": "Red Hat OpenShift AI includes managed MLflow operator (GA in 3.4)",
        })
        mlflow.log_param("deployment_mode", "standalone")
        mlflow.log_metric("openshift_ai_operator_ga_version", 3.4)
        print(f"  Logged Red Hat ecosystem info to run: {run.info.run_id}")
    print()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Open the MLflow UI at {TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  You should see the run with parameters, metrics, and tags.")


if __name__ == "__main__":
    main()
