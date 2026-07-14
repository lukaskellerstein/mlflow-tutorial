"""L1-M8.1 — Authentication and Permissions

Demonstrates MLflow's authentication concepts, permission model,
and how to configure auth for multi-user/production deployments.
"""

import os

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M8_auth/1_auth_permissions")


def main() -> None:
    # ── Part 1: When You Need Authentication ──────────────────────
    print("=" * 60)
    print("Part 1: When You Need Authentication")
    print("=" * 60)
    print()
    print("Local development (our setup):")
    print("  - Single user, no auth needed")
    print("  - MLflow server started without --app-name basic-auth")
    print()
    print("Multi-user / production:")
    print("  - Multiple data scientists sharing one server")
    print("  - Need to control who can modify experiments/models")
    print("  - Enable auth to protect data and enforce access control")
    print()

    # ── Part 2: How to Enable Authentication ──────────────────────
    print("=" * 60)
    print("Part 2: How to Enable Authentication")
    print("=" * 60)
    print()
    print("Start the server with auth enabled:")
    print("  mlflow server --app-name basic-auth")
    print()
    print("Default admin credentials (change immediately!):")
    print("  Username: admin")
    print("  Password: password")
    print()
    print("Set credentials via environment variables:")
    print('  export MLFLOW_TRACKING_USERNAME="admin"')
    print('  export MLFLOW_TRACKING_PASSWORD="password"')
    print()
    print("Or set them programmatically in Python:")
    print('  os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"')
    print('  os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"')
    print()

    # ── Part 3: Permission Model ──────────────────────────────────
    print("=" * 60)
    print("Part 3: Permission Model")
    print("=" * 60)
    print()
    print("Permission levels (from least to most access):")
    print("  NO_PERMISSIONS — no access at all")
    print("  READ           — view experiments, runs, models")
    print("  EDIT           — READ + create/modify runs and models")
    print("  MANAGE         — EDIT + delete, grant permissions")
    print()
    print("Permissions apply at two scopes:")
    print("  1. Experiment-level — control access per experiment")
    print("  2. Model-level      — control access per registered model")
    print()
    print("Admin API examples (when auth is enabled):")
    print("  from mlflow.server import get_app_client")
    print('  auth_client = get_app_client("basic-auth", "http://...:5000")')
    print('  auth_client.create_user("alice", "secret123")')
    print('  auth_client.update_experiment_permission(')
    print('      "experiment_id", "alice", "EDIT"')
    print("  )")
    print()

    # ── Part 4: Demo Without Auth ─────────────────────────────────
    print("=" * 60)
    print("Part 4: Demo — Our Server Works Without Auth")
    print("=" * 60)
    print()

    with mlflow.start_run(run_name="auth_demo_run") as run:
        mlflow.log_param("auth_enabled", False)
        mlflow.log_param("environment", "local_development")
        mlflow.log_metric("demo_score", 1.0)

        print(f"Run ID:     {run.info.run_id}")
        print(f"Experiment: {run.info.experiment_id}")
        print("Logged param:  auth_enabled=False")
        print("Logged param:  environment=local_development")
        print("Logged metric: demo_score=1.0")

    print()
    print("Our local server accepted the run without credentials.")
    print("In production, enable auth to protect your MLflow data.")
    print()
    print("=" * 60)
    print("Lesson complete! View results at http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
