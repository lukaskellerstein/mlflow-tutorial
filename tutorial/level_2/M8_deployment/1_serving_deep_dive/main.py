"""
L2-8.1 — Model Serving Deep Dive

Explores MLflow model serving in depth:
- Preparing multiple models for serving with signatures and input examples
- Serving CLI configuration options (workers, host, port, timeout)
- Custom PyFunc models with preprocessing in predict()
- Docker containerization via `mlflow models build-docker`
- Health check and monitoring endpoints (/ping, /health, /version)

NOTE: This lesson does NOT start a live server. It prepares models,
registers them, and documents the serving patterns you would use
in production.
"""

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Part 1 — Prepare multiple models for serving
# ---------------------------------------------------------------------------


def prepare_models() -> tuple[str, str]:
    """Train RF and GB on the wine dataset, log with signatures, register."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    df_train = pd.DataFrame(X_train, columns=wine.feature_names)
    df_test = pd.DataFrame(X_test, columns=wine.feature_names)

    model_uris: list[str] = []

    configs = [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    for name, model in configs:
        with mlflow.start_run(run_name=f"serve_{name}") as run:
            model.fit(df_train, y_train)
            preds = model.predict(df_test)
            accuracy = (preds == y_test).mean()

            signature = infer_signature(df_test, preds)
            input_example = df_test.head(3)

            mlflow.log_params({"model_type": name, "n_estimators": 100})
            mlflow.log_metric("accuracy", accuracy)

            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=input_example,
            )

            # Register the model
            reg_name = f"serving-demo-{name}"
            mlflow.register_model(info.model_uri, reg_name)
            model_uris.append(info.model_uri)

            print(f"  {name:25s} accuracy={accuracy:.4f}  uri={info.model_uri}")

    return model_uris[0], model_uris[1]


# ---------------------------------------------------------------------------
# Part 2 — Serving configurations
# ---------------------------------------------------------------------------

SERVING_CONFIG = {
    "model_uri": "models:/serving-demo-RandomForest/1",
    "host": "0.0.0.0",
    "port": 5001,
    "workers": 4,
    "timeout": 120,
    "enable_mlserver": False,
    "no_conda": True,
}


def show_serving_configurations() -> None:
    """Print CLI options and log a config file as an artifact."""
    print("  CLI command:")
    print(
        f"    mlflow models serve \\\n"
        f"      --model-uri {SERVING_CONFIG['model_uri']} \\\n"
        f"      --host {SERVING_CONFIG['host']} \\\n"
        f"      --port {SERVING_CONFIG['port']} \\\n"
        f"      --workers {SERVING_CONFIG['workers']} \\\n"
        f"      --timeout {SERVING_CONFIG['timeout']} \\\n"
        f"      --no-conda"
    )
    print()
    print("  Environment variable equivalents:")
    env_vars = {
        "MLFLOW_MODEL_URI": SERVING_CONFIG["model_uri"],
        "MLFLOW_HOST": SERVING_CONFIG["host"],
        "MLFLOW_PORT": str(SERVING_CONFIG["port"]),
        "MLFLOW_WORKERS": str(SERVING_CONFIG["workers"]),
    }
    for key, val in env_vars.items():
        print(f"    export {key}={val}")

    # Log the serving config as a JSON artifact
    with mlflow.start_run(run_name="serving_config"):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "serving_config.json"
            config_path.write_text(json.dumps(SERVING_CONFIG, indent=2))
            mlflow.log_artifact(str(config_path))
        print(f"\n  Serving config logged as artifact: serving_config.json")


# ---------------------------------------------------------------------------
# Part 3 — Custom request/response handling via PyFunc
# ---------------------------------------------------------------------------


class WineClassifierWithPreprocessing(mlflow.pyfunc.PythonModel):
    """Custom PyFunc that bundles a scaler and classifier.

    This demonstrates how to embed preprocessing in the serving layer so
    clients can send raw (unscaled) feature values.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import pickle
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
        with open(context.artifacts["classifier"], "rb") as f:
            self.classifier = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """Scale inputs, predict class, and return class name."""
        wine = load_wine()
        scaled = self.scaler.transform(model_input)
        class_ids = self.classifier.predict(scaled)
        class_names = [wine.target_names[i] for i in class_ids]
        return pd.DataFrame({"class_id": class_ids, "class_name": class_names})


def log_custom_pyfunc() -> str:
    """Train scaler+model, wrap in a custom PyFunc, log and register."""
    import pickle

    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    df_test = pd.DataFrame(X_test, columns=wine.feature_names)

    df_train_full = pd.DataFrame(X_train, columns=wine.feature_names)
    scaler = StandardScaler().fit(df_train_full)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(scaler.transform(df_train_full), y_train)

    with mlflow.start_run(run_name="custom_pyfunc_serving") as run:
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = str(Path(tmp) / "scaler.pkl")
            clf_path = str(Path(tmp) / "classifier.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            with open(clf_path, "wb") as f:
                pickle.dump(clf, f)

            artifacts = {"scaler": scaler_path, "classifier": clf_path}
            sample_input = df_test.head(3)
            sample_output = pd.DataFrame({
                "class_id": [0, 0, 0],
                "class_name": ["class_0", "class_0", "class_0"],
            })
            signature = infer_signature(sample_input, sample_output)

            info = mlflow.pyfunc.log_model(
                name="custom_model",
                python_model=WineClassifierWithPreprocessing(),
                artifacts=artifacts,
                signature=signature,
                input_example=sample_input,
            )

        mlflow.register_model(info.model_uri, "serving-demo-CustomPyFunc")
        print(f"  Custom PyFunc model URI: {info.model_uri}")

        # Verify it works by loading and predicting
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        result = loaded.predict(df_test.head(3))
        print(f"  Sample prediction:\n{result.to_string(index=False)}")
        return info.model_uri


# ---------------------------------------------------------------------------
# Part 4 — Docker containerization
# ---------------------------------------------------------------------------


def show_docker_containerization() -> None:
    """Print Docker build commands and log a deployment guide."""
    print("  Build a Docker image for a registered model:")
    print("    mlflow models build-docker \\")
    print("      --model-uri models:/serving-demo-RandomForest/1 \\")
    print("      --name mlflow-wine-server")
    print()
    print("  Run the container:")
    print("    podman run -p 5001:8080 mlflow-wine-server")
    print()
    print("  Send a prediction request:")
    print('    curl -X POST http://localhost:5001/invocations \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"dataframe_split": {"columns": [...], "data": [[...]]}}\'')

    deployment_guide = (
        "# Deployment Guide\n\n"
        "## Docker Build\n"
        "mlflow models build-docker --model-uri models:/<name>/<version> --name <image>\n\n"
        "## Endpoints\n"
        "- POST /invocations  — prediction requests\n"
        "- GET  /ping         — liveness probe (returns 200)\n"
        "- GET  /health       — same as /ping\n"
        "- GET  /version      — MLflow version info\n\n"
        "## Input Formats\n"
        "- dataframe_split: {columns: [...], data: [[...]]}\n"
        "- dataframe_records: [{col: val, ...}, ...]\n"
        "- instances: [[val, ...], ...] (TF Serving compatible)\n\n"
        "## Health Checks (Kubernetes)\n"
        "livenessProbe:\n"
        "  httpGet: {path: /ping, port: 8080}\n"
        "readinessProbe:\n"
        "  httpGet: {path: /health, port: 8080}\n"
    )

    with mlflow.start_run(run_name="deployment_guide"):
        with tempfile.TemporaryDirectory() as tmp:
            guide_path = Path(tmp) / "deployment_guide.md"
            guide_path.write_text(deployment_guide)
            mlflow.log_artifact(str(guide_path))
        print("\n  Deployment guide logged as artifact: deployment_guide.md")


# ---------------------------------------------------------------------------
# Part 5 — Health checks and monitoring
# ---------------------------------------------------------------------------


def show_health_and_monitoring() -> None:
    """Print endpoint details and monitoring best practices."""
    endpoints = {
        "GET /ping": "Liveness check. Returns 200 with empty body when server is ready.",
        "GET /health": "Alias for /ping. Use in Kubernetes readiness probes.",
        "GET /version": "Returns MLflow version. Useful for debugging deployments.",
        "POST /invocations": "Prediction endpoint. Accepts JSON or CSV input.",
    }
    for ep, desc in endpoints.items():
        print(f"  {ep:25s} {desc}")

    print()
    print("  Monitoring best practices:")
    practices = [
        "Track request latency (p50, p95, p99) via a reverse proxy or sidecar.",
        "Log prediction counts and error rates to Prometheus.",
        "Set up alerts for latency spikes or error rate > threshold.",
        "Use structured logging (JSON) for easier aggregation.",
        "Monitor model staleness — track when the served version was last updated.",
    ]
    for i, p in enumerate(practices, 1):
        print(f"    {i}. {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Part 1: Preparing multiple models for serving")
    print("=" * 60)
    rf_uri, gb_uri = prepare_models()
    print()

    print("=" * 60)
    print("Part 2: Serving configurations")
    print("=" * 60)
    show_serving_configurations()
    print()

    print("=" * 60)
    print("Part 3: Custom PyFunc with preprocessing")
    print("=" * 60)
    custom_uri = log_custom_pyfunc()
    print()

    print("=" * 60)
    print("Part 4: Docker containerization")
    print("=" * 60)
    show_docker_containerization()
    print()

    print("=" * 60)
    print("Part 5: Health checks and monitoring endpoints")
    print("=" * 60)
    show_health_and_monitoring()
    print()

    print("=" * 60)
    print("Part 6: OpenShift AI Managed MLflow")
    print("=" * 60)
    print("  The `mlflowoperator` DSC component deploys MLflow as a managed")
    print("  service on OpenShift AI clusters. Key differences from standalone:")
    print()
    print("    Managed (OpenShift AI)         Standalone (this tutorial)")
    print("    ─────────────────────          ─────────────────────────")
    print("    HA handled by operator         Single server instance")
    print("    TLS configured automatically   Manual TLS setup")
    print("    RBAC via OpenShift             MLflow auth or none")
    print("    Workbenches connect auto       Set MLFLOW_TRACKING_URI")
    print("    Lifecycle managed              You manage upgrades")
    print()
    print("  The tracking code is identical — you only write the Python SDK")
    print("  calls. The serving patterns in this lesson (Docker, PyFunc,")
    print("  health checks) apply to both modes.")
    print()

    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("  Experiment: L2/M8_deployment/1_serving_deep_dive")
    print("  Registered models: serving-demo-RandomForest,")
    print("    serving-demo-GradientBoosting, serving-demo-CustomPyFunc")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M8_deployment/1_serving_deep_dive")
    main()
