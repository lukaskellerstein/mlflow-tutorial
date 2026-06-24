"""
L2-2.3 — Model Registry Workflows

Full registry lifecycle: train multiple models on the same dataset,
register them as versions of a single registered model, evaluate each
version on held-out test data, promote the best to "champion", and
demonstrate alias-based model loading for serving.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "L2-wine-classifier"


def train_models(
    X_train, y_train, X_test, y_test
) -> dict[str, dict]:
    """Train three classifiers and log each as its own MLflow run."""
    configs = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

    results: dict[str, dict] = {}
    for name, clf in configs.items():
        with mlflow.start_run(run_name=f"train_{name}") as run:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")

            mlflow.log_param("algorithm", name)
            # Log key hyperparameters (skip internal/nested keys)
            params_to_log = {
                k: str(v)
                for k, v in clf.get_params(deep=False).items()
                if k != "random_state"
            }
            mlflow.log_params(params_to_log)
            mlflow.log_metrics(
                {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}
            )
            mlflow.sklearn.log_model(clf, name="model")

            results[name] = {
                "run_id": run.info.run_id,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
            }
            print(f"  {name:25s}  acc={acc:.4f}  f1={f1:.4f}")

    return results


def register_models(results: dict[str, dict]) -> dict[str, str]:
    """Register each trained model as a new version of MODEL_NAME."""
    versions: dict[str, str] = {}
    for name, info in results.items():
        model_uri = f"runs:/{info['run_id']}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        versions[name] = mv.version
        print(f"  {name:25s} -> {MODEL_NAME} v{mv.version}")
    return versions


def evaluate_versions(
    client: MlflowClient,
    versions: dict[str, str],
    results: dict[str, dict],
) -> None:
    """Log evaluation metrics as version tags for easy comparison."""
    for name, version in versions.items():
        info = results[name]
        client.set_model_version_tag(
            MODEL_NAME, version, "eval_accuracy", f"{info['accuracy']:.4f}"
        )
        client.set_model_version_tag(
            MODEL_NAME, version, "eval_f1", f"{info['f1']:.4f}"
        )
        client.set_model_version_tag(
            MODEL_NAME, version, "eval_precision", f"{info['precision']:.4f}"
        )
        client.set_model_version_tag(
            MODEL_NAME, version, "eval_recall", f"{info['recall']:.4f}"
        )
        print(f"  v{version} ({name}): accuracy={info['accuracy']:.4f}  f1={info['f1']:.4f}")


def promote_best(
    client: MlflowClient,
    versions: dict[str, str],
    results: dict[str, dict],
) -> tuple[str, str]:
    """Set aliases and descriptions on the best two models."""
    ranked = sorted(results.keys(), key=lambda n: results[n]["f1"], reverse=True)
    champion_name, challenger_name = ranked[0], ranked[1]
    champ_ver = versions[champion_name]
    chall_ver = versions[challenger_name]

    # Set aliases
    client.set_registered_model_alias(MODEL_NAME, "champion", champ_ver)
    client.set_registered_model_alias(MODEL_NAME, "challenger", chall_ver)

    # Set model-level description
    client.update_registered_model(
        MODEL_NAME,
        description=(
            "Wine quality classifier trained on the UCI Wine dataset. "
            "Multiple algorithms compared; best promoted to champion."
        ),
    )

    # Set version-level descriptions and tags
    for name, version in versions.items():
        role = "champion" if name == champion_name else (
            "challenger" if name == challenger_name else "archived"
        )
        desc = (
            f"{name} | f1={results[name]['f1']:.4f} | "
            f"accuracy={results[name]['accuracy']:.4f} | role={role}"
        )
        client.update_model_version(MODEL_NAME, version, description=desc)
        client.set_model_version_tag(MODEL_NAME, version, "role", role)
        client.set_model_version_tag(MODEL_NAME, version, "algorithm", name)

    print(f"  champion   -> v{champ_ver} ({champion_name}, f1={results[champion_name]['f1']:.4f})")
    print(f"  challenger -> v{chall_ver} ({challenger_name}, f1={results[challenger_name]['f1']:.4f})")

    return champion_name, challenger_name


def serve_champion(X_test, y_test) -> None:
    """Load the champion model by alias and run predictions."""
    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion_model = mlflow.pyfunc.load_model(champion_uri)

    preds = champion_model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"  Loaded: {champion_uri}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Sample predictions: {[int(p) for p in preds[:8]]}")


def compare_versions(
    client: MlflowClient,
    versions: dict[str, str],
    results: dict[str, dict],
) -> None:
    """Build and display a comparison table of all registered versions."""
    rows = []
    for name, version in versions.items():
        mv = client.get_model_version(MODEL_NAME, version)
        aliases = mv.aliases if hasattr(mv, "aliases") else []
        rows.append({
            "Version": f"v{version}",
            "Algorithm": name,
            "Accuracy": f"{results[name]['accuracy']:.4f}",
            "F1": f"{results[name]['f1']:.4f}",
            "Precision": f"{results[name]['precision']:.4f}",
            "Recall": f"{results[name]['recall']:.4f}",
            "Aliases": ", ".join(aliases) if aliases else "-",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main() -> None:
    """Execute the full model registry workflow."""
    client = MlflowClient()

    # Prepare data
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )

    # Step 1: Train
    print("=" * 70)
    print("Step 1: Train three models on the Wine dataset")
    print("=" * 70)
    results = train_models(X_train, y_train, X_test, y_test)
    print()

    # Step 2: Register
    print("=" * 70)
    print("Step 2: Register all models as versions of", MODEL_NAME)
    print("=" * 70)
    versions = register_models(results)
    print()

    # Step 3: Evaluate / tag versions
    print("=" * 70)
    print("Step 3: Evaluate each version and tag with metrics")
    print("=" * 70)
    evaluate_versions(client, versions, results)
    print()

    # Step 4: Promote
    print("=" * 70)
    print("Step 4: Promote best to champion, runner-up to challenger")
    print("=" * 70)
    champion_name, challenger_name = promote_best(client, versions, results)
    print()

    # Step 5: Serve
    print("=" * 70)
    print("Step 5: Load champion model by alias and predict")
    print("=" * 70)
    serve_champion(X_test, y_test)
    print()

    # Step 6: Compare
    print("=" * 70)
    print("Step 6: Comparison table of all registered versions")
    print("=" * 70)
    compare_versions(client, versions, results)
    print()

    print("=" * 70)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M2_advanced_models/3_registry_workflows")
    main()
