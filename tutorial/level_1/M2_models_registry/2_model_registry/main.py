"""
L1-2.2 — Model Registry

Demonstrates: registering models, listing versions, setting aliases
(champion/challenger), adding descriptions and tags, loading by alias.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "L1-iris-classifier"


def main() -> None:
    # Step 1: Prepare data
    print("=" * 60)
    print("Step 1: Loading the Iris dataset")
    print("=" * 60)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print()

    # Step 2: Train and log two models
    print("=" * 60)
    print("Step 2: Training and logging two models")
    print("=" * 60)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }
    results: dict[str, dict] = {}
    for name, clf in models.items():
        with mlflow.start_run(run_name=name) as run:
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(clf, name="model")
            results[name] = {"run_id": run.info.run_id, "accuracy": acc}
            print(f"  {name:25s}  accuracy={acc:.4f}  run_id={run.info.run_id}")
    print()

    # Step 3: Register both models in the Model Registry
    print("=" * 60)
    print("Step 3: Registering models in the Model Registry")
    print("=" * 60)
    versions: dict[str, str] = {}
    for name, info in results.items():
        model_uri = f"runs:/{info['run_id']}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        versions[name] = mv.version
        print(f"  Registered {name} as {MODEL_NAME} version {mv.version}")
    print()

    # Step 4: Explore registered versions with MlflowClient
    print("=" * 60)
    print("Step 4: Listing registered model versions")
    print("=" * 60)
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for mv in all_versions:
        print(f"  Version {mv.version}  |  run_id={mv.run_id}  |  status={mv.status}")
    print()

    # Step 5: Set aliases -- champion and challenger
    print("=" * 60)
    print("Step 5: Setting aliases (champion / challenger)")
    print("=" * 60)
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    other_name = [n for n in results if n != best_name][0]
    champion_ver = versions[best_name]
    challenger_ver = versions[other_name]

    client.set_registered_model_alias(MODEL_NAME, "champion", champion_ver)
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger_ver)
    print(f"  champion   -> v{champion_ver} ({best_name}, "
          f"acc={results[best_name]['accuracy']:.4f})")
    print(f"  challenger -> v{challenger_ver} ({other_name}, "
          f"acc={results[other_name]['accuracy']:.4f})")
    print()

    # Step 6: Add descriptions and tags
    print("=" * 60)
    print("Step 6: Adding descriptions and tags")
    print("=" * 60)
    client.update_registered_model(
        MODEL_NAME,
        description="Iris flower classifier trained in L1-M2 Model Registry lesson.",
    )
    print(f"  Set model description for '{MODEL_NAME}'")
    for name, version in versions.items():
        desc = f"{name} on Iris (accuracy={results[name]['accuracy']:.4f})"
        client.update_model_version(MODEL_NAME, version, description=desc)
        client.set_model_version_tag(MODEL_NAME, version, "algorithm", name)
        print(f"  Version {version}: description and tag 'algorithm={name}' set")
    print()

    # Step 7: Load champion model and run predictions
    print("=" * 60)
    print("Step 7: Loading champion model by alias and predicting")
    print("=" * 60)
    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion_model = mlflow.sklearn.load_model(champion_uri)
    predictions = champion_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"  Loaded model: {champion_uri}")
    print(f"  Predictions on test set: {predictions[:10]} ...")
    print(f"  Accuracy: {acc:.4f}")
    print()

    print("=" * 60)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M2_models_registry/2_model_registry")
    main()
