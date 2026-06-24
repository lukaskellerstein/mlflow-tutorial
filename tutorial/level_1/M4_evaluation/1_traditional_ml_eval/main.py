"""
L1-4.1 — Traditional ML Evaluation

Demonstrates mlflow.models.evaluate() for traditional ML models:
- Evaluating a classifier with built-in metrics (accuracy, precision, recall, F1)
- Accessing evaluation artifacts (confusion matrix, ROC curves)
- Comparing two models side by side using evaluate()
"""

import pandas as pd
import mlflow
import mlflow.models
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def evaluate_model(
    model, model_name: str, X_train, y_train,
    eval_df: pd.DataFrame, feature_names: list[str],
) -> dict:
    """Train, log, and evaluate a model. Returns the metrics dict."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        # Log the trained model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)
        print(f"  Model logged: {model_info.model_uri}")

        # Evaluate with mlflow.models.evaluate()
        result = mlflow.models.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="classifier",
            feature_names=feature_names,
            evaluators="default",
            evaluator_config={"log_model_explainability": False},
        )

        # Print metrics
        print(f"\n  Metrics for {model_name}:")
        for name, value in sorted(result.metrics.items()):
            if isinstance(value, float):
                print(f"    {name:>25s}: {value:.4f}")
            else:
                print(f"    {name:>25s}: {value}")

        # List artifacts
        print(f"\n  Artifacts for {model_name}:")
        for name, artifact in result.artifacts.items():
            print(f"    {name}: {type(artifact).__name__}")

        return result.metrics


def main() -> None:
    # Step 1: Load data and prepare evaluation DataFrame
    print("=" * 60)
    print("Step 1: Loading the Wine dataset")
    print("=" * 60)
    wine = load_wine()
    feature_names = list(wine.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42,
    )
    # mlflow.models.evaluate() expects a DataFrame with features + target
    eval_df = pd.DataFrame(X_test, columns=feature_names)
    eval_df["target"] = y_test
    print(f"  Training samples: {len(X_train)}")
    print(f"  Evaluation samples: {len(X_test)}")
    print(f"  Classes: {list(wine.target_names)}\n")

    # Step 2: Evaluate a RandomForest classifier
    print("=" * 60)
    print("Step 2: Evaluating RandomForest classifier")
    print("=" * 60)
    rf_metrics = evaluate_model(
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "random_forest", X_train, y_train, eval_df, feature_names,
    )
    print()

    # Step 3: Evaluate a GradientBoosting classifier
    print("=" * 60)
    print("Step 3: Evaluating GradientBoosting classifier")
    print("=" * 60)
    gb_metrics = evaluate_model(
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "gradient_boosting", X_train, y_train, eval_df, feature_names,
    )
    print()

    # Step 4: Compare the two models side by side
    print("=" * 60)
    print("Step 4: Model comparison")
    print("=" * 60)
    compare_keys = [
        "accuracy_score", "f1_score", "precision_score",
        "recall_score", "log_loss", "roc_auc",
    ]
    print(f"  {'Metric':<25s} {'RandomForest':>14s} {'GradientBoost':>14s}")
    print(f"  {'-' * 25} {'-' * 14} {'-' * 14}")
    for key in compare_keys:
        rf_val, gb_val = rf_metrics.get(key), gb_metrics.get(key)
        if not isinstance(rf_val, float) or not isinstance(gb_val, float):
            continue
        # For log_loss lower is better; for all others higher is better
        better_rf = (rf_val < gb_val) if key == "log_loss" else (rf_val > gb_val)
        rf_mark = " *" if better_rf else ""
        gb_mark = " *" if not better_rf and rf_val != gb_val else ""
        print(f"  {key:<25s} {rf_val:>14.4f}{rf_mark}  {gb_val:>10.4f}{gb_mark}")

    print(f"\n  (* = better)\n")
    print("=" * 60)
    print("Done! View evaluation results in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("  Look for artifacts: confusion_matrix, ROC curves, PR curves")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M4_evaluation/1_traditional_ml_eval")
    main()
