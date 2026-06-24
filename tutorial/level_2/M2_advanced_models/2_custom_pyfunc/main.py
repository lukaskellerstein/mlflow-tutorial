"""
L2-M2.2 — Custom PyFunc Models

Demonstrates advanced custom PyFunc patterns:
- Part 1: PythonModel with load_context() for artifact loading
- Part 2: PyFunc with params support (runtime-configurable LLM calls)
- Part 3: Multi-model ensemble wrapped as a single PyFunc
"""

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M2_advanced_models/2_custom_pyfunc"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Part 1: PythonModel with load_context()
# ---------------------------------------------------------------------------

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    """A custom PyFunc that loads a sklearn model from artifacts in load_context()."""

    def load_context(self, context) -> None:
        import joblib

        model_path = context.artifacts["sklearn_model"]
        self.model = joblib.load(model_path)
        print(f"  [load_context] Loaded sklearn model from {model_path}")

    def predict(self, context, model_input, params=None):
        predictions = self.model.predict(model_input)
        return predictions


def part1_load_context() -> None:
    section("Part 1: PythonModel with load_context()")

    # Train a sklearn model
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    print(f"  Trained RandomForest — accuracy: {accuracy:.4f}")

    # Save the sklearn model to a temp file
    import joblib

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "rf_model.joblib"
        joblib.dump(rf, model_path)
        print(f"  Saved sklearn model to {model_path}")

        # Log as a custom PyFunc with artifact dependency
        with mlflow.start_run(run_name="part1_load_context"):
            mlflow.log_param("wrapper_type", "SklearnWrapper")
            mlflow.log_param("underlying_model", "RandomForestClassifier")
            mlflow.log_metric("training_accuracy", accuracy)

            signature = infer_signature(X_test, rf.predict(X_test))

            model_info = mlflow.pyfunc.log_model(
                name="sklearn_wrapper",
                python_model=SklearnWrapper(),
                artifacts={"sklearn_model": str(model_path)},
                signature=signature,
            )
            print(f"  Logged custom PyFunc model: {model_info.model_uri}")

            # Load it back and predict
            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            test_df = pd.DataFrame(X_test[:5], columns=[f"f{i}" for i in range(4)])
            preds = loaded.predict(test_df)
            print(f"  Predictions on 5 samples: {preds}")
            print(f"  Expected:                 {y_test[:5]}")


# ---------------------------------------------------------------------------
# Part 2: PyFunc with params support
# ---------------------------------------------------------------------------

class LLMTextProcessor(mlflow.pyfunc.PythonModel):
    """A PyFunc that calls Ollama with runtime-configurable params."""

    def predict(self, context, model_input, params=None):
        import ollama

        params = params or {}
        temperature = params.get("temperature", 0.7)
        style = params.get("style", "concise")

        results = []
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = [str(model_input)]

        for text in texts:
            prompt = f"Rewrite the following text in a {style} style:\n\n{text}"
            response = ollama.chat(
                model="gemma4:e2b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            result = response["message"]["content"]
            results.append(result)

        return results


def part2_params_support() -> None:
    section("Part 2: PyFunc with params support")

    with mlflow.start_run(run_name="part2_params_support"):
        mlflow.log_param("model_class", "LLMTextProcessor")
        mlflow.log_param("llm_model", "gemma4:e2b")

        input_example = pd.DataFrame({"text": ["The quick brown fox jumps."]})
        signature = infer_signature(
            input_example,
            ["A fox leapt swiftly."],
            params={"temperature": 0.7, "style": "concise"},
        )

        model_info = mlflow.pyfunc.log_model(
            name="text_processor",
            python_model=LLMTextProcessor(),
            signature=signature,
        )
        print(f"  Logged LLMTextProcessor model: {model_info.model_uri}")

        # Load and test with different params
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        test_input = pd.DataFrame(
            {"text": ["MLflow is an open source platform for the ML lifecycle."]}
        )

        print("\n  --- Test 1: concise style, temperature=0.3 ---")
        result1 = loaded.predict(
            test_input, params={"temperature": 0.3, "style": "concise"}
        )
        print(f"  Result: {result1[0][:120]}...")

        print("\n  --- Test 2: formal style, temperature=0.9 ---")
        result2 = loaded.predict(
            test_input, params={"temperature": 0.9, "style": "formal"}
        )
        print(f"  Result: {result2[0][:120]}...")

        mlflow.log_metric("num_styles_tested", 2)


# ---------------------------------------------------------------------------
# Part 3: Multi-model ensemble as a single PyFunc
# ---------------------------------------------------------------------------

class EnsembleModel(mlflow.pyfunc.PythonModel):
    """A PyFunc that loads multiple sklearn models and averages their predictions."""

    def load_context(self, context) -> None:
        import joblib

        manifest_path = context.artifacts["manifest"]
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.models = {}
        for name in self.manifest["model_names"]:
            path = context.artifacts[name]
            self.models[name] = joblib.load(path)
            print(f"  [load_context] Loaded model '{name}' from {path}")

        print(f"  [load_context] Ensemble ready with {len(self.models)} models")

    def predict(self, context, model_input, params=None):
        # Collect probability predictions from each model
        all_probas = []
        for name, model in self.models.items():
            probas = model.predict_proba(model_input)
            all_probas.append(probas)

        # Average the probabilities and take argmax
        avg_probas = np.mean(all_probas, axis=0)
        return np.argmax(avg_probas, axis=1)


def part3_ensemble() -> None:
    section("Part 3: Multi-model ensemble as a single PyFunc")

    # Prepare data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train three different models
    models = {
        "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=50, random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        import joblib

        artifacts = {}
        individual_accuracies = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            individual_accuracies[name] = acc
            print(f"  Trained {name:25s} — accuracy: {acc:.4f}")

            path = Path(tmp_dir) / f"{name}.joblib"
            joblib.dump(model, path)
            artifacts[name] = str(path)

        # Save manifest with model names
        manifest = {"model_names": list(models.keys())}
        manifest_path = Path(tmp_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        artifacts["manifest"] = str(manifest_path)

        # Log ensemble as a single PyFunc
        with mlflow.start_run(run_name="part3_ensemble"):
            for name, acc in individual_accuracies.items():
                mlflow.log_metric(f"{name}_accuracy", acc)
            mlflow.log_param("ensemble_models", list(models.keys()))
            mlflow.log_param("ensemble_strategy", "probability_averaging")

            signature = infer_signature(
                X_test, models["random_forest"].predict(X_test)
            )

            model_info = mlflow.pyfunc.log_model(
                name="ensemble_model",
                python_model=EnsembleModel(),
                artifacts=artifacts,
                signature=signature,
            )
            print(f"\n  Logged ensemble model: {model_info.model_uri}")

            # Load and test
            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(4)])
            ensemble_preds = loaded.predict(test_df)

            ensemble_acc = np.mean(ensemble_preds == y_test)
            mlflow.log_metric("ensemble_accuracy", ensemble_acc)

            print(f"\n  Individual model accuracies:")
            for name, acc in individual_accuracies.items():
                print(f"    {name:25s} {acc:.4f}")
            print(f"    {'ENSEMBLE':25s} {ensemble_acc:.4f}")

            print(f"\n  Sample predictions (first 10):")
            print(f"    Ensemble:  {ensemble_preds[:10]}")
            print(f"    Actual:    {y_test[:10]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    part1_load_context()
    part2_params_support()
    part3_ensemble()

    section("Done!")
    print(f"Open MLflow UI at {TRACKING_URI}")
    print(f"Look for experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
