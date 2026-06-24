"""L1-9.1 — MLflow Projects: reproducible ML workflows via MLproject files."""

import os
import textwrap

import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M9_projects/1_mlflow_projects")

LESSON_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_file(name: str, content: str) -> None:
    with open(os.path.join(LESSON_DIR, name), "w") as f:
        f.write(content)
    print(f"  Wrote {name}")


def part1_what_is_mlflow_project() -> None:
    print("=" * 60)
    print("Part 1: What is an MLflow Project?")
    print("=" * 60)
    print(textwrap.dedent("""
        An MLflow Project is a directory (or Git repo) containing:
          1. MLproject file  - entry points & parameters
          2. Environment spec - python_env.yaml / conda.yaml / Dockerfile
          3. Code files       - training / inference scripts
        Benefits: reproducibility, parameterization, portability.
    """))


def part2_create_project_files() -> None:
    print("=" * 60)
    print("Part 2: Creating MLflow Project Files")
    print("=" * 60)
    mlproject = textwrap.dedent("""\
        name: iris-training
        python_env: python_env.yaml
        entry_points:
          main:
            parameters:
              n_estimators: {type: int, default: 100}
              max_depth: {type: int, default: 5}
            command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth}"
    """)
    _write_file("MLproject", mlproject)
    print(f"\n{textwrap.indent(mlproject, '    ')}")
    _write_file("python_env.yaml", textwrap.dedent("""\
        python: "3.10"
        build_dependencies: [pip]
        dependencies: [mlflow>=2.0, scikit-learn>=1.0]
    """))
    _write_file("train.py", textwrap.dedent("""\
        import argparse, mlflow
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--n-estimators", type=int, default=100)
            p.add_argument("--max-depth", type=int, default=5)
            a = p.parse_args()
            X, y = load_iris(return_X_y=True)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
            clf = RandomForestClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth)
            clf.fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            mlflow.log_params({"n_estimators": a.n_estimators, "max_depth": a.max_depth})
            mlflow.log_metric("accuracy", acc)
            print(f"Accuracy: {acc:.4f}")
        if __name__ == "__main__":
            main()
    """))


def part3_show_project_commands() -> None:
    print("\n" + "=" * 60)
    print("Part 3: Running MLflow Projects")
    print("=" * 60)
    print(textwrap.dedent("""
        Run locally:    mlflow run . -P n_estimators=200 -P max_depth=10
        Run from Git:   mlflow run https://github.com/<user>/<repo>.git
        Skip env setup: mlflow run . --env-manager local
        Environment types: python_env (virtualenv), conda, docker
    """))


def part4_run_training_directly() -> None:
    print("=" * 60)
    print("Part 4: Running the Training Directly")
    print("=" * 60)
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = {"n_estimators": 150, "max_depth": 6}
    with mlflow.start_run(run_name="iris_project_demo"):
        mlflow.log_params(params)
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        mlflow.sklearn.log_model(clf, name="model")
        print(f"  Params:   {params}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Run ID:   {mlflow.active_run().info.run_id}")
        print("\n  Model and metrics logged. View at http://127.0.0.1:5000")


def main() -> None:
    part1_what_is_mlflow_project()
    part2_create_project_files()
    part3_show_project_commands()
    part4_run_training_directly()
    print("\n" + "=" * 60)
    print("Lesson complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
