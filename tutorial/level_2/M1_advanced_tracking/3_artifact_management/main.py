"""
L2-1.3 — Artifact Management Deep Dive

Demonstrates how to log various artifact types in MLflow: images (PIL and
numpy), tables (pandas DataFrames), matplotlib figures, text/JSON files,
organized subdirectories, and bulk directory uploads.
"""

import json
import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import mlflow

matplotlib.use("Agg")  # non-interactive backend — no GUI windows

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M1_advanced_tracking/3_artifact_management"


def create_sample_plot() -> tuple[plt.Figure, np.ndarray]:
    """Create a matplotlib figure and return both the figure and a numpy image."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), label="sin(x)", linewidth=2)
    ax.plot(x, np.cos(x), label="cos(x)", linewidth=2)
    ax.set_title("Trigonometric Functions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Render figure to numpy array (RGBA -> RGB, uint8)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img_array = rgba[:, :, :3].copy()  # drop alpha channel

    return fig, img_array


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("L2-1.3 — Artifact Management Deep Dive")
    print("=" * 60)
    print()

    with mlflow.start_run(run_name="artifact_management_demo") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        print()

        # --------------------------------------------------------------
        # Part 1: Log images with mlflow.log_image()
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 1: Logging images (PIL and numpy)")
        print("=" * 60)

        # 1a — PIL Image
        pil_img = Image.new("RGB", (200, 200), color=(30, 144, 255))
        mlflow.log_image(pil_img, artifact_file="images/pil_sample.png")
        print("  Logged PIL image  -> images/pil_sample.png")

        # 1b — Numpy array image (from matplotlib render)
        fig, img_array = create_sample_plot()
        mlflow.log_image(img_array, artifact_file="images/numpy_plot.png")
        print("  Logged numpy image -> images/numpy_plot.png")
        print()

        # --------------------------------------------------------------
        # Part 2: Log tables with mlflow.log_table()
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 2: Logging tables (pandas DataFrame)")
        print("=" * 60)

        results_df = pd.DataFrame(
            {
                "model": ["RandomForest", "GradientBoosting", "SVM", "LogisticRegression"],
                "accuracy": [0.92, 0.95, 0.88, 0.85],
                "f1_score": [0.91, 0.94, 0.87, 0.84],
                "train_time_sec": [12.3, 45.6, 78.1, 2.4],
            }
        )
        mlflow.log_table(data=results_df, artifact_file="tables/model_comparison.json")
        print("  Logged DataFrame   -> tables/model_comparison.json")
        print(f"  Table shape: {results_df.shape[0]} rows x {results_df.shape[1]} columns")
        print()

        # --------------------------------------------------------------
        # Part 3: Log figures with mlflow.log_figure()
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 3: Logging matplotlib figures directly")
        print("=" * 60)

        # 3a — Reuse the trig figure
        mlflow.log_figure(fig, "figures/trig_functions.png")
        print("  Logged figure      -> figures/trig_functions.png")

        # 3b — A bar chart
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(results_df["model"], results_df["accuracy"], color="steelblue")
        ax2.set_title("Model Accuracy Comparison")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0.7, 1.0)
        ax2.tick_params(axis="x", rotation=25)
        fig2.tight_layout()
        mlflow.log_figure(fig2, "figures/accuracy_bars.png")
        print("  Logged figure      -> figures/accuracy_bars.png")
        plt.close("all")
        print()

        # --------------------------------------------------------------
        # Part 4: Log text and JSON artifacts with mlflow.log_artifact()
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 4: Logging text and JSON artifacts")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 4a — JSON config file
            config = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "optimizer": "adam",
                "architecture": "transformer",
            }
            config_path = os.path.join(tmp_dir, "training_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            mlflow.log_artifact(config_path, artifact_path="config")
            print("  Logged JSON        -> config/training_config.json")

            # 4b — Plain text notes
            notes = (
                "Experiment Notes\n"
                "================\n"
                "- Ran on 2026-06-23\n"
                "- Dataset: synthetic classification (10k samples)\n"
                "- Best model: GradientBoosting with 0.95 accuracy\n"
            )
            notes_path = os.path.join(tmp_dir, "experiment_notes.txt")
            with open(notes_path, "w") as f:
                f.write(notes)
            mlflow.log_artifact(notes_path, artifact_path="docs")
            print("  Logged text        -> docs/experiment_notes.txt")
        print()

        # --------------------------------------------------------------
        # Part 5: Organize artifacts in subfolders
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 5: Organizing artifacts in subfolders")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Data preprocessing config
            preprocess_cfg = {"scaling": "standard", "missing": "median", "features": 40}
            pp_path = os.path.join(tmp_dir, "preprocessing.json")
            with open(pp_path, "w") as f:
                json.dump(preprocess_cfg, f, indent=2)
            mlflow.log_artifact(pp_path, artifact_path="config/preprocessing")
            print("  Logged -> config/preprocessing/preprocessing.json")

            # Model hyperparameters
            hyper_cfg = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}
            hp_path = os.path.join(tmp_dir, "hyperparameters.json")
            with open(hp_path, "w") as f:
                json.dump(hyper_cfg, f, indent=2)
            mlflow.log_artifact(hp_path, artifact_path="config/model")
            print("  Logged -> config/model/hyperparameters.json")
        print()

        # --------------------------------------------------------------
        # Part 6: Log a directory of files with mlflow.log_artifacts()
        # --------------------------------------------------------------
        print("=" * 60)
        print("Part 6: Logging an entire directory (log_artifacts)")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create several report files
            for split in ["train", "validation", "test"]:
                report = {
                    "split": split,
                    "n_samples": {"train": 8000, "validation": 1000, "test": 1000}[split],
                    "accuracy": round(np.random.uniform(0.85, 0.96), 4),
                    "loss": round(np.random.uniform(0.05, 0.30), 4),
                }
                report_path = os.path.join(tmp_dir, f"{split}_report.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

            mlflow.log_artifacts(tmp_dir, artifact_path="reports")
            print("  Logged directory   -> reports/")
            print("    - reports/train_report.json")
            print("    - reports/validation_report.json")
            print("    - reports/test_report.json")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("Artifacts logged in this run:")
    print("  images/pil_sample.png                      (PIL image)")
    print("  images/numpy_plot.png                      (numpy array image)")
    print("  tables/model_comparison.json               (pandas DataFrame table)")
    print("  figures/trig_functions.png                  (matplotlib figure)")
    print("  figures/accuracy_bars.png                   (matplotlib figure)")
    print("  config/training_config.json                 (JSON via log_artifact)")
    print("  docs/experiment_notes.txt                   (text via log_artifact)")
    print("  config/preprocessing/preprocessing.json     (nested subfolder)")
    print("  config/model/hyperparameters.json            (nested subfolder)")
    print("  reports/train_report.json                   (directory upload)")
    print("  reports/validation_report.json              (directory upload)")
    print("  reports/test_report.json                    (directory upload)")
    print()
    print(f"  Open the MLflow UI at {TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Click on the run and browse the 'Artifacts' tab to explore")
    print("  the folder structure and preview images, tables, and files.")


if __name__ == "__main__":
    main()
