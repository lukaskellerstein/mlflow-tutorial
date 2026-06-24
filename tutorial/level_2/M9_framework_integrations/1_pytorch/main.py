"""
L2-9.1 — PyTorch + MLflow

Trains a simple neural network (3-layer MLP) on the Iris dataset using a
custom PyTorch training loop with full MLflow tracking.  Then compares
accuracy against a scikit-learn baseline trained on the same data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow.models import infer_signature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M9_framework_integrations/1_pytorch"

EPOCHS = 40
LEARNING_RATE = 0.01
HIDDEN_SIZE = 32
BATCH_SIZE = 16
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Part 1: Define the neural network
# ---------------------------------------------------------------------------
class IrisNet(nn.Module):
    """Three-layer MLP for Iris classification (4 features -> 3 classes)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_data() -> tuple:
    """Load Iris, scale, split, and convert to PyTorch tensors."""
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, X_train_t, X_test_t, y_train_t, y_test_t


def compute_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Return accuracy (0-1) on the given data."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("L2-9.1 — PyTorch + MLflow")
    print("=" * 60)
    print()

    (X_train_np, X_test_np, y_train_np, y_test_np,
     X_train, X_test, y_train, y_test) = prepare_data()

    # ------------------------------------------------------------------ #
    # Part 2: Training loop with MLflow tracking
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Part 2: Training PyTorch model with MLflow tracking")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    model = IrisNet(input_dim=4, hidden_dim=HIDDEN_SIZE, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="pytorch_iris_mlp") as pytorch_run:
        # Log hyper-parameters
        mlflow.log_params({
            "model": "IrisNet (3-layer MLP)",
            "hidden_size": HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "random_seed": RANDOM_SEED,
        })

        n_samples = X_train.shape[0]
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            # Shuffle each epoch
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, BATCH_SIZE):
                idx = perm[start : start + BATCH_SIZE]
                X_batch, y_batch = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                # Log step-level loss every 10 steps
                if global_step % 10 == 0:
                    mlflow.log_metric("step_loss", loss.item(), step=global_step)

            avg_loss = epoch_loss / n_batches
            train_acc = compute_accuracy(model, X_train, y_train)
            test_acc = compute_accuracy(model, X_test, y_test)

            # Log epoch-level metrics
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"loss={avg_loss:.4f} | "
                      f"train_acc={train_acc:.4f} | "
                      f"test_acc={test_acc:.4f}")

        final_test_acc = compute_accuracy(model, X_test, y_test)
        mlflow.log_metric("final_test_accuracy", final_test_acc)
        print()
        print(f"  Final test accuracy: {final_test_acc:.4f}")
        print()

        # -------------------------------------------------------------- #
        # Part 3: Log the trained model
        # -------------------------------------------------------------- #
        print("=" * 60)
        print("Part 3: Logging model with mlflow.pytorch.log_model()")
        print("=" * 60)

        # Build signature from numpy arrays (what the pyfunc wrapper expects)
        sample_input = pd.DataFrame(X_test_np[:5], columns=[f"f{i}" for i in range(4)])
        model.eval()
        with torch.no_grad():
            sample_output = model(X_test[:5]).numpy()
        signature = infer_signature(sample_input, sample_output)

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="iris_mlp",
            signature=signature,
            input_example=sample_input,
            serialization_format="pickle",
        )
        print(f"  Model logged to: {model_info.model_uri}")
        print()

        # -------------------------------------------------------------- #
        # Part 4: Load and predict
        # -------------------------------------------------------------- #
        print("=" * 60)
        print("Part 4: Loading model and running predictions")
        print("=" * 60)

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        test_df = pd.DataFrame(X_test_np, columns=[f"f{i}" for i in range(4)])
        predictions = loaded_model.predict(test_df)

        # predictions may be a DataFrame or numpy array of logits
        if isinstance(predictions, pd.DataFrame):
            pred_values = predictions.values
        else:
            pred_values = np.asarray(predictions)

        if pred_values.ndim == 2 and pred_values.shape[1] > 1:
            pred_classes = pred_values.argmax(axis=1)
        else:
            pred_classes = pred_values.ravel().astype(int)

        loaded_acc = (pred_classes == y_test_np).mean()
        print(f"  Loaded-model accuracy: {loaded_acc:.4f}")
        print(f"  Sample predictions (first 10): {pred_classes[:10].tolist()}")
        print(f"  Actual labels      (first 10): {y_test_np[:10].tolist()}")
        print()

    # ------------------------------------------------------------------ #
    # Part 5: Compare with scikit-learn
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Part 5: Comparing with scikit-learn LogisticRegression")
    print("=" * 60)

    with mlflow.start_run(run_name="sklearn_iris_logreg"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": 200,
            "random_seed": RANDOM_SEED,
        })

        clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
        clf.fit(X_train_np, y_train_np)

        sklearn_train_acc = clf.score(X_train_np, y_train_np)
        sklearn_test_acc = clf.score(X_test_np, y_test_np)

        mlflow.log_metrics({
            "train_accuracy": sklearn_train_acc,
            "test_accuracy": sklearn_test_acc,
            "final_test_accuracy": sklearn_test_acc,
        })

        mlflow.sklearn.log_model(clf, name="iris_logreg")
        print(f"  sklearn train accuracy: {sklearn_train_acc:.4f}")
        print(f"  sklearn test  accuracy: {sklearn_test_acc:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  {'Model':<30s} {'Test Accuracy':>14s}")
    print(f"  {'-' * 30} {'-' * 14}")
    print(f"  {'PyTorch MLP':<30s} {final_test_acc:>14.4f}")
    print(f"  {'sklearn LogisticRegression':<30s} {sklearn_test_acc:>14.4f}")
    print()
    print(f"  Open the MLflow UI at {TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Compare the two runs side-by-side to see metrics and artifacts.")


if __name__ == "__main__":
    main()
