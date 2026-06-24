# L2-9.1 — PyTorch + MLflow

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

PyTorch gives you full control over the training loop, but that freedom means you also need to instrument tracking yourself. This lesson shows how to train a vanilla `torch.nn.Module` with a custom training loop while logging every detail to MLflow: hyper-parameters, step-level and epoch-level metrics, the trained model with a signature, and inference from a reloaded model. A scikit-learn baseline trained on the same data provides context for the results.

## Prerequisites

- Completed: L1-M1.2 (Tracking Basics), L1-M2.1 (Models & Flavors)
- MLflow server running at http://127.0.0.1:5000
- Basic familiarity with PyTorch (`nn.Module`, optimizers, loss functions)

## Concepts

### Why manual tracking with PyTorch?

Unlike PyTorch Lightning, vanilla PyTorch does not have a `Trainer` that MLflow can hook into automatically. The `mlflow.pytorch.autolog()` function targets Lightning; for a raw training loop you call `mlflow.log_metric()`, `mlflow.log_param()`, and `mlflow.pytorch.log_model()` yourself. This is actually an advantage in a tutorial setting because you see exactly what gets tracked and when.

### The `mlflow.pytorch` flavor

MLflow's PyTorch flavor supports:

| API | Purpose |
|-----|---------|
| `mlflow.pytorch.log_model()` | Log a `torch.nn.Module` as an MLflow artifact |
| `mlflow.pytorch.load_model()` | Load back the native PyTorch model |
| `mlflow.pytorch.save_model()` | Save to a local directory (no run required) |
| `mlflow.pytorch.autolog()` | Automatic logging for **PyTorch Lightning** only |

When you log a PyTorch model, MLflow serializes the model weights and records the conda/pip environment so the model is reproducible. Providing a `signature` is strongly recommended — it tells downstream consumers (model serving, batch inference) what input shape and dtypes the model expects.

### Model signatures for tensors

PyTorch models expect tensors, but MLflow signatures describe columnar data (DataFrames, numpy arrays). The `mlflow.pyfunc` wrapper automatically converts DataFrame input to tensors for you. Use `infer_signature()` with numpy or DataFrame examples to create the signature.

## Step-by-Step

### Step 1: Define the network

A three-layer MLP using `nn.Sequential`: input (4 features) -> hidden (32 units, ReLU) -> hidden (32 units, ReLU) -> output (3 classes).

```python
class IrisNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

### Step 2: Train with MLflow tracking

Inside `mlflow.start_run()`, log hyper-parameters up front with `log_params()`, then log loss and accuracy at every epoch. Step-level loss is logged every 10 training steps to build a fine-grained loss curve.

```python
with mlflow.start_run(run_name="pytorch_iris_mlp"):
    mlflow.log_params({"hidden_size": 32, "learning_rate": 0.01, ...})

    for epoch in range(1, EPOCHS + 1):
        # ... training loop ...
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("test_accuracy", test_acc, step=epoch)
```

### Step 3: Log the model

After training, log the model with a signature and an input example. The signature is inferred from a sample DataFrame input and the corresponding model output.

```python
signature = infer_signature(sample_input_df, sample_output_np)
mlflow.pytorch.log_model(
    pytorch_model=model,
    name="iris_mlp",
    signature=signature,
    input_example=sample_input_df,
)
```

### Step 4: Reload and predict

Load the model back through the `pyfunc` interface and run predictions on the test set. The pyfunc wrapper handles DataFrame-to-tensor conversion automatically.

```python
loaded = mlflow.pyfunc.load_model(model_uri)
predictions = loaded.predict(test_dataframe)
```

### Step 5: Compare with scikit-learn

Train a `LogisticRegression` on the same data in a separate MLflow run. Both runs live under the same experiment, so you can compare them in the UI.

## Running the Lesson

```bash
cd tutorial/level_2/M9_framework_integrations/1_pytorch
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-9.1 — PyTorch + MLflow
============================================================

============================================================
Part 2: Training PyTorch model with MLflow tracking
============================================================
  Epoch   1/40 | loss=1.0912 | train_acc=0.3333 | test_acc=0.3333
  Epoch  10/40 | loss=0.5624 | train_acc=0.8917 | test_acc=0.9000
  Epoch  20/40 | loss=0.2187 | train_acc=0.9583 | test_acc=0.9667
  Epoch  30/40 | loss=0.1105 | train_acc=0.9750 | test_acc=0.9667
  Epoch  40/40 | loss=0.0732 | train_acc=0.9833 | test_acc=1.0000

  Final test accuracy: 1.0000

============================================================
Part 3: Logging model with mlflow.pytorch.log_model()
============================================================
  Model logged to: runs:/<run-id>/iris_mlp

============================================================
Part 4: Loading model and running predictions
============================================================
  Loaded-model accuracy: 1.0000
  Sample predictions (first 10): [2, 0, 1, 0, ...]
  Actual labels      (first 10): [2, 0, 1, 0, ...]

============================================================
Part 5: Comparing with scikit-learn LogisticRegression
============================================================
  sklearn train accuracy: 0.9750
  sklearn test  accuracy: 0.9667

============================================================
Comparison Summary
============================================================
  Model                          Test Accuracy
  ------------------------------ --------------
  PyTorch MLP                          1.0000
  sklearn LogisticRegression           0.9667
```

Exact numbers may vary slightly depending on platform. In the MLflow UI, navigate to the experiment and use the chart view to see the epoch-level loss and accuracy curves for the PyTorch run.

## Key Takeaways

- For vanilla PyTorch (not Lightning), use manual `mlflow.log_metric()` and `mlflow.log_param()` calls inside your training loop. The `autolog()` function is designed for PyTorch Lightning only.
- Use `mlflow.pytorch.log_model()` with a `signature` and `input_example` to make the model self-documenting and servable.
- The `mlflow.pyfunc` wrapper converts DataFrame inputs to tensors automatically, so you can serve PyTorch models through the standard MLflow model serving API.
- Log metrics with the `step` parameter to build loss curves and track convergence over time.
- Running both PyTorch and scikit-learn models under the same experiment makes side-by-side comparison straightforward in the MLflow UI.

## Next Steps

Continue to **L2-9.2 — Hugging Face Transformers + MLflow** to learn how to log and serve pre-trained transformer models with `mlflow.transformers`. Where this lesson used a tiny dataset and a custom loop, the next lesson works with Hugging Face's `Trainer` and autologging.
