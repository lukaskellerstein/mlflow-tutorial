# L2-9.2 -- Hugging Face Transformers + MLflow

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

This lesson demonstrates how to integrate Hugging Face Transformers pipelines with MLflow for model logging, versioning, and inference tracking. You will learn how to log a HF pipeline as an MLflow model, load it back for prediction, and track inference performance metrics -- all without requiring GPU or fine-tuning.

## Prerequisites

- Completed: L1-M2.1 (Models and Flavors), L1-M5.1 (Auto Tracing)
- MLflow server running at http://127.0.0.1:5000
- Internet connection (first run downloads the DistilBERT model, ~260 MB)

## Concepts

### Transformers + MLflow Integration

MLflow provides first-class support for Hugging Face Transformers through the `mlflow.transformers` flavor. This integration allows you to:

- **Log pipelines as models** -- serialize an entire HF pipeline (model + tokenizer + config) as a single MLflow artifact
- **Infer signatures automatically** -- MLflow detects the pipeline's input/output schema
- **Load models via pyfunc** -- use the standard `mlflow.pyfunc.load_model()` interface for framework-agnostic inference
- **Autolog Trainer runs** -- `mlflow.transformers.autolog()` instruments the HF `Trainer` class for fine-tuning workflows

### Pipeline vs Trainer Autolog

The `mlflow.transformers.autolog()` function is designed for **fine-tuning workflows** -- it instruments the HF `Trainer` class to automatically log training metrics, parameters, and checkpoints. For **inference-only** workflows (like this lesson), we use manual MLflow tracing and explicit model logging instead.

### Model Serialization

When you log a HF pipeline with `mlflow.transformers.log_model()`, MLflow saves:
- The model weights
- The tokenizer configuration and vocabulary
- The pipeline task type
- An automatically inferred model signature

This produces a self-contained artifact that can be loaded on any machine with the right dependencies.

## Step-by-Step

### Step 1: Create a HF Pipeline and Trace Inference

We load a small sentiment analysis model (DistilBERT, ~66M parameters) and run inference with MLflow tracing to capture the call.

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # Force CPU
)

# Trace inference with a decorator
@mlflow.trace(name="sentiment_batch_inference")
def traced_inference(texts):
    return classifier(texts)

results = traced_inference(test_texts)
```

### Step 2: Log the Pipeline as an MLflow Model

Use `mlflow.transformers.log_model()` to serialize the entire pipeline. MLflow infers the input/output signature automatically.

```python
model_info = mlflow.transformers.log_model(
    transformers_model=classifier,
    name="sentiment_model",
    task="sentiment-analysis",
    input_example=["This is a great movie!"],
)
```

### Step 3: Load and Compare Predictions

Load the model back using the standard pyfunc interface and verify predictions match.

```python
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/sentiment_model")
pyfunc_results = pyfunc_model.predict(test_texts)
```

Both the direct pipeline and the MLflow-loaded model should produce identical results, confirming the serialization is lossless.

### Step 4: Track Inference Performance

Measure latency and throughput across different batch sizes to understand the model's CPU performance characteristics.

```python
for batch_size in [1, 4, 8, 16]:
    start = time.perf_counter()
    classifier(batch)
    elapsed = (time.perf_counter() - start) * 1000
    mlflow.log_metrics({
        f"latency_ms_batch_{batch_size}": elapsed,
        f"throughput_sps_batch_{batch_size}": batch_size / elapsed * 1000,
    })
```

## Running the Lesson

```bash
cd tutorial/level_2/M9_framework_integrations/2_huggingface
uv sync
uv run python main.py
```

The first run will download the DistilBERT model (~260 MB). Subsequent runs use the cached version.

## Expected Output

```
==========================================================
  L2-9.2: Hugging Face Transformers + MLflow
==========================================================

==========================================================
Part 1: HF Pipeline with MLflow Tracing
==========================================================
  Loading pipeline: distilbert-base-uncased-finetuned-sst-2-english
  Pipeline loaded successfully.
  ...
  Inference results:
    [POSITIVE 0.9999] I absolutely love this product! It works perfectl
    [NEGATIVE 0.9998] This is the worst experience I've ever had.
    ...

==========================================================
Part 2: Log HF Pipeline as MLflow Model
==========================================================
  Logging pipeline with mlflow.transformers.log_model()...
  Model URI: runs:/<run_id>/sentiment_model
  Signature: inputs: [text: string] -> outputs: [label: string, score: double]
  ...

==========================================================
Part 3: Load Model and Compare Predictions
==========================================================
  Comparison: Direct Pipeline vs MLflow pyfunc
  Text                                         Direct          Pyfunc
  ---
  This movie was absolutely fantastic!  POSITIVE 0.999  POSITIVE 0.999
  ...

==========================================================
Part 4: Track Inference Performance Metrics
==========================================================
   Batch   Latency (ms)     Throughput
       1         XX.XXms       XXX.X s/s
       4         XX.XXms       XXX.X s/s
      ...
```

In the MLflow UI at http://127.0.0.1:5000, you will see:
- Four runs under the "L2/M9_framework_integrations/2_huggingface" experiment
- The model artifact with the full pipeline serialized
- Inference traces showing the sentiment analysis calls
- Performance metrics across different batch sizes

## Key Takeaways

- **`mlflow.transformers.log_model()`** serializes an entire HF pipeline (model + tokenizer + config) as a single MLflow artifact with automatic signature inference.
- **`mlflow.transformers.autolog()`** is designed for HF Trainer fine-tuning, not inference. For inference tracking, use manual tracing with `@mlflow.trace`.
- **pyfunc loading** provides a framework-agnostic inference interface -- the loaded model behaves identically to the original pipeline.
- **Performance tracking** with batch-size sweeps helps you understand latency/throughput tradeoffs for deployment planning.
- Model metadata (architecture, hidden size, vocab size) logged as params makes it easy to compare different model configurations in the MLflow UI.

## Next Steps

Continue to **L2-9.3 (Sentence Transformers)** to learn how to log and serve embedding models with MLflow, which is essential for RAG pipelines and semantic search.
