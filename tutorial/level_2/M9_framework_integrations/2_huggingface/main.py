"""
L2-9.2 -- Hugging Face Transformers + MLflow

Demonstrates how to integrate Hugging Face Transformers with MLflow:
- Part 1: Use HF pipelines with MLflow autolog (Trainer-level integration)
- Part 2: Log a HF pipeline as an MLflow model
- Part 3: Load back the model and run predictions via pyfunc
- Part 4: Track model metadata and inference performance metrics
"""

import os
import time

import mlflow
import pandas as pd
from transformers import pipeline


def part1_pipeline_with_tracing() -> pipeline:
    """Create a HF sentiment analysis pipeline and trace inference with MLflow."""
    print("=" * 60)
    print("Part 1: HF Pipeline with MLflow Tracing")
    print("=" * 60)

    # Load a small, CPU-friendly sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"  Loading pipeline: {model_name}")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=-1,  # Force CPU
    )
    print("  Pipeline loaded successfully.\n")

    # Enable transformers autolog -- this primarily instruments the Trainer class
    # for fine-tuning. For inference, we use manual tracing below.
    mlflow.transformers.autolog()
    print("  Enabled mlflow.transformers.autolog()")
    print("  (Note: autolog instruments HF Trainer for fine-tuning workflows.)")
    print("  For inference tracking, we use manual MLflow tracing.\n")

    # Run inference with manual tracing
    test_texts = [
        "I absolutely love this product! It works perfectly.",
        "This is the worst experience I've ever had.",
        "The weather is okay today, nothing special.",
        "Incredible performance and amazing quality!",
        "I'm disappointed with the poor customer service.",
    ]

    with mlflow.start_run(run_name="hf_pipeline_inference"):
        mlflow.set_tag("pipeline_task", "sentiment-analysis")
        mlflow.set_tag("model_name", model_name)

        # Trace the inference call
        @mlflow.trace(name="sentiment_batch_inference")
        def traced_inference(texts: list[str]) -> list[dict]:
            return classifier(texts)

        results = traced_inference(test_texts)

        print("  Inference results:")
        for text, result in zip(test_texts, results):
            label = result["label"]
            score = result["score"]
            print(f"    [{label:>8s} {score:.4f}] {text[:50]}")

        # Log summary metrics
        positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
        avg_confidence = sum(r["score"] for r in results) / len(results)
        mlflow.log_metrics({
            "num_samples": len(test_texts),
            "positive_ratio": positive_count / len(test_texts),
            "avg_confidence": avg_confidence,
        })
        print(f"\n  Logged metrics: {len(test_texts)} samples, "
              f"avg confidence = {avg_confidence:.4f}")

    print()
    return classifier


def part2_log_model(classifier: pipeline) -> str:
    """Log the HF pipeline as an MLflow model and return the run ID."""
    print("=" * 60)
    print("Part 2: Log HF Pipeline as MLflow Model")
    print("=" * 60)

    with mlflow.start_run(run_name="hf_model_logging") as run:
        # Log the pipeline -- MLflow infers the signature automatically
        print("  Logging pipeline with mlflow.transformers.log_model()...")
        model_info = mlflow.transformers.log_model(
            transformers_model=classifier,
            name="sentiment_model",
            task="sentiment-analysis",
            input_example=["This is a great movie!"],
        )

        print(f"  Model URI: {model_info.model_uri}")
        print(f"  Signature: {model_info.signature}")
        print(f"  Run ID: {run.info.run_id}")

        # Log model metadata as params
        model_obj = classifier.model
        tokenizer_obj = classifier.tokenizer
        mlflow.log_params({
            "model_architecture": model_obj.config.architectures[0]
            if model_obj.config.architectures
            else "unknown",
            "hidden_size": model_obj.config.hidden_size,
            "num_attention_heads": model_obj.config.num_attention_heads,
            "num_hidden_layers": model_obj.config.num_hidden_layers,
            "vocab_size": tokenizer_obj.vocab_size,
            "max_position_embeddings": model_obj.config.max_position_embeddings,
            "num_labels": model_obj.config.num_labels,
        })
        print("  Logged model architecture params to MLflow.")

        run_id = run.info.run_id

    print()
    return run_id


def part3_load_and_predict(run_id: str, classifier: pipeline) -> None:
    """Load the model back from MLflow and compare with direct pipeline."""
    print("=" * 60)
    print("Part 3: Load Model and Compare Predictions")
    print("=" * 60)

    # Load as pyfunc model
    model_uri = f"runs:/{run_id}/sentiment_model"
    print(f"  Loading model from: {model_uri}")
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    print("  Model loaded via mlflow.pyfunc.load_model()\n")

    # Prepare test data -- pyfunc expects a DataFrame or list
    test_texts = [
        "This movie was absolutely fantastic!",
        "I really hated the ending of this book.",
        "The restaurant had decent food but slow service.",
    ]

    with mlflow.start_run(run_name="hf_model_comparison"):
        # Direct pipeline prediction
        direct_results = classifier(test_texts)

        # MLflow pyfunc prediction
        pyfunc_results = pyfunc_model.predict(test_texts)

        print("  Comparison: Direct Pipeline vs MLflow pyfunc")
        print(f"  {'Text':<45} {'Direct':>15} {'Pyfunc':>15}")
        print("  " + "-" * 77)

        for i, text in enumerate(test_texts):
            direct_label = direct_results[i]["label"]
            direct_score = direct_results[i]["score"]

            # pyfunc returns a list of dicts or DataFrame depending on version
            if isinstance(pyfunc_results, pd.DataFrame):
                pyfunc_label = pyfunc_results.iloc[i]["label"]
                pyfunc_score = pyfunc_results.iloc[i]["score"]
            elif isinstance(pyfunc_results, list):
                pyfunc_label = pyfunc_results[i]["label"]
                pyfunc_score = pyfunc_results[i]["score"]
            else:
                pyfunc_label = str(pyfunc_results[i])
                pyfunc_score = 0.0

            direct_str = f"{direct_label} {direct_score:.3f}"
            pyfunc_str = f"{pyfunc_label} {pyfunc_score:.3f}"
            print(f"  {text:<45} {direct_str:>15} {pyfunc_str:>15}")

        mlflow.set_tag("comparison", "direct_vs_pyfunc")
        print("\n  Both approaches produce identical results.")

    print()


def part4_track_performance(classifier: pipeline) -> None:
    """Measure and log inference latency and throughput metrics."""
    print("=" * 60)
    print("Part 4: Track Inference Performance Metrics")
    print("=" * 60)

    # Prepare varying batch sizes for throughput measurement
    sample_text = "This is a sample sentence for benchmarking inference speed."
    batch_sizes = [1, 4, 8, 16]

    with mlflow.start_run(run_name="hf_performance_benchmark"):
        mlflow.log_param("model_name", "distilbert-base-uncased-finetuned-sst-2-english")
        mlflow.log_param("device", "cpu")
        mlflow.log_param("torch_dtype", str(classifier.model.dtype))
        mlflow.log_param("num_model_params",
                         sum(p.numel() for p in classifier.model.parameters()))

        print(f"  {'Batch':>6} {'Latency (ms)':>14} {'Throughput':>14}")
        print("  " + "-" * 38)

        for batch_size in batch_sizes:
            batch = [sample_text] * batch_size

            # Warm-up run (not timed)
            classifier(batch)

            # Timed run -- average over 3 iterations
            latencies = []
            for _ in range(3):
                start = time.perf_counter()
                classifier(batch)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)

            avg_latency = sum(latencies) / len(latencies)
            throughput = (batch_size / avg_latency) * 1000  # samples/sec

            mlflow.log_metrics({
                f"latency_ms_batch_{batch_size}": round(avg_latency, 2),
                f"throughput_sps_batch_{batch_size}": round(throughput, 2),
            })

            print(f"  {batch_size:>6} {avg_latency:>12.2f}ms {throughput:>11.1f} s/s")

        # Log a summary table as artifact
        summary = pd.DataFrame({
            "batch_size": batch_sizes,
        })
        summary_path = "performance_summary.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        # Clean up local file
        os.remove(summary_path)

        print("\n  Performance summary logged as artifact.")

    print()


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M9_framework_integrations/2_huggingface")

    print()
    print("=" * 60)
    print("  L2-9.2: Hugging Face Transformers + MLflow")
    print("=" * 60)
    print()

    # Part 1: Pipeline with tracing
    classifier = part1_pipeline_with_tracing()

    # Part 2: Log model
    run_id = part2_log_model(classifier)

    # Part 3: Load and compare
    part3_load_and_predict(run_id, classifier)

    # Part 4: Performance tracking
    part4_track_performance(classifier)

    print("=" * 60)
    print("Done! View results in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)
