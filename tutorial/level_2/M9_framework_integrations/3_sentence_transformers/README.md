# L2-9.3 — Sentence Transformers + MLflow

**Level:** Practitioner
**Duration:** ~45 minutes

## Overview

Sentence Transformers produce dense vector embeddings that capture semantic meaning, enabling tasks like semantic search, clustering, and text similarity. This lesson shows how to log sentence-transformer models with MLflow's native `sentence_transformers` flavor, evaluate embedding quality with cosine similarity, build a semantic search demo, and compare two embedding models side by side.

## Prerequisites

- Completed: L1-M2.1 (Models & Flavors), L2-M9.2 (Hugging Face)
- MLflow server running at http://127.0.0.1:5000
- Internet access to download `all-MiniLM-L6-v2` on first run (~80 MB)
- (Optional) Ollama with `nomic-embed-text` pulled for the comparison step

## Concepts

### Sentence Transformers

The `sentence-transformers` library (also known as SBERT) fine-tunes transformer models to produce fixed-length embeddings optimized for semantic similarity. Unlike raw BERT embeddings, SBERT embeddings can be compared directly with cosine similarity in a meaningful way.

### MLflow's sentence_transformers Flavor

MLflow provides a first-class `sentence_transformers` flavor with `log_model()`, `save_model()`, and `load_model()`. The flavor automatically captures model weights, configuration, and dependency versions. Models logged this way can be loaded as native SentenceTransformer objects or as generic PyFunc models for serving.

### Embedding Quality Evaluation

Good embeddings place semantically similar texts close together and dissimilar texts far apart. We measure this with cosine similarity on curated pairs of similar and dissimilar sentences.

## Step-by-Step

### Step 1: Load the Model and Generate Embeddings

We load `all-MiniLM-L6-v2`, a small but effective model that maps sentences to 384-dimensional vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "Embedding models are useful"])
print(embeddings.shape)  # (2, 384)
```

### Step 2: Log the Model to MLflow

MLflow's native flavor handles serialization and signature inference:

```python
signature = mlflow.models.infer_signature(
    model_input="Sample text",
    model_output=model.encode("Sample text"),
)
mlflow.sentence_transformers.log_model(
    model=model,
    name="sentence_transformer_model",
    signature=signature,
)
```

### Step 3: Evaluate Similarity on Curated Pairs

Encode pairs of similar and dissimilar sentences, compute cosine similarity, and log the scores as metrics to MLflow.

### Step 4: Semantic Search

Embed a collection of documents plus a query, then rank documents by cosine similarity to the query. The results are logged as a table artifact.

### Step 5: Compare Models

Generate embeddings from both `all-MiniLM-L6-v2` and Ollama's `nomic-embed-text` for the same sentences, compute pairwise cosine similarities, and log a comparison table. This step is skipped gracefully if Ollama is not available.

## Running the Lesson

```bash
cd tutorial/level_2/M9_framework_integrations/3_sentence_transformers
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-9.3 — Sentence Transformers + MLflow
============================================================

============================================================
Part 1: Load model and generate embeddings
============================================================
  Model: all-MiniLM-L6-v2
  Embedding dimensions: 384
  Number of sentences: 3
  Embedding matrix shape: (3, 384)

  Run ID: <run-id>

============================================================
Part 2: Log embedding model with mlflow.sentence_transformers
============================================================
  Logged model artifact: runs:/<run-id>/sentence_transformer_model
  Signature: ...

============================================================
Part 3: Evaluate embedding quality (cosine similarity)
============================================================
  Similar pairs:
    [0.7xxx] "The cat sat on the mat." <-> "A kitten was resting on a rug."
    [0.7xxx] "MLflow tracks experiments." <-> "MLflow logs metrics and parameters."
  Dissimilar pairs:
    [0.0xxx] "The cat sat on the mat." <-> "The stock market closed higher today."
    [0.0xxx] "MLflow tracks experiments." <-> "The recipe calls for two eggs."

============================================================
Part 4: Semantic search over a small document collection
============================================================
  Query: "How do I measure text similarity?"
  Top 3 results:
    1. [0.xxxx] Cosine similarity is commonly used for text similarity tasks.
    2. [0.xxxx] Sentence transformers map text to dense vector embeddings.
    3. [0.xxxx] ...

============================================================
Part 5: Compare all-MiniLM-L6-v2 vs nomic-embed-text
============================================================
  all-MiniLM-L6-v2 dims: 384
  nomic-embed-text dims: 768
  ...comparison table...

============================================================
Done!
============================================================
```

In the MLflow UI you will see:
- The logged `sentence_transformer_model` artifact with full model weights
- Similarity metrics for each pair
- A `search_results.json` table with ranked documents
- A `model_comparison.json` table comparing the two embedding models

## Key Takeaways

- MLflow's `sentence_transformers` flavor lets you log, version, and serve embedding models with full dependency tracking.
- Cosine similarity on curated pairs is a quick way to sanity-check embedding quality.
- Semantic search is a direct application: embed documents once, then retrieve the closest matches for any query.
- Comparing embedding models side by side (dimensions, similarity distributions) helps you pick the right model for your use case.
- All artifacts -- model weights, metrics, result tables -- live together in a single MLflow run for reproducibility.

## Next Steps

Continue to Level 3 for production patterns, or revisit L2-M3 (Deep Evaluation) to apply evaluation techniques to embedding-based pipelines such as RAG.
