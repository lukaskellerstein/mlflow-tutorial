"""
L2-9.3 — Sentence Transformers + MLflow

Demonstrates how to log, evaluate, and compare sentence-transformer embedding
models using MLflow: model logging with the native flavor, cosine-similarity
evaluation, semantic search, and a side-by-side comparison with the local
Ollama nomic-embed-text model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M9_framework_integrations/3_sentence_transformers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Return the full pairwise cosine-similarity matrix."""
    return cosine_similarity(embeddings)


def top_k_similar(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Return indices and scores of the k most similar documents."""
    sims = cosine_similarity(query_emb.reshape(1, -1), doc_embs)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_indices]


def get_ollama_embeddings(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray | None:
    """Generate embeddings via a local Ollama model. Returns None on failure."""
    try:
        import ollama
        result = ollama.embed(model=model, input=texts)
        return np.array(result["embeddings"])
    except Exception as exc:
        print(f"  [warning] Ollama embedding failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("L2-9.3 — Sentence Transformers + MLflow")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Part 1: Load a sentence-transformer model and generate embeddings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 1: Load model and generate embeddings")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(model_name)

    sample_sentences = [
        "MLflow is an open-source platform for managing ML lifecycles.",
        "Sentence transformers produce dense vector representations of text.",
        "Cosine similarity measures the angle between two vectors.",
    ]
    embeddings = st_model.encode(sample_sentences)

    print(f"  Model: {model_name}")
    print(f"  Embedding dimensions: {embeddings.shape[1]}")
    print(f"  Number of sentences: {len(sample_sentences)}")
    print(f"  Embedding matrix shape: {embeddings.shape}")
    print()

    with mlflow.start_run(run_name="sentence_transformers_demo") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        print()

        # ------------------------------------------------------------------
        # Part 2: Log the embedding model to MLflow
        # ------------------------------------------------------------------
        print("=" * 60)
        print("Part 2: Log embedding model with mlflow.sentence_transformers")
        print("=" * 60)

        sample_input = "MLflow is great for tracking experiments."
        sample_output = st_model.encode(sample_input)
        signature = mlflow.models.infer_signature(
            model_input=sample_input,
            model_output=sample_output,
        )

        model_info = mlflow.sentence_transformers.log_model(
            model=st_model,
            name="sentence_transformer_model",
            signature=signature,
            input_example=sample_input,
        )

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("embedding_dim", embeddings.shape[1])
        mlflow.log_metric("num_sample_sentences", len(sample_sentences))

        print(f"  Logged model artifact: {model_info.model_uri}")
        print(f"  Signature: {signature}")
        print()

        # ------------------------------------------------------------------
        # Part 3: Evaluate embedding quality — similarity pairs
        # ------------------------------------------------------------------
        print("=" * 60)
        print("Part 3: Evaluate embedding quality (cosine similarity)")
        print("=" * 60)

        similar_pairs = [
            ("The cat sat on the mat.", "A kitten was resting on a rug."),
            ("MLflow tracks experiments.", "MLflow logs metrics and parameters."),
        ]
        dissimilar_pairs = [
            ("The cat sat on the mat.", "The stock market closed higher today."),
            ("MLflow tracks experiments.", "The recipe calls for two eggs."),
        ]

        print("  Similar pairs:")
        for i, (s1, s2) in enumerate(similar_pairs):
            embs = st_model.encode([s1, s2])
            sim = float(cosine_similarity(embs[:1], embs[1:])[0, 0])
            mlflow.log_metric(f"similar_pair_{i}_cosine", round(sim, 4))
            print(f"    [{sim:.4f}] \"{s1}\" <-> \"{s2}\"")

        print("  Dissimilar pairs:")
        for i, (s1, s2) in enumerate(dissimilar_pairs):
            embs = st_model.encode([s1, s2])
            sim = float(cosine_similarity(embs[:1], embs[1:])[0, 0])
            mlflow.log_metric(f"dissimilar_pair_{i}_cosine", round(sim, 4))
            print(f"    [{sim:.4f}] \"{s1}\" <-> \"{s2}\"")
        print()

        # ------------------------------------------------------------------
        # Part 4: Semantic search demo
        # ------------------------------------------------------------------
        print("=" * 60)
        print("Part 4: Semantic search over a small document collection")
        print("=" * 60)

        documents = [
            "MLflow provides experiment tracking for machine learning.",
            "LangChain is a framework for building LLM applications.",
            "Docker containers package software with all dependencies.",
            "Sentence transformers map text to dense vector embeddings.",
            "PostgreSQL is a powerful open-source relational database.",
            "Cosine similarity is commonly used for text similarity tasks.",
            "Python is a popular language for data science and AI.",
            "Kubernetes orchestrates containerized workloads at scale.",
        ]

        doc_embeddings = st_model.encode(documents)
        query = "How do I measure text similarity?"
        query_emb = st_model.encode([query])

        results = top_k_similar(query_emb[0], doc_embeddings, k=3)

        print(f"  Query: \"{query}\"")
        print(f"  Top {len(results)} results:")
        search_rows = []
        for rank, (idx, score) in enumerate(results, start=1):
            print(f"    {rank}. [{score:.4f}] {documents[idx]}")
            search_rows.append({"rank": rank, "score": round(score, 4), "document": documents[idx]})
            mlflow.log_metric(f"search_rank_{rank}_score", round(score, 4))

        search_df = pd.DataFrame(search_rows)
        mlflow.log_table(data=search_df, artifact_file="search_results.json")
        print("  Logged search results table -> search_results.json")
        print()

        # ------------------------------------------------------------------
        # Part 5: Compare with nomic-embed-text (local Ollama)
        # ------------------------------------------------------------------
        print("=" * 60)
        print("Part 5: Compare all-MiniLM-L6-v2 vs nomic-embed-text")
        print("=" * 60)

        comparison_sentences = [
            "Machine learning automates data-driven decisions.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing analyzes human language.",
            "Computer vision enables machines to interpret images.",
        ]

        st_embs = st_model.encode(comparison_sentences)
        st_sim_matrix = compute_pairwise_cosine(st_embs)

        ollama_embs = get_ollama_embeddings(comparison_sentences)

        if ollama_embs is not None:
            ollama_sim_matrix = compute_pairwise_cosine(ollama_embs)

            print(f"  all-MiniLM-L6-v2 dims: {st_embs.shape[1]}")
            print(f"  nomic-embed-text dims: {ollama_embs.shape[1]}")
            mlflow.log_param("comparison_model", "nomic-embed-text")
            mlflow.log_metric("st_embedding_dim", st_embs.shape[1])
            mlflow.log_metric("ollama_embedding_dim", ollama_embs.shape[1])

            comparison_rows = []
            print()
            print(f"  {'Pair':<12} {'MiniLM':>8} {'Nomic':>8} {'Delta':>8}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
            n = len(comparison_sentences)
            pair_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    st_score = float(st_sim_matrix[i, j])
                    ol_score = float(ollama_sim_matrix[i, j])
                    delta = st_score - ol_score
                    label = f"({i},{j})"
                    print(f"  {label:<12} {st_score:>8.4f} {ol_score:>8.4f} {delta:>+8.4f}")
                    comparison_rows.append({
                        "pair": label,
                        "sentence_a": comparison_sentences[i],
                        "sentence_b": comparison_sentences[j],
                        "minilm_cosine": round(st_score, 4),
                        "nomic_cosine": round(ol_score, 4),
                        "delta": round(delta, 4),
                    })
                    mlflow.log_metric(f"minilm_pair_{pair_idx}", round(st_score, 4))
                    mlflow.log_metric(f"nomic_pair_{pair_idx}", round(ol_score, 4))
                    pair_idx += 1

            comp_df = pd.DataFrame(comparison_rows)
            mlflow.log_table(data=comp_df, artifact_file="model_comparison.json")
            print()
            print("  Logged comparison table -> model_comparison.json")
        else:
            print("  Skipping comparison — Ollama / nomic-embed-text not available.")
            print("  To enable: ollama pull nomic-embed-text")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print(f"  Open the MLflow UI at {TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Explore: model artifact, similarity metrics, search results,")
    print("  and the model comparison table in the Artifacts tab.")


if __name__ == "__main__":
    main()
