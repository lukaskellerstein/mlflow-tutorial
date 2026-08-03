"""
L1-M3.2 — Custom PyFunc Models

Demonstrates wrapping a RAG pipeline as a custom MLflow PyFunc model:
- PythonModel with load_context() for initializing LLM and vector DB
- In-memory Qdrant vector store with sample documents
- predict() that retrieves relevant docs and generates answers
- Runtime params support (temperature, top_k)
"""

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature
from openai import OpenAI

TRACKING_URI = "http://127.0.0.1:5555"
EXPERIMENT_NAME = "L1/M3_models_registry/2_custom_pyfunc"

# Sample documents about MLflow for the RAG knowledge base
DOCUMENTS = [
    "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for experiment tracking, model packaging, versioning, and deployment.",
    "MLflow Tracking allows you to log parameters, metrics, and artifacts during ML experiments. Each experiment contains runs, and each run records the inputs, outputs, and metadata of a training session.",
    "The MLflow Model Registry provides a central model store for versioning and stage transitions. Teams use it to manage model lifecycle from development through staging to production.",
    "MLflow Models use a standard format that supports multiple deployment tools. A model directory contains an MLmodel file describing flavors, a conda.yaml for dependencies, and the serialized model artifacts.",
    "MLflow Projects package data science code for reproducible runs on any platform. A project is a directory or Git repo with an MLproject file specifying the entry points, parameters, and environment.",
    "MLflow Evaluate provides tools for evaluating LLM and traditional ML models. It supports built-in metrics like toxicity and readability, plus custom metrics and LLM-as-judge evaluation.",
    "MLflow Tracing captures detailed execution traces for LLM applications. Traces show each step in a chain or agent workflow, including inputs, outputs, and latency for every span.",
    "MLflow supports autologging for many frameworks including scikit-learn, PyTorch, and LangChain. Autologging automatically captures parameters, metrics, and model artifacts without manual log calls.",
]


class RAGModel(mlflow.pyfunc.PythonModel):
    """A RAG pipeline wrapped as a single MLflow PyFunc model."""

    def load_context(self, context) -> None:
        """Initialize LLM client, embeddings, and Qdrant vector store."""
        from langchain_openai import OpenAIEmbeddings
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # Load config from artifact
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(
            model=self.config["embedding_model"],
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            check_embedding_ctx_length=False,
        )

        # Initialize Qdrant in-memory
        self.qdrant = QdrantClient(":memory:")

        # Load and index documents from artifact
        docs_path = context.artifacts["documents"]
        with open(docs_path) as f:
            documents = json.load(f)

        # Create collection and add docs
        vectors = self.embeddings.embed_documents(documents)
        vector_size = len(vectors[0])
        self.qdrant.create_collection(
            collection_name="knowledge_base",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vec, payload={"text": doc}) for i, (vec, doc) in enumerate(zip(vectors, documents))
        ]
        self.qdrant.upsert(collection_name="knowledge_base", points=points)
        print(f"  [load_context] Indexed {len(documents)} documents")

    def predict(self, context, model_input, params=None):
        """Retrieve relevant docs and generate answers for each query."""

        params = params or {}
        temperature = params.get("temperature", 0.7)
        top_k = params.get("top_k", 3)

        client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
        )

        results = []
        if isinstance(model_input, pd.DataFrame):
            queries = model_input["query"].tolist()
        else:
            queries = [str(model_input)]

        for query in queries:
            # Retrieve relevant docs
            query_vector = self.embeddings.embed_query(query)
            search_results = self.qdrant.query_points(
                collection_name="knowledge_base",
                query=query_vector,
                limit=top_k,
            )
            context_text = "\n".join([(p.payload or {})["text"] for p in search_results.points])

            # Generate answer with retrieved context
            response = client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question based on the following context. "
                            "Be concise and accurate.\n\n"
                            f"Context:\n{context_text}"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=temperature,
            )
            results.append(response.choices[0].message.content or "")

        return results


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("Step 1: Preparing RAG model artifacts")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save config
        config = {
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "llm_model": "google/gemma-4-e4b",
            "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        }
        config_path = Path(tmp_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Save documents
        docs_path = Path(tmp_dir) / "documents.json"
        with open(docs_path, "w") as f:
            json.dump(DOCUMENTS, f)

        print(f"  Config saved: LLM={config['llm_model']}")
        print(f"  Documents saved: {len(DOCUMENTS)} entries")

        # Log the RAG model
        print("\n" + "=" * 60)
        print("Step 2: Logging RAG model as PyFunc")
        print("=" * 60)

        input_example = pd.DataFrame({"query": ["What is MLflow?"]})
        signature = infer_signature(
            input_example,
            ["MLflow is an open-source platform..."],
            params={"temperature": 0.7, "top_k": 3},
        )

        with mlflow.start_run(run_name="rag_pyfunc_model") as run:
            mlflow.log_param("llm_model", config["llm_model"])
            mlflow.log_param("embedding_model", config["embedding_model"])
            mlflow.log_param("num_documents", len(DOCUMENTS))
            mlflow.log_param("vector_db", "qdrant_in_memory")

            model_info = mlflow.pyfunc.log_model(
                name="rag_model",
                python_model=RAGModel(),
                artifacts={
                    "config": str(config_path),
                    "documents": str(docs_path),
                },
                signature=signature,
                pip_requirements=[
                    "openai",
                    "langchain-openai",
                    "qdrant-client",
                ],
            )
            print(f"  Logged model: {model_info.model_uri}")
            print(f"  Run ID: {run.info.run_id}")

    # Load and test (outside the temp dir -- artifacts are in MLflow now)
    print("\n" + "=" * 60)
    print("Step 3: Loading and testing the RAG model")
    print("=" * 60)

    loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    test_queries = [
        "What is MLflow Tracking?",
        "How does the Model Registry work?",
        "What is MLflow Tracing?",
    ]

    for query in test_queries:
        single_df = pd.DataFrame({"query": [query]})
        result = loaded.predict(single_df)
        print(f"\n  Query: {query}")
        print(f"  Answer: {result[0][:200]}...")

    # Test with custom params
    print("\n" + "=" * 60)
    print("Step 4: Testing with custom params")
    print("=" * 60)

    single_df = pd.DataFrame({"query": ["What does MLflow do?"]})
    result = loaded.predict(single_df, params={"temperature": 0.2, "top_k": 2})
    print("\n  Query: What does MLflow do? (temperature=0.2, top_k=2)")
    print(f"  Answer: {result[0][:200]}...")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Open MLflow UI at {TRACKING_URI}")
    print(f"Look for experiment: {EXPERIMENT_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
