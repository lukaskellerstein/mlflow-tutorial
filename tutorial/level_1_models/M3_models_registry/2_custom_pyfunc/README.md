# L1-M3.2 — Custom PyFunc Models

**Level:** Practitioner
**Duration:** ~1 hour

## Overview

This lesson demonstrates how to wrap a complete RAG (Retrieval-Augmented Generation) pipeline as a single MLflow PyFunc model. You will build a `PythonModel` subclass that initializes an embedding model, populates an in-memory Qdrant vector store, and generates answers by retrieving relevant context before calling an LLM. The result is a self-contained, deployable model artifact that bundles the entire RAG workflow behind a standard `predict()` interface.

## Prerequisites

- Completed: L1-M3.1 (Models and Flavors — PyFunc basics)
- MLFlow server running at <http://127.0.0.1:5555>
- LMStudio running with `google/gemma-4-e4b` and `text-embedding-nomic-embed-text-v1.5` models loaded

## Concepts

### Why Custom PyFunc?

MLflow's built-in flavors (sklearn, pytorch, etc.) cover single-model use cases, but real-world LLM applications are multi-component systems. A RAG pipeline combines an embedding model, a vector database, retrieval logic, prompt construction, and an LLM -- none of which fit neatly into a single built-in flavor. Wrapping the entire pipeline as a custom PyFunc gives you:

- **Single deployment unit** -- one model artifact that includes all components
- **Standard interface** -- callers use `predict()` without knowing about the internals
- **Artifact management** -- MLflow stores and restores all configuration and data files
- **Runtime configurability** -- `params` let callers adjust behavior (temperature, number of retrieved docs) without relogging

### Key Methods

| Method | When it runs | Purpose |
|---|---|---|
| `load_context(self, context)` | Once, when the model is loaded | Initialize heavy resources: LLM clients, embedding models, vector stores. Access bundled files via `context.artifacts`. |
| `predict(self, context, model_input, params=None)` | Every prediction request | Run the RAG pipeline: embed the query, retrieve docs, generate an answer. Accept runtime overrides via `params`. |

### The artifacts Dict

When you call `mlflow.pyfunc.log_model(..., artifacts={"config": "/path/to/config.json", "documents": "/path/to/docs.json"})`, MLflow copies each file into the model artifact store. At load time, `context.artifacts["config"]` returns the local path to the restored file. This is how the RAG model gets its configuration and document corpus without hardcoding paths.

### Runtime Params for LLM Configuration

The `params` argument to `predict()` lets callers control LLM behavior at inference time:

- **`temperature`** -- controls randomness of the generated answer (default: 0.7)
- **`top_k`** -- number of documents to retrieve from the vector store (default: 3)

The params schema is captured in the model signature, so MLflow can validate inputs at serving time.

## Step-by-Step

### Step 1: Prepare RAG Artifacts

Before logging the model, we create two JSON files that will be bundled as artifacts:

- **`config.json`** -- LLM endpoint URL, API key, model names for the LLM and embedding model
- **`documents.json`** -- the document corpus to index in the vector store

```python
config = {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "llm_model": "google/gemma-4-e4b",
    "embedding_model": "text-embedding-nomic-embed-text-v1.5",
}
```

Externalizing configuration into an artifact (rather than hardcoding) makes the model portable -- you can change the LLM endpoint without rebuilding the model.

### Step 2: Define the RAGModel PyFunc

The `RAGModel` class subclasses `mlflow.pyfunc.PythonModel` and implements two methods:

**`load_context()`** runs once when the model is loaded. It:
1. Reads the config and document artifacts
2. Initializes the `OpenAIEmbeddings` client for embedding queries and documents
3. Creates an in-memory Qdrant collection
4. Embeds all documents and inserts them into the vector store

```python
class RAGModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Read config from artifact
        config_path = context.artifacts["config"]
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize embeddings and Qdrant, index documents
        ...
```

**`predict()`** runs on every request. For each query it:
1. Embeds the query using the same embedding model
2. Retrieves the top-k most similar documents from Qdrant
3. Constructs a prompt with the retrieved context
4. Calls the LLM and returns the generated answer

### Step 3: Log and Load the Model

We log the model with `mlflow.pyfunc.log_model()`, passing:
- `python_model=RAGModel()` -- the model class
- `artifacts={"config": ..., "documents": ...}` -- files to bundle
- `signature` -- input/output schema with params
- `pip_requirements` -- Python packages needed at load time

```python
model_info = mlflow.pyfunc.log_model(
    name="rag_model",
    python_model=RAGModel(),
    artifacts={"config": str(config_path), "documents": str(docs_path)},
    signature=signature,
    pip_requirements=["openai", "langchain-openai", "qdrant-client"],
)
```

After logging, we load the model back with `mlflow.pyfunc.load_model()` and run test queries. The `load_context()` method fires automatically, rebuilding the vector store from the bundled artifacts.

### Step 4: Test with Custom Params

We call `predict()` with custom `params` to demonstrate runtime configurability:

```python
result = loaded.predict(
    pd.DataFrame({"query": ["What does MLflow do?"]}),
    params={"temperature": 0.2, "top_k": 2},
)
```

Lower temperature produces more deterministic answers. Fewer retrieved documents (top_k=2) focuses the context on the most relevant matches.

## Running the Lesson

```bash
cd tutorial/level_2/M2_advanced_models/2_custom_pyfunc
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
Step 1: Preparing RAG model artifacts
============================================================
  Config saved: LLM=google/gemma-4-e4b
  Documents saved: 8 entries

============================================================
Step 2: Logging RAG model as PyFunc
============================================================
  [load_context] Indexed 8 documents
  Logged model: runs:/.../rag_model
  Run ID: abc123...

============================================================
Step 3: Loading and testing the RAG model
============================================================
  [load_context] Indexed 8 documents

  Query: What is MLflow Tracking?
  Answer: MLflow Tracking allows you to log parameters, metrics, and artifacts...

  Query: How does the Model Registry work?
  Answer: The MLflow Model Registry provides a central model store...

  Query: What is MLflow Tracing?
  Answer: MLflow Tracing captures detailed execution traces for LLM applications...

============================================================
Step 4: Testing with custom params
============================================================

  Query: What does MLflow do? (temperature=0.2, top_k=2)
  Answer: MLflow is an open-source platform for managing the end-to-end...

============================================================
Done!
Open MLflow UI at http://127.0.0.1:5555
Look for experiment: L1/M3_models_registry/2_custom_pyfunc
============================================================
```

In the MLflow UI you will see:
- One run named `rag_pyfunc_model` with logged parameters (llm_model, embedding_model, num_documents, vector_db)
- The `rag_model` artifact containing the serialized PyFunc, config.json, and documents.json

## Key Takeaways

- **`PythonModel`** lets you wrap arbitrary multi-component systems (like a RAG pipeline) as a single deployable MLflow model.
- **`load_context()`** is the right place to initialize heavy resources (LLM clients, vector stores, embedding models) -- it runs once at load time, not on every prediction.
- **The `artifacts` dict** bundles configuration files and data with the model. MLflow copies them into the artifact store and restores them automatically on load.
- **Runtime `params`** (temperature, top_k) let callers tune model behavior at inference time without relogging or redeploying.
- **In-memory Qdrant** works well for demos and small corpora. For production, replace with a persistent Qdrant instance and store only connection config in the artifact.

## Next Steps

Continue to L2-M2.3 (Registry Workflows) to learn how to manage model lifecycle stages, aliases, and promotion workflows in the MLflow Model Registry.
