---
globs: ["tutorial/**/*.py", "mlflow-local/**/*.py"]
---

# Python Coding Standards for Tutorial Code

## Style

- Target Python 3.10+. Use type hints on function signatures.
- Use `if __name__ == "__main__":` guard in every `main.py`.
- Use `asyncio.run(main())` for async lessons (agents, LangGraph).
- Import order: stdlib, third-party, local — separated by blank lines.
- Use f-strings for string formatting.
- Keep functions short and focused — this is tutorial code, readability is paramount.

## Complexity by Level

- **Level 1**: Simple, single-concept scripts. Minimal abstraction. One main function.
- **Level 2**: Multi-step projects. Helper functions OK. Can import from local modules.
- **Level 3**: Production-quality code. Classes, error handling, configuration. Can span multiple files.

## MLFlow Connection

Always set the tracking URI explicitly in code at the top of `main.py`:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

## Ollama / LLM Setup

Use LangChain's ChatOllama for LLM calls:

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="gemma4:e2b", temperature=0.7)
```

When using the LLM directly (without LangChain), use the `ollama` Python package.

## Error Handling

- Wrap Ollama calls with a check that the model is available.
- Print clear error messages if MLFlow server is not running.
- Do not silently swallow exceptions — this is educational code.

## Dependencies

- Use `uv add` to add dependencies, never `pip install`.
- Common dependencies by topic:
  - All lessons: `mlflow`
  - LLM lessons: `langchain-ollama`, `langchain-core`, `langchain`
  - LangGraph lessons: `langgraph`, `langchain-ollama`
  - Evaluation lessons: `pandas`, `mlflow[genai]`
  - RAG lessons: `chromadb`, `langchain-chroma`
  - Traditional ML lessons: `scikit-learn`, `xgboost`
  - PyTorch lessons: `torch`, `pytorch-lightning`
  - HuggingFace lessons: `transformers`, `datasets`
  - Production lessons: `fastapi`, `uvicorn`
  - Monitoring lessons: `prometheus-client`

## Console Output

Print section headers and results so users can follow along:

```python
print("=" * 60)
print("Step 1: Setting up MLFlow experiment")
print("=" * 60)
```

Print key metrics and results inline — don't force users to check the UI for everything.
