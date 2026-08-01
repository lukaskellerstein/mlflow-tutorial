---
globs: ["tutorial/**/*.py"]
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

- **Level 1 (Models)**: End-to-end topic coverage. Helper functions OK. Merged lessons may use multiple parts.
- **Level 2 (Agents)**: Agent-specific projects. Assumes L1 knowledge. Can import from local modules.
- **Level 3 (Advanced)**: Production-quality code. Classes, error handling, configuration. Can span multiple files.

## MLFlow Connection

Always set the tracking URI explicitly in code at the top of `main.py`:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5555")
```

## LMStudio / LLM Setup

Models are served locally via LMStudio with an OpenAI-compatible API.

### Direct usage (preferred for most lessons)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Small model — fast, for simple tasks and basic examples
response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)

# Large MoE model — for complex tasks, evaluation judges, agents
response = client.chat.completions.create(
    model="google/gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)
```

### With LangChain (for agent/chain lessons)

`ChatOpenAI.api_key` is typed `SecretStr | None`. Pydantic coerces a plain string
at runtime, but the type checker rejects it — so wrap the key in `SecretStr` and
declare `pydantic>=2` in the lesson's `pyproject.toml`. (The plain string form
stays correct for `openai.OpenAI`, which accepts `str`.)

```python
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Small model
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="google/gemma-4-e4b",
    temperature=0.7,
)

# Large MoE model
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("lm-studio"),
    model="google/gemma-4-26b-a4b",
    temperature=0.7,
)
```

### Embeddings (RAG / vector DB)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,
)
```

### Which model to use where
- **Level 1 lessons**: use `google/gemma-4-e4b` (fast, keeps lessons snappy)
- **Level 2/3 agent and evaluation lessons**: use `google/gemma-4-26b-a4b` (better reasoning)
- **LLM-as-judge / evaluation judges**: use `google/gemma-4-26b-a4b` (judge quality matters)
- **RAG / embeddings**: use `text-embedding-nomic-embed-text-v1.5`

## Error Handling

- Check that LMStudio server is reachable before making LLM calls.
- Print clear error messages if MLFlow server is not running.
- Do not silently swallow exceptions — this is educational code.

## Dependencies

- Use `uv add` to add dependencies, never `pip install`.
- Common dependencies by topic:
  - All lessons: `mlflow`
  - LLM lessons (direct): `openai`
  - LLM lessons (LangChain): `langchain-openai`, `langchain-core`, `langchain`
  - LangGraph lessons: `langgraph`, `langchain-openai`
  - DeepAgents lessons: `deepagents`, `langchain-openai`
  - Evaluation lessons: `pandas`, `mlflow[genai]`
  - RAG lessons: `qdrant-client`, `langchain-qdrant`
  - Fine-tuning lessons: `transformers`, `datasets`
  - Production lessons: `fastapi`, `uvicorn`
  - Monitoring lessons: `prometheus-client`
- Any lesson that calls `mlflow.langchain.autolog()` must declare the `langchain`
  meta-package, even if the code only imports `langchain_openai`/`langgraph`.
  MLflow's version check imports `langchain` itself, so without it autolog raises
  `ModuleNotFoundError: No module named 'langchain'`.
- Declare every module the lesson imports directly. Do not rely on a package
  arriving transitively (e.g. `numpy` via `pandas`, `requests` via `mlflow`).

## Console Output

Print section headers and results so users can follow along:

```python
print("=" * 60)
print("Step 1: Setting up MLFlow experiment")
print("=" * 60)
```

Print key metrics and results inline — don't force users to check the UI for everything.
