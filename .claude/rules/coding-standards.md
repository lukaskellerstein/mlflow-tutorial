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

## LLM Setup — always through the LiteLLM gateway

**Every lesson talks to the LiteLLM gateway. Nothing calls LMStudio or
OpenRouter directly.** LMStudio still serves the local models, but it sits
*behind* the gateway — a lesson never names its URL or a raw model key.

That indirection is the point: which model an alias resolves to, the fallback
order when it errors or a prompt overflows, and each model's context window all
live in `infra/litellm/config.yaml`. Changing any of them is a config change, not
an edit to 40 lessons. A lesson that hardcodes `http://localhost:1234/v1` opts
out of all of it and reintroduces exactly the sprawl this replaced.

| Alias | Resolves to | Use for |
|:--|:--|:--|
| `gemma-chat` | LMStudio `google/gemma-4-26b-a4b` | the lesson's own LLM call |
| `gemma-judge` | **OpenRouter** `google/gemma-4-26b-a4b-it` | LLM-as-judge, scorers, simulators (hosted — the local Q4 build loops mid-JSON) |
| `gemma-agent` | LMStudio `google/gemma-4-26b-a4b` | agent loops, tool calling |
| `gemma-tight` | same model, 7168 guard | context-overflow demos |
| `nomic-embed` | LMStudio nomic embeddings | RAG / vector DB |
| `gemma-26b-free` / `gemma-31b-free` | OpenRouter, free tier | sweeps needing a fixed cloud model |
| `frontier` / `gpt-mini` | OpenAI `gpt-5.4-mini` | hosted frontier baseline |

### Direct usage (preferred for most lessons)

```python
from openai import OpenAI

# The LiteLLM gateway from infra/, not a provider directly. The aliases below are
# defined in infra/litellm/config.yaml, which also owns the fallback order and
# each model's context window. Swapping model or provider is a change there,
# never here.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin

client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)

# Small model — fast, for simple tasks and basic examples
response = client.chat.completions.create(
    model="gemma-chat",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)

# The judge that grades it — named separately so the two can diverge later
response = client.chat.completions.create(
    model="gemma-judge",
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
    base_url=GATEWAY_URL,
    api_key=SecretStr(GATEWAY_KEY),
    model="gemma-chat",
    temperature=0.7,
)

# Agent model
llm = ChatOpenAI(
    base_url=GATEWAY_URL,
    api_key=SecretStr(GATEWAY_KEY),
    model="gemma-agent",
    temperature=0.7,
)
```

### Embeddings (RAG / vector DB)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url=GATEWAY_URL,
    api_key=GATEWAY_KEY,
    model="nomic-embed",
    check_embedding_ctx_length=False,
)
```

### Which model to use where

- **The lesson's own LLM call** (the thing under observation): use `gemma-chat`
- **Agent loops and tool calling**: use `gemma-agent`
- **LLM-as-judge, scorers, simulators**: use `gemma-judge`
- **A lesson that runs an agent AND judges it**: name BOTH. They are the same
  model today; the split is what lets them stop being one later.
- **RAG / embeddings**: use `nomic-embed`
- **A sweep comparing configurations**: use a cloud alias (`gemma-26b-free`,
  `gemma-31b-free`). The local aliases carry an error fallback, so an unloaded
  model does not fail the sweep — it silently substitutes a different model, and
  an independent variable that can change without telling you is worse than a
  crash.

### Server-side judges are the exception

A judge started with `scorer.start()` runs **inside the MLflow server**, which
cannot use the constants above — it has neither your base URL nor your key. It
needs an MLflow AI Gateway endpoint, and that endpoint reaches LiteLLM by its
CONTAINER name, `http://litellm:4000/v1`. Two traps, both silent:

- The key is `api_base` inside `auth_config`. `base_url` is not a synonym, and
  an `api_base` in `secret_value` is ignored just as quietly.
- Either mistake sends the request to the provider's own API, surfacing as an
  authentication error about a key you never sent.

`L1-M4.3.1` and `L2-M2.3.1` are the worked examples.

## Error Handling

- Check that the LiteLLM gateway is reachable before making LLM calls.
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
