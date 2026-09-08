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

## LLM Setup — always through the MLflow AI Gateway

**Every lesson talks to the MLflow AI Gateway. Nothing calls Unsloth directly.**
There is no separate proxy container: the gateway IS the MLflow tracking server,
at `http://127.0.0.1:5555/gateway/mlflow/v1`. Unsloth Studio serves every model,
but it sits *behind* the gateway — a lesson never names its URL or a raw model
key.

That indirection is the point: which model an alias resolves to lives in
`infra/mlflow/gateway/seed_gateway.py`. Changing it is a config change, not an
edit to 50 lessons. A lesson that hardcodes `http://localhost:8888/v1` opts out
of all of it and reintroduces exactly the sprawl this replaced.

| Alias | Resolves to | Use for |
|:--|:--|:--|
| `gemma-chat` | Unsloth `gemma-4-26B-A4B-it-qat` | the lesson's own LLM call |
| `gemma-judge` | Unsloth `gemma-4-26B-A4B-it-qat` | LLM-as-judge, scorers, simulators |
| `gemma-agent` | Unsloth `gemma-4-26B-A4B-it-qat` | agent loops, tool calling |
| `gemma-tight` | same model | context-overflow demos — **the guard is gone**, see below |
| `gemma-31b-local` | Unsloth `gemma-4-31B-it-qat` | the denser local model |
| `nomic-embed` | Unsloth `Nomic-embed-text-v1.5` | RAG / vector DB |
| `text-embedding-3-small` | Unsloth `Nomic-embed-text-v1.5` | MLflow's judge aligner, which hardcodes this name |
| `gpt-4.1-mini` | Unsloth `gemma-4-26B-A4B-it-qat` | MLflow's aligner chat model, likewise hardcoded |

That is the whole list. **Eight aliases, three models, one provider, and no
fallback anywhere.** There is no OpenRouter alias, no OpenAI alias, and no
hosted model to escape to — so a lesson that cannot reach Unsloth fails and
names the cause instead of quietly answering from something else. Nothing here
needs the network or costs anything.

**Unsloth holds ONE model at a time.** `Settings → API → Model auto-switch` must
be on, or every alias but the loaded one fails with `400 ... 'Switch model by
request' is off`. With it on, a swap costs 4–14 s.

### What this gateway cannot do

Four things LiteLLM did have no equivalent here. Do not write a lesson that
assumes them:

- **No `max_input_tokens` pre-call check.** `gemma-tight` cannot guard a prompt
  before the call; overflow fails at the model.
- **No context-window fallbacks.** MLflow falls back on ERROR only.
- **No `drop_params`.** Every parameter is forwarded exactly as sent. This is now
  a feature, not a loss: Unsloth honours `response_format` correctly where
  LMStudio compiled a decoding grammar that masked EOS and ran to `max_tokens`.
- **No OpenAI-shaped embeddings route.** `POST /gateway/mlflow/v1/embeddings`
  answers 404. The only alias-addressed embedding route is
  `POST /gateway/<alias>/mlflow/invocations`, which the OpenAI SDK cannot drive —
  see `L1-M3.2` for the fifteen-line bridge and `L2-M2.1.1.2` for the one call in
  the tutorial that has to go around the gateway entirely.

### Direct usage (preferred for most lessons)

```python
from openai import OpenAI

# The MLflow AI Gateway -- the tracking server itself, not a provider directly.
# The aliases below are defined in infra/mlflow/gateway/seed_gateway.py, which also
# owns the fallback order. Swapping model or provider is a change there, never
# here.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all

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

**`OpenAIEmbeddings(base_url=GATEWAY_URL)` does not work.** There is no
`/gateway/mlflow/v1/embeddings` — it answers 404. The only route that takes an
alias puts it in the PATH, which the OpenAI SDK cannot drive, so post to it
directly:

```python
import json
import urllib.request

MLFLOW_URL = "http://127.0.0.1:5555"


def embed(texts: list[str], alias: str = "nomic-embed") -> list[list[float]]:
    request = urllib.request.Request(
        f"{MLFLOW_URL}/gateway/{alias}/mlflow/invocations",
        data=json.dumps({"input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return [row["embedding"] for row in json.load(response)["data"]]
```

`L1-M3.2` wraps that in an `embed_documents` / `embed_query` pair so a vector
store can use it. Copy that shape rather than reaching for a client library.

### Which model to use where

- **The lesson's own LLM call** (the thing under observation): use `gemma-chat`
- **Agent loops and tool calling**: use `gemma-agent`
- **LLM-as-judge, scorers, simulators**: use `gemma-judge`
- **A lesson that runs an agent AND judges it**: name BOTH. They are the same
  model today; the split is what lets them stop being one later.
- **RAG / embeddings**: use `nomic-embed`
- **A sweep comparing models**: use `gemma-agent` and `gemma-31b-local`. They
  are the two distinct local models, and neither can substitute for the other
  behind your back — an independent variable that changes without telling you is
  worse than a crash. Unsloth pays a 4–14 s auto-switch each time the sweep
  crosses between them.

### Server-side judges are the exception

A judge started with `scorer.start()` runs **inside the MLflow server**, which
cannot use the constants above — it samples its own traces on its own schedule,
long after your script has exited, so it has no base URL to borrow. It names a
gateway ENDPOINT instead:

```python
scorer = scorer_cls(name=..., model="gateway:/gemma-judge")
```

`openai:/gemma-judge` registers happily and then refuses to start, because an
`openai:/` model is resolved client-side and the server has no client.

Nothing has to be built. Every alias in `infra/mlflow/gateway/seed_gateway.py` IS a
gateway endpoint, seeded on every `podman compose up -d` — which is the main
simplification this gateway bought. `L1-M4.3.1` and `L2-M2.3.1` are the worked
examples, and both now only *check* that the endpoint is there.

## Error Handling

- Check that the gateway is reachable before making LLM calls. It is the MLflow
  server, so `GET http://127.0.0.1:5555/health` answers for both — and it is
  unauthenticated and exempt from the Host header check, so it works before
  anything else is configured. `/v1/models` does NOT exist here.
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
