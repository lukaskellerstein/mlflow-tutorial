# Reference Sources

Always consult these sources when building lessons. Do NOT guess at APIs — read the source code and docs first.

## MLFlow

- **Source code**: `~/Projects/Github/mlflow/mlflow`
  - Python API: `~/Projects/Github/mlflow/mlflow/mlflow/` (the inner `mlflow/` package)
  - GenAI module: `~/Projects/Github/mlflow/mlflow/mlflow/genai/`
  - LangChain integration: `~/Projects/Github/mlflow/mlflow/mlflow/langchain/`
  - Evaluation: `~/Projects/Github/mlflow/mlflow/mlflow/metrics/`
  - Tracing: `~/Projects/Github/mlflow/mlflow/mlflow/tracing/`
- **Documentation**: `~/Projects/Github/mlflow/mlflow/docs/docs`

## LangChain

- **Source code**: `~/Projects/Github/langchain-ai/langchain`
- **Examples**: `~/Projects/Github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain`
  - Agents: `.../10_agent/`
- v1.0+ API only: use `create_agent` from `langchain.agents` (no legacy chains)
- Verify that APIs used in reference code are still current

## LangGraph

- **Source code**: `~/Projects/Github/langchain-ai/langgraph`
- **Examples**: `~/Projects/Github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph`
  - Agents: `.../5_agent/`
  - Multi-agent: `.../6_agents/`

## DeepAgents

- **Source code**: `~/Projects/Github/langchain-ai/deepagents`
- **Examples**: `~/Projects/Github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/3_deepagents`

## Claude Agent SDK

- **Source code**: `~/Projects/Github/anthropics/claude-agent-sdk-python`
- **Examples**: `~/Projects/Github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`

## Temporal.io

- **Examples**: `~/Projects/Github/lukaskellerstein/my-workflows/temporal-io/my-python`
  - AI workflows: `.../MY/5_AI/`
- **Docker reference**: `~/Projects/Github/lukaskellerstein/my-workflows/temporal-io/docker`

## LMStudio

- **CLI docs**: <https://lmstudio.ai/docs/cli>
- **Headless mode**: <https://lmstudio.ai/docs/developer/core/headless>
- **Model**: Gemma4-E4B — <https://lmstudio.ai/models/google/gemma-4-e4b>

## Evaluation

- **SWE-Bench**: <https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified>

## Qdrant (Vector DB)

- Used for RAG lessons, runs via Podman Compose in `infra/`

## Infrastructure

- All services run via `podman compose up -d` from `infra/`
- MLflow: <http://localhost:5555> (PostgreSQL backend)
- LMStudio: <http://localhost:1234> (native, not in Podman)
- Temporal: <http://localhost:8080> (UI), localhost:7233 (gRPC)
- Qdrant: <http://localhost:6333> (REST), localhost:6334 (gRPC)
- Grafana: <http://localhost:3000> (admin/admin)
- Prometheus: <http://localhost:9090>

## How to Use References

1. **Before implementing a lesson**: read the corresponding reference code to understand patterns
2. **Before using an API**: check the MLFlow source/docs to verify it exists and get the correct signature
3. **Adapt, don't copy**: reference code is inspiration — tutorial code should be clean, self-contained, and educational
4. **Check for deprecations**: LangChain APIs change frequently. Always verify against current docs
5. **Cross-reference**: when building an agent eval lesson, check both the agent framework reference AND the MLFlow eval reference
