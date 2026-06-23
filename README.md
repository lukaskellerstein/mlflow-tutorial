# MLFlow Tutorial: From LLMs to AI Agents

A comprehensive, three-level hands-on tutorial for MLFlow — from platform basics through production AI agent evaluation.

## Course Structure

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 — Essentials** | Breadth: every major MLflow feature | 22 lessons | ~10-12 hours |
| **Level 2 — Practitioner** | Depth: real-world projects | 26 lessons | ~25-30 hours |
| **Level 3 — Expert** | Mastery: production agent evaluation | 19 lessons + 2 capstones | ~25-35 hours |

See [tutorial/tutorial_new_syllabus.md](./tutorial/tutorial_new_syllabus.md) for the full syllabus.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Podman](https://podman.io/) + [Podman Compose](https://github.com/containers/podman-compose) (`brew install podman podman-compose`)
- [Ollama](https://ollama.ai/) installed natively (for Apple Silicon GPU access)
- Basic understanding of LLMs and Python async/await

## Setup

### 1. Start Podman machine

```bash
podman machine init
podman machine start
```

### 2. Pull Ollama models (native, not in Podman)

```bash
ollama pull gemma4:e2b          # Small 2B model (Level 1, fast tasks)
ollama pull gemma4:26b          # Large MoE model (Level 2-3, agents, judges)
ollama pull nomic-embed-text    # Embedding model (RAG / vector DB)
```

### 3. Start all infrastructure

```bash
cd infra
podman compose up -d
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Temporal UI | http://localhost:8080 |
| Qdrant | http://localhost:6333/dashboard |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

See [infra/README.md](./infra/README.md) for full details.

### 4. Run a lesson

```bash
cd tutorial/level_1/M1_core_platform/1_architecture_overview
uv sync
uv run python main.py
```

## Technical Stack

- **MLFlow** 2.x+ — tracking, evaluation, tracing, model registry, serving
- **Ollama** — local LLM inference (no API costs)
  - `gemma4:e2b` — small model for simple tasks
  - `gemma4:26b` — large MoE model for complex tasks and LLM-as-judge
  - `nomic-embed-text` — embeddings for RAG
- **LangChain v1.0+ / LangGraph** — agent framework
- **Qdrant** — vector database for RAG examples
- **Claude Agent SDK / Codex SDK / DeepAgents** — custom integration examples (Level 3)
- **Grafana + Prometheus** — production monitoring dashboards (Level 3)
- **Temporal.io** — workflow orchestration (optional, Level 2)
- **Podman Compose** — all infrastructure in one command

## Project Structure

```
infra/                       # All infrastructure (Podman Compose)
  compose.yml                #   Single file to start everything
  mlflow/                    #   MLflow Dockerfile
  temporal/                  #   Temporal config
  grafana/                   #   Grafana provisioning
  prometheus/                #   Prometheus config
  postgres/                  #   PostgreSQL init script
tutorial/
  tutorial_new_syllabus.md   # Full syllabus (source of truth)
  level_1/                   # Essentials — breadth across all features
    M1_core_platform/        #   Tracking, search, system metrics
    M2_models_registry/      #   Models, flavors, registry, PyFunc
    M3_autologging/          #   Traditional ML + LLM autologging
    M4_evaluation/           #   ML eval, LLM eval, LLM-as-judge
    M5_tracing/              #   Auto and manual tracing
    M6_genai_features/       #   Prompts, scorers, judges, datasets
    M7_data_datasets/        #   Dataset logging and lineage
    M8_deployment/           #   Model serving, AI Gateway
    M9_projects/             #   MLflow Projects
    M10_auth/                #   Authentication and permissions
  level_2/                   # Practitioner — depth in each area
    M1_advanced_tracking/    #   Nested runs, async, artifacts, client API
    M2_advanced_models/      #   Signatures, custom PyFunc, registry workflows
    M3_deep_evaluation/      #   Custom metrics, RAG eval, GenAI framework
    M4_advanced_tracing/     #   LangGraph, Temporal, OpenTelemetry
    M5_agent_observability/  #   LangChain/LangGraph agents, multi-agent
    M6_prompt_engineering/   #   Prompt management and optimization
    M7_ai_gateway/           #   Gateway routing and configuration
    M8_deployment/           #   Docker serving, batch prediction
    M9_framework_integrations/ # PyTorch, HuggingFace, Sentence Transformers
  level_3/                   # Expert — mastery and production
    M1_agent_evaluation/     #   Testing, metrics, comparison, optimization
    M2_custom_integrations/  #   Claude SDK, Codex SDK, DeepAgents, autolog
    M3_production/           #   Tracing at scale, Grafana, CI/CD
    M4_advanced_features/    #   Plugins, enterprise, MCP, data management
    M5_capstones/            #   Full production projects
```
