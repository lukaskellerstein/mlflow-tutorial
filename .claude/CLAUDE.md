# MLFlow Tutorial Project

## Purpose

This is a comprehensive, three-level tutorial for MLFlow covering the full platform — from basic tracking through production AI agent evaluation. The primary focus is **LLMs and AI agents** (not traditional ML training). The special emphasis is on **evaluation (Evals) for AI agents** built with LangChain/LangGraph, DeepAgents, Claude Agent SDK, and Codex SDK.

The tutorial is structured in three progressive levels:
- **Level 1 — Essentials**: Breadth-first. Touch every major MLflow feature (~30 min lessons). Understand the landscape.
- **Level 2 — Practitioner**: Go deeper with real-world projects (~1-2 hour lessons). Build muscle memory.
- **Level 3 — Expert**: Production patterns, custom integrations, advanced agent evaluation. Mastery.

The full syllabus lives in `syllabus.md` (project root) — always consult it for module structure, lesson topics, deliverables, and time estimates before creating or modifying any lesson.

## Technical Stack

- **Python**: 3.10+
- **Package manager**: `uv` (every lesson is a standalone `uv` project)
- **LLM provider**: Ollama (local, no API costs)
- **LLM models**:
  - `gemma4:26b` — large MoE model for complex tasks (evaluation judges, agents)
  - `gemma4:e2b` — small 2B model for simple/fast tasks (basic examples, testing)
  - `nomic-embed-text` — embedding model (137M params, 768 dims) for RAG/vector DB
- **MLFlow**: latest 2.x+
- **Agent frameworks**: LangChain v1.0+, LangGraph (latest), Claude Agent SDK, Codex SDK, DeepAgents
- **Traditional ML** (supporting context only): scikit-learn, XGBoost, PyTorch, Hugging Face Transformers
- **Vector DB**: Qdrant (via Podman Compose)
- **Workflow orchestration**: Temporal.io (via Podman Compose, Level 2)
- **Observability**: Grafana + Prometheus (via Podman Compose, Level 3)
- **Container runtime**: Podman (not Docker)

## Project Layout

```
infra/                          # All infrastructure (Podman Compose)
  compose.yml                   #   Single file to start everything
  mlflow/                       #   MLflow Dockerfile
  temporal/                     #   Temporal config
  grafana/                      #   Grafana provisioning
  prometheus/                   #   Prometheus config
  postgres/                     #   PostgreSQL init script
syllabus.md                     # Master syllabus — the source of truth (project root)
tutorial/
  level_1/                      # Level 1: Essentials (breadth)
    M1_core_platform/
    M2_models_registry/
    M3_autologging/
    M4_evaluation/
    M5_tracing/
    M6_genai_features/
    M7_data_datasets/
    M8_deployment/
    M9_projects/
    M10_auth/
  level_2/                      # Level 2: Practitioner (depth)
    M1_advanced_tracking/
    M2_advanced_models/
    M3_deep_evaluation/
    M4_advanced_tracing/
    M5_agent_observability/
    M6_prompt_engineering/
    M7_ai_gateway/
    M8_deployment/
    M9_framework_integrations/
  level_3/                      # Level 3: Expert (mastery)
    M1_agent_evaluation/
    M2_custom_integrations/
    M3_production/
    M4_advanced_features/
    M5_capstones/
```

Each lesson is a self-contained directory:
```
N_lesson_name/
  pyproject.toml        # uv project — declares dependencies
  main.py               # Working code (the lesson implementation)
  README.md             # Lesson guide with explanation, steps, expected output
  .gitignore            # Ignore .venv, __pycache__, mlruns, mlartifacts
```

## Starting Infrastructure

```bash
cd infra
podman compose up -d
```

This starts MLflow, Temporal, Qdrant, Grafana, Prometheus, PostgreSQL, and Elasticsearch.

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Temporal UI | http://localhost:8080 |
| Qdrant | http://localhost:6333/dashboard |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

Ollama runs natively (not in Podman) for Apple Silicon GPU access.

## Running a Lesson

```bash
cd tutorial/<level>/<module>/<lesson>
uv sync
uv run python main.py
```

## Key Commands

- `podman compose up -d` — start all infrastructure (from `infra/`)
- `podman compose down` — stop all services (preserves data)
- `podman compose down -v` — stop and wipe all data
- `uv init` — scaffold a new lesson project
- `uv add <package>` — add a dependency
- `uv run python main.py` — run the lesson code
- `ollama pull gemma4:e2b` — pull the small LLM
- `ollama pull gemma4:26b` — pull the large MoE LLM
- `ollama pull nomic-embed-text` — pull the embedding model

## Reference Sources

See `.claude/rules/references.md` for the full map of external source code, documentation, and code samples to consult when building lessons.

## Rules

Modular instructions are in `.claude/rules/`. Read them — they cover:
- `tutorial-structure.md` — three-level layout and file conventions
- `coding-standards.md` — Python style for tutorial code
- `mlflow-patterns.md` — MLFlow APIs and patterns to use
- `agent-evaluation.md` — the core focus: agent Evals
- `references.md` — where to find source code, docs, and code samples
- `lesson-content.md` — how to write README.md guides
