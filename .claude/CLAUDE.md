# MLFlow Tutorial Project

## Purpose

A comprehensive, three-level tutorial for MLFlow. The primary focus is **LLMs and AI agents** (not traditional ML training). Special emphasis on **evaluation (Evals) for AI agents** built with LangChain/LangGraph, DeepAgents, and Claude Agent SDK.

The three levels:
- **Level 1 — Essentials**: Breadth-first. Every major MLflow feature (~30 min lessons).
- **Level 2 — Practitioner**: Real-world projects (~1-2 hour lessons).
- **Level 3 — Expert**: Production patterns, custom integrations, advanced agent evaluation.

## Source of Truth

The full syllabus — module structure, lesson topics, deliverables, time estimates — lives in **`syllabus.md`** (project root). Always consult it before creating or modifying any lesson.

## Technical Stack

- **Python**: 3.10+
- **Package manager**: `uv` (every lesson is a standalone `uv` project)
- **LLM provider**: LMStudio (local, no API costs, OpenAI-compatible API)
- **LLM server**: `http://localhost:1234` with OpenAI-compatible endpoint at `/v1/`
- **LLM models**:
  - `google/gemma-4-e4b` — small 4B model for simple/fast tasks (Level 1 lessons)
  - `google/gemma-4-26b-a4b` — large 26B MoE model for complex tasks (Level 2/3, evaluation judges, agents)
  - `text-embedding-nomic-embed-text-v1.5` — embedding model for RAG/vector DB
- **MLFlow**: latest 2.x+
- **Agent frameworks**: LangChain v1.0+, LangGraph (latest), Claude Agent SDK, DeepAgents
- **Vector DB**: Qdrant (via Podman Compose)
- **Workflow orchestration**: Temporal.io (via Podman Compose, Level 2)
- **Observability**: Grafana + Prometheus (via Podman Compose, Level 3)
- **Container runtime**: Podman (not Docker)

## Starting Infrastructure

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

LMStudio runs natively (not in Podman) for Apple Silicon GPU access.

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
- `lms ls` — list available models in LMStudio
- `lms ps` — show loaded models
- `lms load <model>` — load a model
- `lms server start` — start LMStudio server

## Rules

Modular instructions are in `.claude/rules/`. They cover:
- `tutorial-structure.md` — lesson file conventions and principles
- `coding-standards.md` — Python style for tutorial code
- `mlflow-patterns.md` — MLFlow APIs and patterns to use
- `agent-evaluation.md` — the core focus: agent Evals
- `references.md` — where to find source code, docs, and code samples
- `lesson-content.md` — how to write README.md guides
