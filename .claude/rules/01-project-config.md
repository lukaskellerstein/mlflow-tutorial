---
description: Project configuration — architecture, paths, dev environment
---

# Project Config

- **Project**: MLFlow Tutorial — a three-level MLflow course aimed at LLMs and AI
  agents, with evaluation (Evals) of agents as the core focus.
- **Architecture**: 43 independently runnable lesson leaves under `tutorial/`,
  plus one shared podman-compose infrastructure stack under `infra/`. There is
  **no** uv workspace tying the leaves together — that is deliberate.
- **Structure**:
  - `tutorial/level_1_models/` — tracking, tracing, registry, evaluation,
    prompts, deployment/gateway, finetuning (M1–M7)
  - `tutorial/level_2_agents/` — agent frameworks, custom integrations, agent
    evaluation, benchmarks (M1–M4)
  - `tutorial/level_3_advanced/` — production, advanced tracing, extensibility,
    capstones (M1–M4)
  - `infra/` — `compose.yml` and per-service config for the whole stack
  - `syllabus.md` — the source of truth for module/lesson structure
  - `GOAL.md` — what the tutorial is for
- **Build**: none. There is no build step and no packaging; each lesson is run
  directly from source.
- **Run locally**: `cd tutorial/<level>/<module>/<lesson> && uv sync && uv run python main.py`
- **Test**: there is no test suite. Verification is running the lesson's
  `main.py` end to end against a live MLflow server and LMStudio, then checking
  the run appears in the MLflow UI. See `06-testing.md`.
- **Key dependencies**: `mlflow>=3.0`, `openai>=1.0`, LangChain v1.0+, LangGraph,
  Claude Agent SDK, DeepAgents, pandas, Qdrant, Temporal
- **Package manager**: `uv` — one project, one `.venv` and one `uv.lock` per leaf

## Leaves

This repo is **not one project**. Each lesson below is independently runnable and
keeps its own environment — no workspaces, by design.

| Level | Modules | Lessons |
|:--|:--|--:|
| `level_1_models` | M1 tracking, M2 tracing, M3 models/registry, M4 evaluation, M5 prompt engineering, M6 deployment/gateway, M7 finetuning | 21 |
| `level_2_agents` | M1 agent frameworks, M2 custom integrations, M3 agent evaluation, M4 agent benchmarks | 14 |
| `level_3_advanced` | M1 production, M2 advanced tracing, M3 extensibility, M4 capstones | 8 |

Every leaf carries its own `pyproject.toml`, `.venv` and `uv.lock`. The repo-root
`pyrightconfig.json` holds one `executionEnvironments` entry per leaf — **re-run
`gen-pyrightconfig.py` from mac-setup when a leaf is added**, or the new lesson
resolves its imports against nothing.

## Services and ports

Started with `podman compose up -d` from `infra/`.

| Service | URL | Notes |
|:--|:--|:--|
| MLflow UI | http://localhost:5555 | tracking server + artifact store |
| Temporal UI | http://localhost:8080 | Level 3 only |
| Qdrant | http://localhost:6333/dashboard | vector DB for RAG lessons |
| Grafana | http://localhost:3000 | admin/admin, Level 3 |
| Prometheus | http://localhost:9090 | Level 3 |
| LMStudio | http://localhost:1234/v1/ | **not** in podman — runs natively for Apple Silicon GPU access |

`infra/.env` is local-only and untracked; the committed `infra/.env.example`
carries the variable names and the localhost-only development defaults
(`admin`/`admin` and friends). First-time setup: `cp .env.example .env` from
`infra/`. Do not put a real secret in either file — see `12-security.md`.
