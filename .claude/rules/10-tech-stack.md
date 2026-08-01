---
description: "Reference: Technology stack — Python 3.10+/uv, MLflow 3.x, LangChain/LangGraph/Claude Agent SDK, podman compose"
---

# Reference: Technology Stack

## Backend

- **Language**: Python 3.10+ (leaves currently resolve to 3.12)
- **Framework**: none — lessons are plain scripts with a `main()`, not a service
- **Package manager**: `uv`. One project, one `.venv`, one `uv.lock` per leaf.
  No workspace, deliberately — every lesson must run standalone.
- **MLflow**: 3.x. Lessons pin `mlflow>=3.0`; the server image is
  `ghcr.io/mlflow/mlflow:latest`.
- **LLM provider**: LMStudio — local, no API cost, OpenAI-compatible at
  `http://localhost:1234/v1/`. Runs natively rather than in a container so it can
  reach the Apple Silicon GPU.
- **Models**:
  - `google/gemma-4-e4b` — small 4B, simple/fast tasks (Level 1)
  - `google/gemma-4-26b-a4b` — 26B MoE, complex tasks, evaluation judges, agents
    (Levels 2–3)
  - `text-embedding-nomic-embed-text-v1.5` — embeddings for RAG / vector DB
- **Agent frameworks**: LangChain v1.0+, LangGraph, Claude Agent SDK, DeepAgents
- **Data**: PostgreSQL 16 (MLflow + Temporal backing store), Qdrant (vectors),
  Elasticsearch 7.17 (Temporal visibility)

## Infrastructure

- **Deploy**: none — this is a tutorial, nothing ships anywhere
- **Local stack**: `podman compose` from `infra/`. **Podman, not Docker** — do
  not translate commands from Docker documentation without checking them.
- **Orchestration**: Temporal.io (Level 3 only)
- **Observability**: Grafana + Prometheus (Level 3 only)

## Scripting & Automation

- Default: Python, consistent with the rest of the stack
- Shell scripts only for trivial one-liners (`infra/postgres/init-databases.sh`
  is the only one, and it stays that size)

## Conventions this machine imposes

- **One formatter per filetype.** Python formats with the ruff CLI chain. Biome
  owns the JS/TS family; prettier and eslint are not installed on this machine.
- **Tools run only where the repo carries their config file.** This repo carries
  `ruff.toml`, `.editorconfig` and `.hadolint.yaml`, so ruff, shfmt and hadolint
  are on. Markdown linting is on from a global config, not a repo one.
- **basedpyright is configured at the repo root, not per leaf.** See
  `pyrightconfig.json` and `05-implement.md`.

The contract behind all of this is `projects/tooling.md` in mac-setup, applied by
`/lint-format-lsp`.
