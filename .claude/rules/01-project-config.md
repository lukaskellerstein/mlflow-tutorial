---
description: Project configuration — architecture, paths, dev environment
---

# Project Config

- **Project**: MLFlow Tutorial — a three-level MLflow course aimed at LLMs and AI
  agents, with evaluation (Evals) of agents as the core focus.
- **Architecture**: 62 independently runnable lesson leaves under `tutorial/`,
  plus one shared podman-compose infrastructure stack under `infra/`. There is
  **no** uv workspace tying the leaves together — that is deliberate.
- **Structure**:
  - `tutorial/level_1_models/` — tracking, tracing, registry, evaluation,
    prompt registry, deployment/gateway, optimization (M1–M7)
  - `tutorial/level_2_agents/` — agent frameworks, agent evaluation
    (instruments / offline / online), agent optimization (M1–M3)
  - `tutorial/level_3_advanced/` — production, advanced tracing, extensibility,
    capstones (M1–M4)
  - `infra/` — `compose.yml` and per-service config for the whole stack
  - `syllabus.md` — the source of truth for module/lesson structure
  - `GOAL.md` — what the tutorial is for
- **Build**: none. There is no build step and no packaging; each lesson is run
  directly from source.
- **Run locally**: `cd tutorial/<level>/<module>/<lesson> && uv sync && uv run python main.py`
- **Test**: there is no test suite. Verification is running the lesson's
  `main.py` end to end against a live MLflow server and Unsloth Studio, then checking
  the run appears in the MLflow UI. See `06-testing.md`.
- **Key dependencies**: `mlflow>=3.15`, `openai>=1.0`, LangChain v1.0+, LangGraph,
  Claude Agent SDK, DeepAgents, pandas, Qdrant, Temporal
- **Package manager**: `uv` — one project, one `.venv` and one `uv.lock` per leaf

## Leaves

This repo is **not one project**. Each lesson below is independently runnable and
keeps its own environment — no workspaces, by design.

| Level | Modules | Lessons |
|:--|:--|--:|
| `level_1_models` | M1 tracking, M2 tracing, M3 models/registry, M4 evaluation (`1_fundamentals`, `2_offline`, `3_online`), M5 prompt registry, M6 deployment, M7 optimization | 18 |
| `level_2_agents` | M1 agent frameworks (`1_turn`, `2_conversation`), M2 agent evaluation (`1_instruments`, `2_offline`, `3_online`), M3 agent optimization | 33 |
| `level_3_advanced` | M1 production, M2 advanced tracing, M3 extensibility, M4 capstones | 11 |

Three modules — `L1-M4_evaluation`, `L2-M1_agent_frameworks` and
`L2-M2_agent_evaluation` — carry an extra **group** tier between module and
lesson (`2_offline/1_genai_custom_metrics/`, `1_turn/2_deepagents/`). Everywhere
else a lesson sits directly under its module.

`L2-M1` splits by the same **scope** axis `L2-M2` uses, and carries the same
three frameworks in both branches: `1_turn/` runs one task from an empty message
list, `2_conversation/` runs four dependent turns in one session.

**Every `L2-M2` group carries a second tier on top of that**, and `L2-M2` is the
only module in the repo that does. All three split by scope:

| Group | `1_turn/` | `2_conversation/` | other |
|:--|--:|--:|:--|
| `1_instruments/` | 3 | 4 | `3_dataset_store/` 2 |
| `2_offline/` | 7 | 4 | — |
| `3_online/` | 2 | 2 | — |

`3_online` has one lesson per branch on purpose. The tier is kept because it
answers, at a glance, the question a reader asks on every visit — is this lesson
about a turn or a conversation — and both branches are expected to grow.

Each of those branches, and each group above them, carries a group `README.md`.
So do `L2-M1`'s two branches and `L2-M1` itself. Apart from
`level_2_agents/README.md` they are the only group-level READMEs in the
tutorial — every other README sits beside a `main.py`.

Every leaf carries its own `pyproject.toml`, `.venv` and `uv.lock`. The repo-root
`pyrightconfig.json` holds one `executionEnvironments` entry per leaf — **re-run
`gen-pyrightconfig.py` from mac-setup when a leaf is added**, or the new lesson
resolves its imports against nothing.

## Services and ports

Started from `infra/` in two tiers: `podman compose up -d` brings up what Level 1
and Level 2 need; `podman compose --profile level3 up -d` adds the Level 3
services on top. Plain `down` stops only the tier it can see — use
`--profile level3 down` to stop everything.

| Service | Tier | URL | Notes |
|:--|:--|:--|:--|
| MLflow UI | L1+ | <http://localhost:5555> | tracking server + artifact store |
| MLflow AI Gateway | L1+ | <http://localhost:5555/gateway/mlflow/v1> | every lesson's LLM entry point; the MLflow server itself |
| Qdrant | L1+ | <http://localhost:6333/dashboard> | vector DB for RAG lessons |
| Temporal UI | L3 | <http://localhost:8080> | `--profile level3` |
| Grafana | L3 | <http://localhost:3000> | admin/admin, `--profile level3` |
| Prometheus | L3 | <http://localhost:9090> | `--profile level3` |
| Unsloth Studio | host | <http://127.0.0.1:8888/v1/> | **not** in podman — runs natively for Apple Silicon GPU access. Needs a key, and `Model auto-switch` ON |

`infra/.env` is local-only and untracked; the committed `infra/.env.example`
carries the variable names and the localhost-only development defaults
(`admin`/`admin` and friends). First-time setup: `cp .env.example .env` from
`infra/`. Do not put a real secret in either file — see `12-security.md`.
