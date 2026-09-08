# WORKFLOW — MANDATORY FOR ANY PROMPT THAT RESULTS IN CHANGES

**If you are going to use the Edit or Write tool, or run a lesson, or touch the
podman stack, you MUST complete the workflow in `rules/` before reporting
completion.** Applies to every type of work — new lessons, fixes to existing
ones, infra changes, README and syllabus edits. No exceptions.

Steps, in order (each phase's detailed procedure is in the correspondingly-numbered
`rules/` file — already loaded into context, no need to open it):

1. **Understand** → [`rules/02-understand.md`](rules/02-understand.md)
2. **Plan** → [`rules/03-plan.md`](rules/03-plan.md) *(skip for trivial changes)*
3. **Implement** → [`rules/05-implement.md`](rules/05-implement.md)
4. **Test** → [`rules/06-testing.md`](rules/06-testing.md)
5. **Report** → [`rules/08-report.md`](rules/08-report.md)

Reference files: [`rules/01-project-config.md`](rules/01-project-config.md)
(architecture, leaves, services and ports),
[`rules/09-code-quality.md`](rules/09-code-quality.md),
[`rules/10-tech-stack.md`](rules/10-tech-stack.md),
[`rules/11-communication.md`](rules/11-communication.md),
[`rules/12-security.md`](rules/12-security.md),
[`rules/machine-tools.md`](rules/machine-tools.md) (the `nvim-tools` and
`lukas-ps` CLIs — pre-approved, read-only),
[`rules/lsp.md`](rules/lsp.md) (the `LSP` tool — only in repos that opted in,
and deferred, so it must be loaded before it can be called).

Path-scoped rules load automatically when you touch a matching file:
[`tutorial-structure.md`](rules/tutorial-structure.md) (lesson conventions),
[`coding-standards.md`](rules/coding-standards.md) (Python style for tutorial
code), [`mlflow-patterns.md`](rules/mlflow-patterns.md) (MLflow APIs),
[`agent-evaluation.md`](rules/agent-evaluation.md) (the core focus),
[`lesson-content.md`](rules/lesson-content.md) (README format),
[`references.md`](rules/references.md) (where the real source code is).

**NEVER report completion without first running the lesson end to end and
confirming the result in the MLflow UI.** Writing a lesson that imports cleanly
but was never executed is the failure mode this repo is most exposed to — MLflow
3.x moved a lot of API surface, and code that looks right frequently is not.
Verification is YOUR responsibility — the user should never need to ask you to
test.

**Trivial changes** (a typo, a comment, a README wording fix): skip step 2. State
what you'll do and proceed.

## MLFlow Tutorial at a glance

A comprehensive, three-level tutorial for MLFlow. The primary focus is **LLMs and
AI agents** (not traditional ML training). Special emphasis on **evaluation
(Evals) for AI agents** built with LangChain/LangGraph, DeepAgents, and Claude
Agent SDK.

The three levels:

- **Level 1 — Models**: Everything about models/LLMs end-to-end. Tracking,
  tracing, evaluation (offline and online), prompt registry, deployment, AI
  gateway, optimization.
- **Level 2 — AI Agents**: Agent frameworks (LangChain, LangGraph, Claude SDK,
  DeepAgents), agent evaluation (instruments / offline including benchmarks /
  online), agent optimization.
- **Level 3 — Advanced**: Production patterns, infrastructure (OpenTelemetry,
  Temporal, Grafana), extensibility, capstones.

Four things worth knowing before touching anything:

- **62 lesson leaves today**, one per lesson in `syllabus.md`, each
  independently runnable with its own `pyproject.toml`, `.venv` and `uv.lock`.
  There is **no uv workspace** — that is deliberate.
- **Three modules carry a group tier.** `L1-M4_evaluation` and
  `L2-M2_agent_evaluation` split into `1_fundamentals`/`1_instruments`,
  `2_offline` and `3_online`; `L2-M1_agent_frameworks` splits into `1_turn` and
  `2_conversation`. Their lessons are one level deeper than everywhere else. A
  lesson's experiment name always equals its path.
- **All three `L2-M2` groups go one level deeper still**, and `L2-M2` is the
  only module in the repo that does. Each splits by **scope**:
  `1_instruments/` → `1_turn/`, `2_conversation/`, `3_dataset_store/`;
  `2_offline/` and `3_online/` → `1_turn/`, `2_conversation/`. Those lessons sit
  four levels under the level directory. Every one of those branches carries a
  group `README.md`, as do `L2-M1`'s two groups and the module itself — with
  `level_2_agents/`, the only group-level READMEs in the tutorial.
- **The syllabus is the source of truth.** Module structure, lesson topics,
  deliverables and time estimates live in **`syllabus.md`** at the project root.
  Always consult it before creating or modifying any lesson.
- **Unsloth Studio runs natively, not in podman**, so it can reach the Apple Silicon
  GPU. Everything else is in the compose stack.
- **Adding a lesson means re-running `gen-pyrightconfig.py`** from mac-setup, or
  the new leaf resolves its imports against nothing.

### Technical stack

- **Python**: 3.10+
- **Package manager**: `uv` (every lesson is a standalone `uv` project)
- **LLM provider**: Unsloth Studio (local, no API costs, OpenAI-compatible API)
- **LLM server**: `http://127.0.0.1:8888` with OpenAI-compatible endpoint at `/v1/`.
  It needs a key on every route, and `Settings → API → Model auto-switch` ON —
  it holds one model at a time
- **LLM models**:
  - `unsloth/gemma-4-26B-A4B-it-qat-GGUF` — the 26B MoE behind `gemma-chat`,
    `gemma-judge` and `gemma-agent`; the default for the whole tutorial
  - `unsloth/gemma-4-31B-it-qat-GGUF` — the denser model behind `gemma-31b-local`
  - `second-state/Nomic-embed-text-v1.5-Embedding-GGUF` — embedding model for RAG/vector DB
- **MLFlow**: 3.x — every lesson pins `mlflow>=3.15` and locks 3.15.2; the
  server image is `ghcr.io/mlflow/mlflow:latest`
- **Agent frameworks**: LangChain v1.0+, LangGraph (latest), Claude Agent SDK,
  DeepAgents
- **Vector DB**: Qdrant (via Podman Compose)
- **Workflow orchestration**: Temporal.io (via Podman Compose, Level 3)
- **Observability**: Grafana + Prometheus (via Podman Compose, Level 3)
- **Container runtime**: Podman (not Docker)

### Starting infrastructure

The stack has **two tiers**, selected by a compose profile:

```bash
cd infra
podman compose up -d                     # Level 1 + Level 2: postgres, mlflow, mlflow-seed, qdrant
podman compose --profile level3 up -d    # Level 3: adds temporal (+elasticsearch, UI, admin-tools), prometheus, grafana
```

| Service | Tier | URL |
|---------|------|-----|
| MLflow UI | L1+ | <http://localhost:5555> |
| MLflow AI Gateway | L1+ | <http://localhost:5555/gateway/mlflow/v1> |
| Qdrant | L1+ | <http://localhost:6333/dashboard> |
| Temporal UI | L3 | <http://localhost:8080> |
| Grafana | L3 | <http://localhost:3000> (admin/admin) |
| Prometheus | L3 | <http://localhost:9090> |

A Level 3 lesson that needs Temporal or Prometheus fails with a connection
error on the default tier — check `podman compose ps` before debugging the
lesson. And **`podman compose down` without the profile leaves running L3
containers untouched**; stop everything with `--profile level3 down`.
`COMPOSE_PROFILES=level3` in `infra/.env` makes the profile implicit.

Unsloth Studio runs natively (not in Podman) for Apple Silicon GPU access.

### Running a lesson

```bash
cd tutorial/<level_N_domain>/<module>/<lesson>
uv sync
uv run python main.py
```

Level directories: `level_1_models/`, `level_2_agents/`, `level_3_advanced/`

### Key commands

- `podman compose up -d` — start the Level 1 + 2 tier (from `infra/`)
- `podman compose --profile level3 up -d` — start everything, for Level 3
- `podman compose down` / `--profile level3 down` — stop that tier (preserves data)
- `podman compose --profile level3 down -v` — stop and wipe all data
- `uv init` — scaffold a new lesson project
- `uv add <package>` — add a dependency
- `uv run python main.py` — run the lesson code
- `podman compose run --rm mlflow-seed --reset --prune` — rebuild every gateway
  alias after editing `infra/mlflow/gateway/seed_gateway.py`
- `podman compose logs mlflow-seed` — what the seeder built, and what it skipped

Full facts → [`rules/01-project-config.md`](rules/01-project-config.md); stack and
conventions → [`rules/10-tech-stack.md`](rules/10-tech-stack.md).

## Standing authorizations — do NOT ask before doing these

These actions are pre-approved. Run them yourself when the situation calls for it.

### Read-only inspection (always safe)

- Reading anything in the repo, including `syllabus.md`, `GOAL.md` and every
  lesson's `main.py` / `README.md` / `mlflow_funcs.md`.
- Reading the upstream source trees listed in
  [`rules/references.md`](rules/references.md) — MLflow, LangChain, LangGraph,
  DeepAgents, Claude Agent SDK. Read the real API rather than guessing at it.
- `podman compose ps`, `podman compose logs`, `podman ps` — from `infra/`.
- `curl` against any local service in the table above (health checks, MLflow
  REST API reads, `http://127.0.0.1:8888/v1/models` with the Unsloth key).
- `curl http://127.0.0.1:8888/v1/status` with the Unsloth key — what it has loaded.
- `uv tree`, `uv lock --check`, `uv pip list` in any lesson directory.
- `git status`, `git diff`, `git log` — any read-only git command.
- This machine's own `nvim-tools` and `lukas-ps` are pre-approved too, and are
  documented once in [`rules/machine-tools.md`](rules/machine-tools.md) — do not
  restate them here.
- Browsing the MLflow, Grafana, Temporal or Qdrant UIs with the Playwright MCP
  server. Close the browser when finished.

### Pre-approved mutations

- **`uv sync`, `uv add <pkg>`, `uv remove <pkg>`, `uv lock`, `uv run python
  main.py` — inside a `tutorial/**/<lesson>/` directory only.** Never at the repo
  root; there is no root project and creating one would break the no-workspace
  rule.
- **Creating and editing lesson files under `tutorial/`** — `main.py`,
  `README.md`, `mlflow_funcs.md`, `pyproject.toml`, `.gitignore`, and new lesson
  directories that match the structure in
  [`rules/tutorial-structure.md`](rules/tutorial-structure.md).
- **`podman compose up -d` and `podman compose down`, run from `infra/`, with
  or without `--profile level3`.** `down` without `-v` preserves the volumes,
  so restarting is free. Testing a Level 3 lesson means bringing the profile up
  first — do that rather than reporting Temporal or Prometheus as unreachable.
- Re-running mac-setup's `gen-pyrightconfig.py` against this repo after adding a
  leaf — then re-adding the three `report*` suppressions the generator drops
  (they are documented in the comment at the top of `pyrightconfig.json`).
- Deleting MLflow runs *you created during this session's testing*, via the UI or
  the REST API.

### Requires confirmation — always ask first

- **`podman compose down -v`, with or without a profile.** It wipes the
  volumes: every MLflow run, experiment, registered model and artifact, plus all
  Temporal history. There is no undo and no backup.
- **Editing `infra/compose.yml`, `infra/.env`, or anything else under `infra/`.**
  One file there affects all 60 lessons at once.
- **Deleting or renaming an existing lesson directory**, or changing the
  module/lesson numbering — `syllabus.md` and every cross-reference depend on it.
- **Editing `syllabus.md`.** It is the source of truth; changing it changes what
  every future lesson is supposed to be.
- **Deleting MLflow experiments, registered models, or runs you did not create.**
- **Bulk-reformatting Markdown with ruff.** It reformats Python inside code
  fences, and the lesson READMEs are the teaching material.
- `git push`, `git push --force`, branch deletes — **never commit unless the user
  explicitly asks**.
- Anything touching secrets, TLS material, tokens, or credential files. A secret
  never enters this repo in plaintext; if one must be versioned at all it is
  SOPS+age — [`rules/12-security.md`](rules/12-security.md).

When in doubt: ask. Nothing here is production, so the stakes are low — but a
`down -v` costs every run logged since the stack last came up, and rebuilding the
tutorial's experiment history by hand is not possible.
