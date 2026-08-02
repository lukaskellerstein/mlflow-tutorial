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
`lukas-ps` CLIs — pre-approved, read-only).

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
  tracing, evaluation, deployment, prompt engineering, AI gateway, fine-tuning.
- **Level 2 — AI Agents**: Agent frameworks (LangChain, LangGraph, Claude SDK,
  DeepAgents), agent evaluation/benchmarking, custom integrations.
- **Level 3 — Advanced**: Production patterns, infrastructure (OpenTelemetry,
  Temporal, Grafana), extensibility, capstones.

Four things worth knowing before touching anything:

- **43 lesson leaves**, each independently runnable with its own `pyproject.toml`,
  `.venv` and `uv.lock`. There is **no uv workspace** — that is deliberate.
- **The syllabus is the source of truth.** Module structure, lesson topics,
  deliverables and time estimates live in **`syllabus.md`** at the project root.
  Always consult it before creating or modifying any lesson.
- **LMStudio runs natively, not in podman**, so it can reach the Apple Silicon
  GPU. Everything else is in the compose stack.
- **Adding a lesson means re-running `gen-pyrightconfig.py`** from mac-setup, or
  the new leaf resolves its imports against nothing.

### Technical stack

- **Python**: 3.10+
- **Package manager**: `uv` (every lesson is a standalone `uv` project)
- **LLM provider**: LMStudio (local, no API costs, OpenAI-compatible API)
- **LLM server**: `http://localhost:1234` with OpenAI-compatible endpoint at `/v1/`
- **LLM models**:
  - `google/gemma-4-e4b` — small 4B model for simple/fast tasks (Level 1 lessons)
  - `google/gemma-4-26b-a4b` — large 26B MoE model for complex tasks (Level 2/3,
    evaluation judges, agents)
  - `text-embedding-nomic-embed-text-v1.5` — embedding model for RAG/vector DB
- **MLFlow**: 3.x — lessons pin `mlflow>=3.0`; the server image is
  `ghcr.io/mlflow/mlflow:latest`
- **Agent frameworks**: LangChain v1.0+, LangGraph (latest), Claude Agent SDK,
  DeepAgents
- **Vector DB**: Qdrant (via Podman Compose)
- **Workflow orchestration**: Temporal.io (via Podman Compose, Level 3)
- **Observability**: Grafana + Prometheus (via Podman Compose, Level 3)
- **Container runtime**: Podman (not Docker)

### Starting infrastructure

```bash
cd infra
podman compose up -d
```

| Service | URL |
|---------|-----|
| MLflow UI | <http://localhost:5555> |
| Temporal UI | <http://localhost:8080> |
| Qdrant | <http://localhost:6333/dashboard> |
| Grafana | <http://localhost:3000> (admin/admin) |
| Prometheus | <http://localhost:9090> |

LMStudio runs natively (not in Podman) for Apple Silicon GPU access.

### Running a lesson

```bash
cd tutorial/<level_N_domain>/<module>/<lesson>
uv sync
uv run python main.py
```

Level directories: `level_1_models/`, `level_2_agents/`, `level_3_advanced/`

### Key commands

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
  REST API reads, `http://localhost:1234/v1/models`).
- `lms ls`, `lms ps` — what LMStudio has loaded.
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
- **`podman compose up -d` and `podman compose down`, run from `infra/`.**
  `down` without `-v` preserves the volumes, so restarting is free.
- Re-running mac-setup's `gen-pyrightconfig.py` against this repo after adding a
  leaf — then re-adding the three `report*` suppressions the generator drops
  (they are documented in the comment at the top of `pyrightconfig.json`).
- Deleting MLflow runs *you created during this session's testing*, via the UI or
  the REST API.

### Requires confirmation — always ask first

- **`podman compose down -v`.** It wipes the volumes: every MLflow run,
  experiment, registered model and artifact, plus all Temporal history. There is
  no undo and no backup.
- **Editing `infra/compose.yml`, `infra/.env`, or anything else under `infra/`.**
  One file there affects all 43 lessons at once.
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
