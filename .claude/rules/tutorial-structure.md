---
globs: ["tutorial/**"]
---

# Tutorial Structure Rules

## Three-Level Architecture

- **`tutorial/level_1_models/`** — Models: everything about models/LLMs end-to-end (tracking, tracing, evaluation, deployment, prompt engineering, AI gateway, fine-tuning)
- **`tutorial/level_2_agents/`** — AI Agents: agent frameworks, custom integrations, agent evaluation, benchmarking
- **`tutorial/level_3_advanced/`** — Advanced: production patterns, infrastructure, extensibility, capstones

Always consult `syllabus.md` (project root) for the full module/lesson breakdown before creating or modifying any lesson.

## Lesson Directory Convention

Every lesson lives in `tutorial/<level_N_domain>/<module>/[<group>/]<lesson>/` and contains exactly:

The `<group>` tier is **optional** and exists only where a module's lessons split
into kinds that the learner needs to keep apart. Two modules use it today:

- `L1-M4_evaluation/` → `1_fundamentals/`, `2_offline/`, `3_online/`
- `L2-M2_agent_evaluation/` → `1_instruments/`, `2_offline/`, `3_online/`

Everywhere else a lesson sits directly under its module. Do not add a group tier
to a module that has no such split — a heading with one child is not a hierarchy.

1. **`pyproject.toml`** — standalone `uv` project. Use `[project]` with `name`, `version`, `description`, `requires-python`, and `dependencies`. Pin major versions only (e.g., `mlflow>=2.0`).
2. **`main.py`** — the working lesson code. This is the primary deliverable.
3. **`README.md`** — lesson guide (see `lesson-content.md` rule for format).
4. **`.gitignore`** — always ignore: `.venv/`, `__pycache__/`, `mlruns/`, `mlartifacts/`, `*.pyc`, `.python-version`.

## pyproject.toml Template

```toml
[project]
name = "mlflow-tutorial-L<level>-<module>-[<group>-]<lesson>"
version = "0.1.0"
description = "<Lesson title from syllabus>"
requires-python = ">=3.10"
dependencies = [
    "mlflow>=3.0",
    # Add lesson-specific deps here
]
```

`dependencies` must be a PEP 508 **array of strings**. Never write a
`[project.dependencies]` table with `pkg = ">=x.y"` entries (Poetry style) — uv
rejects it outright with `invalid type: map, expected a sequence`, and `uv sync`
fails before installing anything. Extras go inside the string too:
`"mlflow[genai]>=3.0"`, not `mlflow = {version = ">=3.0", extras = ["genai"]}`.

Version floors must reflect what the code actually calls — `langchain>=1.0` for
the v1 `create_agent` API, `mlflow>=3.0` for `mlflow.genai` / assessment APIs.
Add deps with `uv add <pkg>` so the file stays valid.

## .gitignore Template

```text
.venv/
__pycache__/
*.pyc
mlruns/
mlartifacts/
.python-version
```

## Principles

- Each lesson must be fully self-contained — `cd` into it, `uv sync && uv run python main.py`, see results.
- All lessons connect to the shared MLFlow server at `http://127.0.0.1:5555`. Set `MLFLOW_TRACKING_URI` in code, not env vars.
- Use `mlflow.set_experiment("L<level>/<module>/[<group>/]<lesson>")` so experiments are organized in the MLFlow UI. **The experiment name is always the lesson's path under `tutorial/level_N_*/`** — if the directory moves, the experiment name moves with it.
- Print meaningful output to the console so the user sees what's happening without needing the MLFlow UI.
- Keep `main.py` under ~300 lines. If a lesson needs helper code, put it in a separate module within the same directory.
- Level 1 (Models) covers each topic end-to-end (basic through advanced). Merged lessons may be longer (~250-350 lines).
- Level 2 (Agents) assumes L1 knowledge — no re-teaching tracking/tracing basics.
- Level 3 (Advanced) should produce production-quality code and integrate multiple concepts.
