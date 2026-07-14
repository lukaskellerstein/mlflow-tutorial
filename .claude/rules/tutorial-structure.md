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

Every lesson lives in `tutorial/<level_N_domain>/<module>/<lesson>/` and contains exactly:

1. **`pyproject.toml`** — standalone `uv` project. Use `[project]` with `name`, `version`, `description`, `requires-python`, and `dependencies`. Pin major versions only (e.g., `mlflow>=2.0`).
2. **`main.py`** — the working lesson code. This is the primary deliverable.
3. **`README.md`** — lesson guide (see `lesson-content.md` rule for format).
4. **`.gitignore`** — always ignore: `.venv/`, `__pycache__/`, `mlruns/`, `mlartifacts/`, `*.pyc`, `.python-version`.

## pyproject.toml Template

```toml
[project]
name = "mlflow-tutorial-L<level>-<module>-<lesson>"
version = "0.1.0"
description = "<Lesson title from syllabus>"
requires-python = ">=3.10"

[project.dependencies]
mlflow = ">=2.0"
# Add lesson-specific deps here
```

## .gitignore Template

```
.venv/
__pycache__/
*.pyc
mlruns/
mlartifacts/
.python-version
```

## Principles

- Each lesson must be fully self-contained — `cd` into it, `uv sync && uv run python main.py`, see results.
- All lessons connect to the shared MLFlow server at `http://127.0.0.1:5000`. Set `MLFLOW_TRACKING_URI` in code, not env vars.
- Use `mlflow.set_experiment("L<level>/<module>/<lesson>")` so experiments are organized in the MLFlow UI.
- Print meaningful output to the console so the user sees what's happening without needing the MLFlow UI.
- Keep `main.py` under ~300 lines. If a lesson needs helper code, put it in a separate module within the same directory.
- Level 1 (Models) covers each topic end-to-end (basic through advanced). Merged lessons may be longer (~250-350 lines).
- Level 2 (Agents) assumes L1 knowledge — no re-teaching tracking/tracing basics.
- Level 3 (Advanced) should produce production-quality code and integrate multiple concepts.
