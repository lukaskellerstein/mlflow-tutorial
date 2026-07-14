---
globs: ["tutorial/**"]
---

# Tutorial Structure Rules

## Three-Level Architecture

- **`tutorial/level_1/`** — Essentials (breadth): every major MLflow feature, short lessons (~30 min)
- **`tutorial/level_2/`** — Practitioner (depth): real-world projects, longer lessons (~1-2 hours)
- **`tutorial/level_3/`** — Expert (mastery): production patterns, agent evaluation, custom integrations

Always consult `syllabus.md` (project root) for the full module/lesson breakdown before creating or modifying any lesson.

## Lesson Directory Convention

Every lesson lives in `tutorial/<level>/<module>/<lesson>/` and contains exactly:

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
- Keep `main.py` under ~200 lines. If a lesson needs helper code, put it in a separate module within the same directory.
- Level 1 lessons should be concise and focused on a single concept.
- Level 2 lessons can be longer and build multi-step projects.
- Level 3 lessons should produce production-quality code and integrate multiple concepts.
