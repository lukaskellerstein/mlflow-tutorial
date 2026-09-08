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
into kinds that the learner needs to keep apart. Three modules use it today:

- `L1-M4_evaluation/` → `1_fundamentals/`, `2_offline/`, `3_online/`
- `L2-M1_agent_frameworks/` → `1_turn/`, `2_conversation/`
- `L2-M2_agent_evaluation/` → `1_instruments/`, `2_offline/`, `3_online/`

Everywhere else a lesson sits directly under its module. Do not add a group tier
to a module that has no such split — see the rule restated below for when a tier
is justified.

`L2-M1` splits on **scope** — how much of an interaction one run covers — and
carries the same three frameworks in both branches, three lessons each. The
turn lessons meet the framework; the conversation lessons teach only what
changes when state has to survive a turn boundary.

**Every `L2-M2` group splits again**, and `L2-M2` is the only module that does
this. The axis is the same **scope** — here, what a scorer is allowed to see:

| Group | `1_turn/` | `2_conversation/` | other |
|:--|--:|--:|:--|
| `1_instruments/` | 3 | 4 | `3_dataset_store/` 2 |
| `2_offline/` | 7 | 4 | — |
| `3_online/` | 2 | 2 | — |

`3_dataset_store` stays outside the split because a record is stored the same way
whatever its scope — that was verified by running the code, not assumed.

**The one-child rule, restated.** The older wording said "a heading with one
child is not a hierarchy". That was too blunt: it was written to stop
*decorative* tiers, and `3_online` breaks the letter of it while serving its
purpose exactly. The rule is:

> Do not add a group tier that carries no information. A tier is justified when
> it answers a question the reader would otherwise ask on every visit — even at
> one lesson per branch — provided every branch is expected to grow.

For `L2-M2` that question is "is this lesson about a turn or a conversation?",
and the reader asks it every time. Do not copy this shape into a module without
an axis that earns it the same way.

Each branch, and each group above it, carries a group `README.md` stating its
axis in one line. A lesson README never repeats that map.

## pyproject.toml Template

```toml
[project]
name = "mlflow-tutorial-L<level>-<module>-[<group>-]<lesson>"
version = "0.1.0"
description = "<Lesson title from syllabus>"
requires-python = ">=3.10"
dependencies = [
    "mlflow>=3.15",
    # Add lesson-specific deps here
]
```

`dependencies` must be a PEP 508 **array of strings**. Never write a
`[project.dependencies]` table with `pkg = ">=x.y"` entries (Poetry style) — uv
rejects it outright with `invalid type: map, expected a sequence`, and `uv sync`
fails before installing anything. Extras go inside the string too:
`"mlflow[genai]>=3.15"`, not `mlflow = {version = ">=3.15", extras = ["genai"]}`.

### Version floors

**MLflow is pinned uniformly, and that is the one exception to the rule below.**
Every leaf carries `mlflow>=3.15` or `mlflow[genai]>=3.15`, and every `uv.lock`
holds 3.15.2. The floor tracks the current release rather than the oldest
version each lesson would tolerate, because 62 leaves running 62 different
MLflow versions makes a bug report impossible to place — "it worked in my
lesson" has to mean the same thing everywhere. When MLflow releases again, bump
all 62 together:

```bash
# from each tutorial/**/<lesson>/ directory
uv lock --upgrade-package mlflow
```

**Every other version floor must reflect what the code actually calls** —
`langchain>=1.0` for the v1 `create_agent` API, `pydantic>=2` wherever a lesson
wraps a key in `SecretStr`. Do not raise one of those to the newest release just
because it exists; a floor is a statement about what the lesson needs. Add deps
with `uv add <pkg>` so the file stays valid.

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
