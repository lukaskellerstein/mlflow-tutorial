---
description: "Step 3: Implement — coding rules and this project's layout"
---

# Step 3: Implement

Write clean code from the start. Follow these rules during implementation:

- Do NOT commit via `git` unless explicitly instructed by the user
- When creating diagrams or graphs, use `mermaid`
- Write clean code from the start — don't plan to "clean it up later"
- Refactor continuously — improve code structure immediately when you see issues
- Remove dead code — delete unused functions, variables, imports, and commented code
- After writing code: review comments, clean up imports, check for side effects
- Before changing a shared signature, check callers with `findReferences` rather
  than grep: [`lsp.md`](lsp.md)

Python style for lesson code is `coding-standards.md`; MLflow API usage is
`mlflow-patterns.md`; lesson README format is `lesson-content.md`. Those three are
glob-scoped and load automatically when you touch a matching file.

## `tutorial/` — the lessons

One directory per lesson, `<N>_<snake_case_name>/`, holding `main.py`,
`README.md`, `mlflow_funcs.md`, `pyproject.toml`, `.gitignore` and `uv.lock`.

- **A lesson is self-contained.** No imports from another lesson, no shared
  helper package, no relative import that climbs out of the leaf. If two lessons
  need the same code, both get a copy — duplication is the correct answer here,
  because each leaf must run standalone.
- **New lesson = new leaf.** `uv init`, then `uv add` its dependencies, then
  re-run mac-setup's `gen-pyrightconfig.py` so the leaf gets an
  `executionEnvironments` entry. Skipping that step means the editor resolves the
  new lesson's imports against nothing.
- **Do not add a `[tool.basedpyright]` table to a new leaf.** Every existing leaf
  has one and all of them are inert — the checker roots at the repo, not the
  leaf, so a nested table is never read. Type-checker settings belong in the root
  `pyrightconfig.json`.
- Structure and naming conventions are in `tutorial-structure.md`.

## `infra/` — the shared stack

`compose.yml` plus one directory per service. Changing it affects every lesson at
once, so it is not a place for casual edits — see the standing authorizations in
`CLAUDE.md`.

- `infra/.env` is untracked — first setup copies it from the committed
  `infra/.env.example`. It holds local-only development defaults. Never put a
  real credential in either file.
- `podman`, not docker. The commands differ in places; do not translate from
  docker docs without checking.

## Repository root

```text
mlflow-tutorial/
├── syllabus.md              source of truth for lesson structure
├── GOAL.md                  what the tutorial is for
├── README.md                setup and orientation
├── ruff.toml                lint + format opt-in (one file serves all 43 leaves)
├── pyrightconfig.json       one executionEnvironments entry per leaf
├── .editorconfig            shfmt opt-in
├── .hadolint.yaml           Dockerfile lint opt-in
├── .mcp.json                MCP servers
├── infra/                   podman compose stack
├── tutorial/
│   ├── level_1_models/      M1..M7
│   ├── level_2_agents/      M1..M4
│   └── level_3_advanced/    M1..M4
└── .claude/
    ├── CLAUDE.md
    ├── hooks/               Playwright desktop hooks
    └── rules/
```

Do not add a top-level directory without a reason that survives being asked
"which of the four things in this repo is it?"
