---
description: "Step 1: Understand — read code, ask questions, identify gaps before any implementation"
---

# Step 1: Understand

- Read relevant code and identify impacted areas
- Baseline the repo's existing problems with `nvim-tools --json --all`, so
  findings you introduce stay distinguishable from ones that were already there.
  For performance or RAM questions, `lukas-ps --json [name]` measures the real
  process tree. Both: [`machine-tools.md`](machine-tools.md).
- For symbol questions — where is this defined, what breaks if I change this
  signature — prefer the `LSP` tool when this repo has it: [`lsp.md`](lsp.md).
- Ask clarifying questions if requirements are ambiguous
- Identify gaps in the current design and opportunities for improvement
- Understand the requirement completely before proceeding
- **For bug reports**: reproduce the issue first — run the lesson's `main.py`
  from its own directory (`cd tutorial/<level>/<module>/<lesson> && uv run
  python main.py`) and read the actual traceback, plus the run in the MLflow UI,
  before attempting a fix.

## Before writing any lesson

Two reads are mandatory and neither is optional:

1. **`syllabus.md`** is the source of truth for module structure, lesson topics,
   deliverables and ordering. Consult it before creating or modifying a lesson.
2. **`references.md`** points at the actual MLflow, LangChain, LangGraph,
   DeepAgents and Claude Agent SDK source trees on this machine. Read the real
   API before writing against it — MLflow 3.x moved a great deal, and a
   plausible-looking call that does not exist is the most common failure here.

## The shape of this repo

A lesson is a **leaf**: its own `pyproject.toml`, its own `.venv`, its own
`uv.lock`, runnable with no reference to any other lesson. When you are asked to
change "the tutorial", establish which leaves are actually in scope before
touching anything — there are 43, and a change that belongs in one rarely belongs
in all of them.
