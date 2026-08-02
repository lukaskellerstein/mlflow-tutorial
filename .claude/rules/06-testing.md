---
description: "Step 4: Testing — define DoD, test, fix and repeat until passing"
---

# Step 4: Testing

**Every code change must be tested before reporting completion. No exceptions.**

There is no automated test suite in this repo. That does not lower the bar — it
means the test is *running the lesson* and confirming what it produced. A lesson
that has never been executed is not finished.

## 4a. Define your Definition of Done

Before testing, **write out your DoD checklist in the conversation** so the user
can see what you intend to verify. Example:

> **Definition of Done for this task:**
>
> - [ ] `uv run python main.py` completes without a traceback
> - [ ] A new run appears under the expected experiment in the MLflow UI
> - [ ] The params/metrics/artifacts the lesson claims to log are actually there
> - [ ] The README's stated output matches what the script printed
> - [ ] Browser closed after checking the UI

## 4b. Test

**Preconditions** — both are external to the repo and both fail confusingly when
absent:

```bash
curl -sf http://localhost:5555/health   && echo "mlflow up"
curl -sf http://localhost:1234/v1/models && echo "lmstudio up"
```

If MLflow is down: `cd infra && podman compose up -d`. If LMStudio is down, say
so and ask — it runs natively for GPU access and cannot be started from compose.
`lms server start` and `lms ps` are the CLI equivalents.

**Lesson changes** — run the lesson from its own directory:

```bash
cd tutorial/<level>/<module>/<lesson>
uv sync
uv run python main.py
```

Read the output. An exit code of 0 is not the test — the test is whether the
lesson *taught* what it claims: the run logged, the metric recorded, the trace
captured, the model registered.

**Verify in the MLflow UI** with Playwright against <http://localhost:5555>:

1. Open a browser via `mcp__playwright-mlflow-tutor__browser_navigate`.
2. Go to the experiment, open the run, and confirm the params, metrics, artifacts
   or traces the lesson claims to produce are visibly there — take a snapshot,
   don't just assert the page loaded.
3. **Close the browser when done.**

> The browser opens on its own desktop/space and is closed automatically at
> session end by `.claude/hooks/`. That is a safety net, not a substitute for
> closing it yourself when the test is finished.

**Infra changes** — `podman compose up -d` from `infra/`, then check every
service the change could touch is actually reachable on the ports in
`01-project-config.md`. `podman compose ps` showing "running" is not enough; a
container can be up and the service inside it broken.

**Every change** — repo-wide lint / format / type check:

```bash
nvim-tools --json --all
```

Your change must not add findings, measured against the baseline you took in the
Understand step. How to read the output (including `gated-off`), and why this
never replaces the project's own suite: [`machine-tools.md`](machine-tools.md).

This one is not optional for doc-only work either. Markdown is the deliverable
here as much as Python is — the lesson READMEs and `syllabus.md` are the teaching
material, and markdownlint is the only thing checking them.

**Doc-only changes** (README, syllabus, notes): run the check above, then
explicitly state why no *runtime* test is needed.

## 4c. Fix and repeat

If a test fails: fix the issue, then retest. Repeat until all DoD items pass. If
you hit a problem you repeatedly cannot resolve, ask the user for help rather
than reporting partial success.

A lesson that fails because a *dependency* moved (MLflow 3.x renamed something,
LangChain v1 changed an import) is a real finding — fix the lesson against the
actual source in `references.md`, do not pin the dependency backwards to make the
error go away.

## 4d. Never report completion without testing

If you write code and stop without verifying it works, you have failed. Testing
is YOUR responsibility — the user should never need to ask you to test.
