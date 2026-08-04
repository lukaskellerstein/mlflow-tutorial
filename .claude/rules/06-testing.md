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
> - [ ] A new run appears under the expected experiment, confirmed via the API
> - [ ] The params/metrics/artifacts the lesson claims to log are actually there
> - [ ] The README's stated output matches what the script printed
> - [ ] Browser closed — only if a UI check was actually needed

## 4b. Test

**Preconditions** — both are external to the repo and both fail confusingly when
absent:

```bash
curl -sf http://localhost:5555/health            && echo "mlflow up"
curl -sf http://localhost:4000/health/readiness  && echo "litellm up"
curl -sf http://localhost:1234/v1/models         && echo "lmstudio up"
```

If MLflow is down: `cd infra && podman compose up -d`.

**Check which of the two LLM paths the lesson actually uses before panicking
about LMStudio.** A lesson on the `gemma-large` alias reaches OpenRouter through
the LiteLLM gateway and does not care that LMStudio is down; only `gemma-small`
and `nomic-embed` are served locally. If a lesson genuinely needs LMStudio and it
is down, say so and ask — it runs natively for GPU access and cannot be started
from compose. `lms server start` and `lms ps` are the CLI equivalents.

> [!note]
> Editing `infra/litellm/config.yaml` needs `podman compose restart litellm` to
> take effect. The file is bind-mounted and read once at startup, so
> `podman compose up -d` is a no-op and leaves the old config running.

**Lesson changes** — run the lesson from its own directory:

```bash
cd tutorial/<level>/<module>/<lesson>
uv sync
uv run python main.py
```

Read the output. An exit code of 0 is not the test — the test is whether the
lesson *taught* what it claims: the run logged, the metric recorded, the trace
captured, the model registered.

**Verify against the MLflow API — not the UI.** The MLflow Python client answers
every question the UI answers about *data*, in a fraction of the tokens and
without a browser. Run it from the lesson's own venv so the client version
matches what the lesson uses:

```bash
cd tutorial/<level>/<module>/<lesson>
uv run python -c "
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5555')
exp = mlflow.set_experiment('<the lesson experiment>')

run = MlflowClient().search_runs([exp.experiment_id], max_results=1,
                                 order_by=['attributes.start_time DESC'])[0]
print('run    :', run.info.run_name, run.info.status)
print('params :', run.data.params)
print('metrics:', run.data.metrics)
print('traces :', len(mlflow.search_traces(locations=[exp.experiment_id],
                                           max_results=500, return_type='list')))
print('scorers:', [(s.name, s.kind.value) for s in mlflow.genai.list_scorers()])
"
```

Assert on the values. "A run exists" is not the test; the test is that the
params, metrics, traces or registered scorers the lesson *claims* to produce are
present and have the right values.

Prefer the Python client to raw REST. The REST API is real and reads are
pre-approved — but it lives at **`/api/3.0/mlflow/...`** (a few legacy endpoints
are still `2.0`, and guessing the wrong one returns a bare 405), and
**MLflow publishes no OpenAPI/swagger spec**: `/openapi.json`, `/swagger.json`
and `/docs` all 404, and no spec file ships in the package. The routes are
generated from protobuf definitions, so `handlers.py` holds almost no literal
paths. Discovering a route means reading `store/tracking/rest_store.py` or
`tracing/client.py` in the lesson's venv. The client already knows them.

**Use Playwright only when the claim is about rendering**, not about data —
a badge, a chart, the trace waterfall, or a lesson whose whole point is what the
learner sees on screen. Two things the API will not tell you: that MLflow badges
an aligned judge as "LLM-as-a-judge (Optimized)", and that a registered scorer
shows `Evaluating traces: OFF`.

1. Open a browser via `mcp__playwright-mlflow-tutor__browser_navigate`.
2. Snapshot the specific element — don't assert the page loaded.
3. **Close the browser when done.**

> The browser opens on its own desktop/space and is closed automatically at
> session end by `.claude/hooks/`. That is a safety net, not a substitute for
> closing it yourself when the test is finished.

> [!tip]
> The API sometimes sees more than the UI. `mlflow.genai.list_scorers()` returns
> objects whose `.name` is the judge's *internal* name, while the UI and
> `/api/3.0/mlflow/scorers/list` show the name it was *registered* under — so an
> aligned judge registered as `foo_aligned` comes back with `.name == "foo"`.
> Reporting registered names from `.name` is wrong, and the UI hides it.

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
