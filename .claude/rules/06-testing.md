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
curl -sf http://localhost:5555/health && echo "mlflow + gateway up"
curl -sf http://127.0.0.1:8888/v1/status -H "Authorization: Bearer $UNSLOTH_API_KEY" && echo "unsloth up"
```

There is no separate gateway port: the MLflow server IS the gateway, so one
probe answers for both. If MLflow is down: `cd infra && podman compose up -d`.
`/v1/models` does not exist on this gateway — do not probe it.

**Level 3 lessons that use Temporal (`L3-M2.2`) or Prometheus/Grafana
(`L3-M1.2`) need the `level3` profile** — those services are not in the default
tier. Check `nc -z localhost 7233` / `curl -sf localhost:9090/-/ready` and, if
closed, `cd infra && podman compose --profile level3 up -d`. A Level 3 lesson
that fails on `localhost:7233` against the default tier is not a lesson bug.

**Every lesson goes through the MLflow AI Gateway; nothing calls a provider
directly.**

**Every alias resolves to Unsloth, and none has a fallback.** So Unsloth being
down fails every lesson that calls a model, loudly and at the first call. That
is the design: there is no hosted provider left in the gateway, so a lesson can
no longer pass while quietly running somewhere else. A failure here is one real
cause, not a mystery about which model answered.

Unsloth runs natively for GPU access and cannot be started from compose. If it
is down, say so and ask.

> [!warning]
> **Unsloth holds ONE model at a time.** `Settings → API → Model auto-switch`
> must be ON, or every alias but the currently loaded model fails with
> `400 ... 'Switch model by request' is off`. This is a GUI setting with no CLI
> equivalent, so it cannot be fixed from here — ask. With it on, a swap costs
> 4–14 s and needs no intervention.

> [!warning]
> **Editing `infra/mlflow/gateway/seed_gateway.py` is not enough on its own.** The
> seeder is idempotent, so `podman compose up -d` reports an existing alias as
> `reused` and it quietly keeps the model it already had. ADDING an alias works
> with `up -d`; CHANGING one needs
> `podman compose run --rm mlflow-seed --reset --prune`. The script is the
> container's entrypoint, so pass only its flags.

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

The browser is placed on its own desktop space by the shared Playwright hooks
(`.claude/hooks/`) — never switch to it, focus it, or move it. Session-end
cleanup is a safety net, not a substitute for closing it yourself.

**A headed browser your own script launches goes through the launch script
too** — `.claude/hooks/playwright-launch.sh npx tsx <script>` — so its window
is born on the scratch space like every other agent browser. Never launch a
headed browser bare: macOS creates new windows on whatever space the user is
looking at. A headless browser needs no placement — run that script directly.

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
