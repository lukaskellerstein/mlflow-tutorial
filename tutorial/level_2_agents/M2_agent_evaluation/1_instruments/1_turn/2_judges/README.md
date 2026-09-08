# L2-M2.1.1.2 — Judges for Agents: Inline, Registered, Aligned

**Level:** AI Agents
**Duration:** 90 min

## Overview

There are three ways to express an LLM-as-judge in MLflow, and they are not
interchangeable. This lesson builds the *same* rubric all three ways on one
dataset, shows the one thing an inline judge structurally cannot do, and then
lets a judge learn a standard nobody ever wrote into its prompt.

## Prerequisites

- Completed: L2-M2.1.1.1 to L2-M2.1.3.1 (the test material: hand-written suites,
  simulation, generation, datasets), L1-M4.1.1 (Evaluation Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- MLflow AI Gateway seeded (`cd infra && podman compose up -d`) — it is the
  MLflow server itself, at <http://127.0.0.1:5555/gateway/mlflow/v1>
- Unsloth Studio running natively with `gemma-4-26B-A4B-it-qat` and
  `Nomic-embed-text-v1.5` available — this lesson's chat, judge and
  embedding aliases all resolve locally, so it needs no network and costs nothing

## Concepts

### The three forms

| Form | Built with | Lives where | Can go online? |
|:--|:--|:--|:--|
| **Inline** | `@scorer` + your own prompt | your script | no |
| **Registered** | `make_judge(...).register()` | the tracking server, versioned | yes |
| **Aligned** | `judge.align(traces)` | same, as a new version | yes |

An inline judge gives you total control: your prompt, your parsing, your
`Feedback`. The cost is that it is invisible to everything except the script it
lives in. Nothing else can call it, no one can see which version produced last
week's scores, and it cannot run anywhere but in your process.

### The dividing line is enforced, not stylistic

Try to register a `@scorer` against an open-source tracking server and MLflow
refuses:

> Custom scorer registration (using @scorer decorator) is not supported outside
> of Databricks tracking environments due to security concerns.

A `@scorer` is stored as *source code* and reconstructed by `exec()`-ing it, so
the open-source server declines to keep one. `make_judge` produces an
`INSTRUCTIONS`-kind judge — data, not code — and registers fine. This is the
practical reason to reach for `make_judge` even when a plain function would do:
**only registered judges become server-side objects.**

### Alignment: teaching a judge the standard you never wrote down

Rubrics carry unwritten assumptions. Here, the support team's real bar is not
"is this helpful?" — it is "does it cite the policy reference the customer
needs?" Nobody put that in the judge's instructions, so a naive judge calls
everything helpful.

`judge.align(traces)` reads human labels off the traces and re-derives the
judge's instructions to match them. You do not edit a prompt; you supply
examples of what you actually meant.

> [!important]
> Alignment reads **human** assessments whose name matches the judge's name. A
> trace with no `Feedback` named `answer_helpfulness` contributes nothing.

## Step-by-Step

### Step 1: Produce something worth judging

Twelve support questions run through a traced agent. Two system prompts
alternate — one required to cite a policy reference, one forbidden from it — so
the answer set is reliably mixed.

```python
@mlflow.trace(name="support_agent")
def support_agent(question: str, cite_policy: bool) -> str:
    system = CITING_PROMPT if cite_policy else TERSE_PROMPT
    ...
```

This detail matters more than it looks. An earlier version of this lesson used
one prompt told to "cite a reference about half the time"; the model cited one
every single time, human and judge agreed 100%, and the lesson demonstrated
nothing. **Construct the variance you need — do not ask a model to produce it.**

### Step 2: The inline judge

```python
@scorer
def inline_helpfulness(inputs: dict, outputs: dict) -> Feedback:
    verdict = build_llm().invoke(INLINE_PROMPT.format(...))
    return Feedback(value=..., rationale=..., source=AssessmentSource(...))
```

### Step 3: Watch registration get refused

```python
inline_helpfulness.register(name="inline_helpfulness")  # raises MlflowException
```

### Step 4: The registered judge

```python
naive_judge = mlflow.genai.make_judge(
    name="answer_helpfulness",
    instructions="... {{ inputs }} ... {{ outputs }} ... true or false.",
    model="openai:/gemma-judge",
    feedback_value_type=bool,
)
registered = naive_judge.register(name="answer_helpfulness")
```

### Step 5: Align it

```python
mlflow.log_feedback(
    trace_id=trace_id,
    name="answer_helpfulness",  # must match the judge
    value=label,
    source=AssessmentSource(source_type="HUMAN", source_id="support-team"),
)
aligned_judge = naive_judge.align(traces)
```

## Gateway wiring — four traps

Both cost real debugging time, and neither error message points at its cause.

**1. `setdefault` is the wrong verb for `OPENAI_BASE_URL`.** If you already
export a real OpenAI key and base URL, `os.environ.setdefault` keeps them and
every judge call goes to `api.openai.com` instead of to this gateway. The lesson
assigns unconditionally.

**2. `make_judge(base_url=...)` wants the FULL endpoint URL.** Not
`http://127.0.0.1:5555/gateway/mlflow/v1`, but
`http://127.0.0.1:5555/gateway/mlflow/v1/chat/completions`. Setting
`OPENAI_BASE_URL` instead avoids the question entirely.

**3. The aligner cannot use this gateway at all.** The default optimizer,
MemAlign, builds a similarity index over your labels, so it needs an embedding
model — it requests `text-embedding-3-small` *by name* and reaches it through the
litellm library, which posts to `{OPENAI_BASE_URL}/embeddings`.

**That route does not exist.** The MLflow AI Gateway serves chat at an
OpenAI-compatible path, which is why every other call in this lesson needs
nothing but a base URL, but it serves embeddings only at
`/gateway/<alias>/mlflow/invocations` — alias in the path, not in a `model`
field. The call 404s, and no gateway alias can fix it. This is the one place in
the whole tutorial where a lesson has to go around the gateway.

The escape hatch is a provider prefix that carries its **own** base URL:
litellm's `lm_studio/` reads `LM_STUDIO_API_BASE` and `LM_STUDIO_API_KEY` rather
than the `OPENAI_*` pair, so the embedding call goes straight to Unsloth while
every chat call still goes through the gateway. Without `UNSLOTH_API_KEY` in your
shell the lesson falls back to `SIMBAAlignmentOptimizer`, which needs no
embeddings but requires ≥10 labelled traces and, on a small model, often learns
nothing.

The name MemAlign asks for is OpenAI's; the model behind it is the **local**
nomic embedder. That substitution is invisible to MemAlign, which only needs
vectors it can compare — it asks for `dimensions=512`, the engine ignores the
parameter, and nomic returns its native 768 for every call alike. Consistency is
what retrieval needs, not any particular width — which is why the embedding
aliases are local-only with no fallback. A cloud embedder would answer at a
different width (the real `text-embedding-3-small` is 1536), and one hop mid-run
would put two widths into a single index.

**4. The aligner needs a chat model too, under another hardcoded name.** MemAlign
distils guidelines with `reflection_lm`, which defaults to `openai:/gpt-4.1-mini`
(`get_default_model()`). The gateway aliases that name as well. Miss it and
alignment does not crash — it logs `Invalid model name passed in
model=gpt-4.1-mini` at ERROR and returns a judge that learned nothing, which
reads as "alignment doesn't help" rather than as a misconfiguration.

**5. If `align()` dies in numpy with a shape mismatch**, e.g.
`operands could not be broadcast together ... (1,512) ... (1,12,768)`, the cause
is dspy's embedding cache in `~/.dspy_cache`, not this lesson:

```bash
rm -rf ~/.dspy_cache
```

dspy keys that cache on the model *name* plus the input text, never on which
model the name resolves to — so vectors cached before `text-embedding-3-small`
was repointed at the local embedder come back at the old width (OpenAI honours
`dimensions=512`; nomic ignores it and returns 768). The corpus text changes
every run so it misses and re-embeds at 768, while the query text is fixed and
hits the stale 512 — hence one 512 vector against a 768 index. The giveaway is
that the gateway logs *no* embedding request for the query at all: it never left
the process.

Delete the whole directory rather than picking out entries that look 512-wide —
a targeted purge misses some, since not every cached embedding is stored as a
plain list. Only machines that ran this lesson before the switch are affected.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/1_turn/2_judges
uv sync
uv run python main.py
```

Expect roughly 5 minutes — 12 agent calls, 36 judge calls and an alignment pass.

## Expected Output

```text
Step 4: REGISTERED judge (make_judge)
  registered as 'answer_helpfulness', kind=ScorerKind.INSTRUCTIONS
  naive verdicts : [True, True, True, True, True, True, True, True, True, True, True, True]
  agreement with humans: 50%

Step 5: ALIGNED judge (judge.align)
  12 traces carrying human feedback named 'answer_helpfulness'
  aligned verdicts: [True, False, True, False, True, False, True, False, ...]
  agreement with humans: 100%

  form                        agreement    registrable?
  ----------------------------------------------------
  inline @scorer                   50%        no (OSS)
  registered make_judge            50%             yes
  registered + aligned            100%             yes
```

The naive judge says `True` to everything, so it agrees with the humans exactly
half the time — the half that happens to be good. The aligned judge recovers the
alternating pattern, because it learned the rule the labels encode.

In the MLflow UI, experiment `L2/M2_agent_evaluation/1_instruments/1_turn/2_judges` holds 12 traces
each carrying a human `Feedback`, and the run `judge_forms_comparison` logs all
three agreement metrics side by side.

> [!note]
> If your alignment run reports no improvement, that is a real result, not a
> broken lesson — twelve labels is very few. The script says so rather than
> claiming a win it did not measure.

## Key Takeaways

- Inline judges are for exploration; they cannot be registered on open-source
  MLflow, so they cannot be shared, versioned, or run server-side.
- `make_judge` produces judges as *data*, which is what makes them registrable —
  and registration is the gate to everything in L2-M2.2.2.
- Alignment beats prompt-editing when you have labels: it derives the rubric
  from what reviewers actually did.
- Alignment reads only human-sourced feedback whose name matches the judge.
- Build the variance your evaluation needs into the data. A model asked to vary
  its own behaviour will not reliably do it.

## Next Steps

**L2-M2.1.1.3 (Agent Quality Metrics and Session Scorers)** moves from one rubric to
a full metric suite, and adds the multi-turn dimension: session-level scorers
like `ConversationCompleteness` and `UserFrustration` that no single-turn judge
can express. The registered judge you built here is reused in L2-M2.2.1.1 and goes
online in L2-M2.2.2.
