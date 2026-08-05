# L2-M2.1.2 — Judges for Agents: Inline, Registered, Aligned

**Level:** AI Agents
**Duration:** 90 min

## Overview

There are three ways to express an LLM-as-judge in MLflow, and they are not
interchangeable. This lesson builds the *same* rubric all three ways on one
dataset, shows the one thing an inline judge structurally cannot do, and then
lets a judge learn a standard nobody ever wrote into its prompt.

## Prerequisites

- Completed: L2-M2.1.1 (Agent Test Generation and Simulation), L1-M4.1.1 (Evaluation
  Fundamentals)
- MLflow server running at <http://127.0.0.1:5555>
- LiteLLM gateway running at <http://localhost:4000> (`cd infra && podman compose up -d`)
- An `OPENROUTER_API_KEY` in `infra/.env` — both the `gemma-large` and
  `text-embedding-3-small` aliases route there

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
    model="openai:/gemma-large",
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

## Gateway wiring — two traps

Both cost real debugging time, and neither error message points at its cause.

**1. `setdefault` is the wrong verb for `OPENAI_API_KEY`.** If you already
export a real OpenAI key, `os.environ.setdefault` keeps it and every judge call
dies at the gateway with `Invalid proxy server token passed`. The lesson assigns
unconditionally.

**2. `make_judge(base_url=...)` wants the FULL endpoint URL.** Not
`http://localhost:4000/v1` (404) and not `http://localhost:4000` (405), but
`http://localhost:4000/v1/chat/completions`. Setting `OPENAI_BASE_URL` instead
avoids the question entirely, and routes the alignment optimizer too.

**3. The aligner needs an embedding model.** The default optimizer, MemAlign,
builds a similarity index over your labels and requests `text-embedding-3-small`
*by name*, with no way to override it from `make_judge`. `infra/litellm/config.yaml`
exposes an alias under exactly that name for this reason. Without it you get a
400 from `/embeddings`. `SIMBAAlignmentOptimizer` needs no embeddings but
requires ≥10 labelled traces and, on a small model, often learns nothing.

## Running the Lesson

```bash
cd tutorial/level_2_agents/M2_agent_evaluation/1_instruments/2_judges
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

In the MLflow UI, experiment `L2/M2_agent_evaluation/1_instruments/2_judges` holds 12 traces
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

**L2-M2.1.3 (Agent Quality Metrics and Session Scorers)** moves from one rubric to
a full metric suite, and adds the multi-turn dimension: session-level scorers
like `ConversationCompleteness` and `UserFrustration` that no single-turn judge
can express. The registered judge you built here is reused in L2-M2.2.1 and goes
online in L2-M2.2.2.
