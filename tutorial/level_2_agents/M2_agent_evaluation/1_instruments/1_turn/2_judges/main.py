"""L2-M2.1.1.2 -- Judges for Agents: Inline, Registered, Aligned.

The same rubric expressed three ways, on one dataset:

  1. INLINE     -- @scorer + a hand-built prompt. Full control, no governance.
  2. REGISTERED -- make_judge(...).register(). Named, versioned, reusable, and
                   the only form the server can run on its own.
  3. ALIGNED    -- judge.align(traces). The judge learns the standard the human
                   labels actually apply, instead of the one you guessed at.

The teaching case: a naive "is this helpful?" judge is far too generous. The
support team's real bar is stricter -- an answer only counts if it cites the
policy reference the customer needs. Alignment closes that gap without anyone
hand-editing the judge's prompt.
"""

from __future__ import annotations

import os
import re

import mlflow
from langchain_openai import ChatOpenAI
from mlflow.entities import AssessmentSource, Feedback, Trace
from mlflow.genai.judges.optimizers import MemAlignOptimizer, SIMBAAlignmentOptimizer
from mlflow.genai.scorers import scorer
from pydantic import SecretStr

# The MLflow AI Gateway -- the tracking server itself, not a provider directly.
# See L2-M1.1.1.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-judge"

# MLflow judges resolve their model through the litellm LIBRARY (which MLflow
# uses internally -- not a proxy), and it reads these two env vars. Setting them
# here is what lets `make_judge(model="openai:/gemma-judge")` and the alignment
# optimizer both reach the gateway with no extra wiring. (`make_judge(base_url=
# ...)` also works, but it wants the FULL endpoint URL -- ".../v1/chat/
# completions" -- not a base. The env vars are less surprising.)
#
# These are assignments, NOT setdefault, and BASE_URL is the one that matters:
# if you have a real OPENAI_API_KEY exported (this machine delivers one via
# ~/Projects/.envrc), setdefault would keep the real base URL too and every
# judge call would go to api.openai.com instead of to this gateway. The lesson
# talks to the gateway, so it states that unconditionally.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

# The judge ALIGNER cannot use the gateway -- see run_alignment() for the whole
# story. These two name the local engine directly, and they are the only raw
# engine details in the tutorial.
UNSLOTH_URL = "http://127.0.0.1:8888/v1"
EMBED_MODEL = "second-state/Nomic-embed-text-v1.5-Embedding-GGUF"

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/1_turn/2_judges"
JUDGE_NAME = "answer_helpfulness"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
# set_experiment returns the Experiment, so keep the id rather than looking it
# up again later with get_experiment_by_name (which is Optional and needs a None check).
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id

# SIMBA needs at least 10 labelled traces, so the dataset is 12.
QUESTIONS = [
    "How long do I have to return a laptop I bought online?",
    "Can I get a refund if I lost the receipt?",
    "Is the battery covered under the warranty?",
    "What happens if the item arrives damaged?",
    "Can I exchange a gift for a different size?",
    "Do I pay for return shipping?",
    "My headphones stopped working after three months -- what now?",
    "Can I return an opened software box?",
    "How long does a refund take to reach my card?",
    "Is accidental damage covered?",
    "Can someone else return an item on my behalf?",
    "What if I miss the return window by two days?",
]

BASE_PROMPT = "You are a retail support agent. Answer the customer's question in two sentences or fewer."
# Two prompts, alternated, so the answer set is reliably MIXED. Asking one
# prompt to vary its own behaviour does not work -- the first version of this
# lesson said "cite a policy reference about half the time" and the model
# obliged on every single answer, which made the human and the naive judge
# agree 100% and destroyed the very gap the lesson exists to show.
CITING_PROMPT = (
    f"{BASE_PROMPT} Always cite the relevant policy reference in the form "
    "P-<number>: P-101 for returns, P-204 for warranties, P-330 for shipping."
)
TERSE_PROMPT = f"{BASE_PROMPT} Never mention a policy reference, code or number."

# The human bar: an answer is only acceptable if it carries a policy reference.
# In a real lesson these labels come from your support team via
# mlflow.genai.labeling; here a regex stands in so the lesson is reproducible.
POLICY_RE = re.compile(r"\bP-\d{3}\b")


def human_label(answer: str) -> bool:
    """Stand-in for a human reviewer applying the team's real standard."""
    return bool(POLICY_RE.search(answer))


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """The agent's LLM. The token cap is load-bearing; see support_agent().

    Gemma 4 reasons before it answers, and the reasoning is charged against
    max_tokens. The gateway cannot supply a default -- an endpoint holds a model
    and a fallback chain and nothing else -- so every caller sets its own.

    Reasoning stays ON here, which is the model's normal behaviour. The cost is
    that an answer occasionally comes back empty; support_agent() retries. To
    turn it off instead, add
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    which Unsloth advertises at /v1/status as `reasoning_style`. Measured on
    this prompt: 1170 completion tokens with reasoning on, 31 with it off. The
    answers differ in more than length, which is why it is not the default here.
    """
    return ChatOpenAI(
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        model=MODEL_ALIAS,
        temperature=temperature,
        # `max_completion_tokens`, not `max_tokens`: the field is called
        # max_tokens on ChatOpenAI but carries that alias, so the plain name is
        # a type error even though pydantic accepts it at runtime.
        max_completion_tokens=4096,
    )


@mlflow.trace(name="support_agent")
def support_agent(question: str, cite_policy: bool) -> str:
    """One traced agent turn. Every call leaves a trace we can judge and align on.

    THE RETRY RAISES THE TEMPERATURE, and that detail is the whole fix.

    Gemma 4 reasons before it answers, and the reasoning is charged against
    max_tokens. At `temperature=0.0` this one question sends it into a reasoning
    loop it never leaves: the call returns HTTP 200 with `content: ""` and
    `finish_reason: "length"`, having spent the entire budget thinking. A bigger
    budget does not help -- measured at 4096, 8192 and 16384, it burned all
    three and answered nothing every time.

    Greedy decoding is what traps it, so the escape is to stop being greedy. The
    same request at `temperature=0.7` answered in 793 tokens and at 1.0 in 1364.
    A plain retry cannot work: at temperature 0 the second call is the first
    call.

    Why bother rather than turning reasoning off: one empty answer fails Step 5
    outright. MemAlign turns each trace into (inputs, outputs, feedback), an
    empty output resolves to zero examples, and it refuses the WHOLE batch
    rather than skipping the trace -- deliberately, so a wrong trace id cannot
    hide among good ones.
    """
    system = CITING_PROMPT if cite_policy else TERSE_PROMPT
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    for attempt, temperature in enumerate((0.0, 0.7, 1.0), start=1):
        reply = build_llm(temperature).invoke(messages)
        if answer := str(reply.content).strip():
            return answer
        finish = reply.response_metadata.get("finish_reason")
        print(f"    empty answer (finish_reason={finish}) -- retrying {attempt}/3 with a higher temperature")

    raise RuntimeError(
        f"The model returned an empty answer three times for {question!r}, at three\n"
        "different temperatures. Gemma 4's reasoning block is spending the whole\n"
        "max_tokens budget and never terminating. The way out is to stop it\n"
        "reasoning for this call:\n"
        '  extra_body={"chat_template_kwargs": {"enable_thinking": False}}\n'
        "which answered the same question in 31 tokens instead of 1170."
    )


# ── 1. The inline judge ─────────────────────────────────────────────────────── #

INLINE_PROMPT = """\
You are grading a retail support answer.

Question: {question}
Answer: {answer}

Is the answer helpful? Reply with exactly one word: yes or no."""


@scorer
def inline_helpfulness(inputs: dict, outputs: dict) -> Feedback:
    """A judge you build yourself: your prompt, your parsing, your Feedback."""
    verdict = build_llm().invoke(INLINE_PROMPT.format(question=inputs["question"], answer=outputs["answer"]))
    said_yes = "yes" in str(verdict.content).strip().lower()[:5]
    return Feedback(
        value=said_yes,
        rationale=f"inline judge replied: {str(verdict.content).strip()[:80]}",
        source=AssessmentSource(source_type="LLM_JUDGE", source_id=MODEL_ALIAS),
    )


def verdict_of(judged: object) -> bool:
    """Coerce a judge's answer to bool.

    `Scorer.__call__` is annotated as the wide union
    `int | float | bool | str | Feedback | list[Feedback]` because scorers may
    return a bare value. Every judge in this lesson returns a Feedback, but the
    type checker cannot know that, so narrow it once here instead of reaching
    for `.value` on the union at five call sites.
    """
    return bool(judged.value) if isinstance(judged, Feedback) else bool(judged)


def agreement(predicted: list[bool], human: list[bool]) -> float:
    return sum(p == h for p, h in zip(predicted, human)) / len(human)


def main() -> None:
    print("=" * 70)
    print("L2-M2.1.1.2  Judges for Agents: Inline, Registered, Aligned")
    print("=" * 70)

    # ── Produce something to judge ──────────────────────────────────────────── #
    print("\nStep 1: running the support agent to produce traces")
    answers: list[str] = []
    trace_ids: list[str] = []
    for i, q in enumerate(QUESTIONS):
        answer = support_agent(q, cite_policy=(i % 2 == 0))
        answers.append(answer)
        trace_ids.append(mlflow.get_last_active_trace_id() or "")
        print(f"  Q: {q[:50]:<50} -> {answer[:42]}")

    humans = [human_label(a) for a in answers]
    print(f"\n  Human labels (cites a policy reference): {humans}")
    print(f"  {sum(humans)}/{len(humans)} answers meet the team's real bar")

    # ── 2. Inline judge ─────────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 2: INLINE judge (@scorer)")
    print("=" * 70)
    inline_scores = [
        verdict_of(inline_helpfulness(inputs={"question": q}, outputs={"answer": a}))
        for q, a in zip(QUESTIONS, answers)
    ]
    inline_agreement = agreement(inline_scores, humans)
    print(f"  inline verdicts : {inline_scores}")
    print(f"  agreement with humans: {inline_agreement:.0%}")

    # ── 3. What inline CANNOT do ────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 3: try to REGISTER the inline judge -- expect a refusal")
    print("=" * 70)
    try:
        inline_helpfulness.register(name="inline_helpfulness")
        print("  registered (you are on Databricks)")
    except Exception as exc:
        print(f"  refused, as designed:\n    {str(exc).splitlines()[0][:150]}")
        print(
            "\n  A @scorer is deserialized by exec()ing its source, so the open-source\n"
            "  server refuses to store one. This is the practical dividing line:\n"
            "  inline judges stay in your script; only make_judge/builtin judges\n"
            "  become server-side objects."
        )

    # ── 4. Registered judge ─────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 4: REGISTERED judge (make_judge)")
    print("=" * 70)
    naive_judge = mlflow.genai.make_judge(
        name=JUDGE_NAME,
        instructions=(
            "You are grading a retail support answer.\n"
            "The question is in {{ inputs }} and the answer is in {{ outputs }}.\n"
            "Is the answer helpful? Answer true or false."
        ),
        model=f"openai:/{MODEL_ALIAS}",
        feedback_value_type=bool,
    )
    registered = naive_judge.register(name=JUDGE_NAME)
    print(f"  registered as {JUDGE_NAME!r}, kind={registered.kind}")
    # Count, not names. A Scorer returned by list_scorers() carries the judge's
    # INTERNAL name in `.name`, which is not necessarily the name it was
    # registered under -- the aligned judge below registers as
    # "answer_helpfulness_aligned" but still reports `.name == "answer_helpfulness"`.
    # Registered names come from the REST route /api/3.0/mlflow/scorers/list.
    print(f"  scorers now on this experiment: {len(mlflow.genai.list_scorers())}")

    naive_scores = [
        verdict_of(naive_judge(inputs={"question": q}, outputs={"answer": a})) for q, a in zip(QUESTIONS, answers)
    ]
    naive_agreement = agreement(naive_scores, humans)
    print(f"  naive verdicts : {naive_scores}")
    print(f"  agreement with humans: {naive_agreement:.0%}")
    print("  ^ the naive judge calls almost everything helpful -- it has no idea")
    print("    the team requires a policy reference. Nobody told it.")

    # ── 5. Alignment ────────────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 5: ALIGNED judge (judge.align)")
    print("=" * 70)
    print("  attaching human labels to the traces...")
    for trace_id, label in zip(trace_ids, humans):
        if not trace_id:
            continue
        mlflow.log_feedback(
            trace_id=trace_id,
            name=JUDGE_NAME,  # must match the judge being aligned
            value=label,
            source=AssessmentSource(source_type="HUMAN", source_id="support-team"),
            rationale="acceptable only if it cites a policy reference",
        )

    # FETCH BY ID, AND FLUSH. Two separate traps, and together they make this
    # lesson fail on its SECOND run while passing on its first.
    #
    # 1. `search_traces(max_results=len(trace_ids))` returns the newest N traces
    #    in the EXPERIMENT, not the N this run just labelled. Run the lesson
    #    twice and that window straddles two runs, so align() gets a trace from
    #    the earlier one that carries no label and dies with
    #    "No valid feedback records found for 1 trace(s)".
    # 2. `log_feedback` goes through the async export queue, so a read taken
    #    immediately afterwards can miss the write that has not landed yet.
    #    `flush=True` waits for the queue instead of guessing at a sleep.
    traces: list[Trace] = []
    for trace_id in trace_ids:
        if not trace_id:
            continue
        if (trace := mlflow.get_trace(trace_id, flush=True)) is not None:
            traces.append(trace)
    print(f"  {len(traces)} traces carrying human feedback named '{JUDGE_NAME}'")
    print("  running alignment (this makes many model calls -- be patient)...")

    # THE ONE PLACE IN THIS TUTORIAL THAT GOES AROUND THE GATEWAY, and it is
    # worth understanding rather than copying.
    #
    # The default optimizer is MemAlign. It builds a similarity index over the
    # labelled examples, so it needs an EMBEDDING model, and it reaches one
    # through the litellm library -- which posts to {OPENAI_BASE_URL}/embeddings.
    #
    # That route does not exist. The MLflow AI Gateway serves chat at an
    # OpenAI-compatible path, which is why every other call in this lesson needs
    # nothing but a base URL, but it serves embeddings ONLY at
    # /gateway/<alias>/mlflow/invocations -- alias in the path, not in a "model"
    # field. So the call 404s, and no gateway alias can fix it.
    #
    # The escape hatch is a provider prefix that carries its OWN base URL:
    # litellm's `lm_studio/` reads LM_STUDIO_API_BASE and LM_STUDIO_API_KEY
    # instead of the OPENAI_* pair, so the embedding call can go straight to
    # Unsloth while every chat call above still goes through the gateway.
    #
    # Without the key, fall back to SIMBA, which uses the chat model alone. It
    # needs at least 10 labelled traces and, on a small model, frequently learns
    # nothing -- so the run stays honest rather than failing.
    unsloth_key = os.environ.get("UNSLOTH_API_KEY", "")
    if unsloth_key:
        os.environ["LM_STUDIO_API_BASE"] = UNSLOTH_URL
        os.environ["LM_STUDIO_API_KEY"] = unsloth_key
        optimizer = MemAlignOptimizer(embedding_model=f"lm_studio:/{EMBED_MODEL}")
        print(f"  optimizer: MemAlign, embeddings direct to {UNSLOTH_URL}")
    else:
        optimizer = SIMBAAlignmentOptimizer(model=f"openai:/{MODEL_ALIAS}")
        print("  optimizer: SIMBA -- no UNSLOTH_API_KEY, so no embedding model")
    aligned_judge = naive_judge.align(traces, optimizer)
    aligned_scores = [
        verdict_of(aligned_judge(inputs={"question": q}, outputs={"answer": a})) for q, a in zip(QUESTIONS, answers)
    ]
    aligned_agreement = agreement(aligned_scores, humans)
    print(f"  aligned verdicts: {aligned_scores}")
    print(f"  agreement with humans: {aligned_agreement:.0%}")

    aligned_judge.register(name=f"{JUDGE_NAME}_aligned")
    print(f"  registered aligned judge as '{JUDGE_NAME}_aligned'")

    # ── 6. Verdict ──────────────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("Step 6: the three forms side by side")
    print("=" * 70)
    with mlflow.start_run(run_name="judge_forms_comparison"):
        mlflow.log_params({"model": MODEL_ALIAS, "cases": len(QUESTIONS), "judge_name": JUDGE_NAME})
        mlflow.log_metrics(
            {
                "inline_agreement": inline_agreement,
                "registered_naive_agreement": naive_agreement,
                "registered_aligned_agreement": aligned_agreement,
            }
        )

    print(f"\n  {'form':<26}{'agreement':>11}   {'registrable?':>13}")
    print(f"  {'-' * 52}")
    print(f"  {'inline @scorer':<26}{inline_agreement:>10.0%}   {'no (OSS)':>13}")
    print(f"  {'registered make_judge':<26}{naive_agreement:>10.0%}   {'yes':>13}")
    print(f"  {'registered + aligned':<26}{aligned_agreement:>10.0%}   {'yes':>13}")

    # Report what was measured. An alignment run that did not help is a real
    # result about this judge, this model and these 12 labels -- not something
    # to paper over with a closing sentence that claims otherwise.
    delta = aligned_agreement - naive_agreement
    if delta > 0.01:
        print(
            f"\n  Alignment closed {delta:.0%} of the gap: the judge learned the standard"
            "\n  the humans apply, with nobody rewriting its instructions by hand."
        )
    elif delta < -0.01:
        print(f"\n  Alignment made this judge WORSE by {-delta:.0%}. Worth investigating")
        print("  before trusting it: too few labels, or labels that disagree with")
        print("  each other, both produce this.")
    else:
        print("\n  Alignment did not move this judge. That is a normal outcome with")
        print("  ~12 labels on a small model -- the honest read is 'not enough signal',")
        print("  not 'alignment does not work'. Add labels and re-run.")
    print(f"\n  MLflow UI: http://localhost:5555  ->  experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
