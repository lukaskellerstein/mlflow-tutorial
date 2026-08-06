"""L2-M2.1.2 -- Judges for Agents: Inline, Registered, Aligned.

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
from typing import cast

import mlflow
from langchain_openai import ChatOpenAI
from mlflow.entities import AssessmentSource, Feedback, Trace
from mlflow.genai.scorers import scorer
from pydantic import SecretStr

# The LiteLLM gateway from infra/, not a provider directly. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-judge"

# MLflow judges resolve their model through LiteLLM, which reads these two env
# vars. Setting them here is what lets `make_judge(model="openai:/gemma-large")`
# and the DSPy alignment optimizer both reach the gateway with no extra wiring.
# (`make_judge(base_url=...)` also works, but it wants the FULL endpoint URL --
# ".../v1/chat/completions" -- not a base. The env vars are less surprising.)
#
# These are assignments, NOT setdefault: if you have a real OPENAI_API_KEY
# exported (this machine delivers one via ~/Projects/.envrc), setdefault would
# keep it and every judge call would be rejected by the gateway with
# "Invalid proxy server token passed" -- a confusing failure a long way from
# its cause. The lesson talks to the gateway, so it states that unconditionally.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

EXPERIMENT = "L2/M2_agent_evaluation/1_instruments/2_judges"
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
    return ChatOpenAI(
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        model=MODEL_ALIAS,
        temperature=temperature,
    )


@mlflow.trace(name="support_agent")
def support_agent(question: str, cite_policy: bool) -> str:
    """One traced agent turn. Every call leaves a trace we can judge and align on."""
    system = CITING_PROMPT if cite_policy else TERSE_PROMPT
    reply = build_llm().invoke([{"role": "system", "content": system}, {"role": "user", "content": question}])
    return str(reply.content).strip()


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
    print("L2-M2.1.2  Judges for Agents: Inline, Registered, Aligned")
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

    # `locations`, not the deprecated `experiment_ids`. The cast is because the
    # return type is `DataFrame | list[Trace]` for every call -- `return_type="list"`
    # settles it at runtime but not for the type checker.
    traces = cast(
        "list[Trace]",
        mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=len(trace_ids), return_type="list"),
    )
    print(f"  {len(traces)} traces carrying human feedback named '{JUDGE_NAME}'")
    print("  running alignment (this makes many model calls -- be patient)...")

    # The default optimizer is MemAlign, which builds a similarity index over the
    # labelled examples and therefore needs an EMBEDDING model -- it requests
    # "text-embedding-3-small" by name, and make_judge gives you no way to change
    # that. infra/litellm/config.yaml exposes an alias under exactly that name for
    # this reason; without it, align() dies on a 400 from /embeddings.
    #
    # If you have no embedding model available, SIMBAAlignmentOptimizer(
    # model=f"openai:/{MODEL_ALIAS}") uses the chat model alone -- but it needs at
    # least 10 labelled traces and, on a small model, frequently learns nothing.
    aligned_judge = naive_judge.align(traces)
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
