"""
L2-M3.3 — Optimizing Against Benchmarks Without Destroying Them

People do optimize against benchmarks -- wanting the best possible SWE-Bench or
GAIA number is a legitimate goal. The trap is that a benchmark you optimize
against stops being a measurement and becomes training data. The number is then
a claim about your tuning loop, not about your agent.

The discipline that saves it is a split you never report on:

    optimize on DEV  ->  report on HELD-OUT  ->  the gap is your overfitting signal

Real benchmarks differ sharply on whether they give you one:

    GAIA                  split="validation" answers public,
                          split="test" answers WITHHELD (leaderboard-scored).
                          You can tune honestly.
    SWE-Bench Verified    split="test" ships gold patches AND the
                          FAIL_TO_PASS / PASS_TO_PASS lists. Nothing is held back,
                          so tuning against it contaminates your only measure.

  Part 1: a benchmark with an explicit dev / held-out split
  Part 2: optimize system prompts against DEV only
  Part 3: score the winner on HELD-OUT and measure the gap
  Part 4: what the gap means, and what to do about it

Builds on L2-M2.2.3 (SWE-Bench), L2-M2.2.4 (GAIA) and L2-M3.1 (Prompt Optimization).
"""

import statistics

import mlflow
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-26b-free"

EXPERIMENT = "L2/M3_agent_optimization/3_benchmark_optimization"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment(EXPERIMENT)
mlflow.langchain.autolog(log_traces=True)


# ---------------------------------------------------------------------------
# Part 1: a benchmark WITH a held-out split
# ---------------------------------------------------------------------------
# The split is deliberately constructed so overfitting is visible in 10 tasks
# instead of 500: DEV is dominated by short factual answers, HELD_OUT keeps some
# of those but adds questions that need an explanation. A prompt tuned to be
# terse therefore wins on DEV and loses on HELD_OUT -- which is exactly the
# failure mode real benchmark tuning produces, just compressed.
DEV = [
    {"q": "What is 12 * 12?", "expect": "144", "kind": "short"},
    {"q": "What is the capital of France?", "expect": "paris", "kind": "short"},
    {"q": "How many days are in a leap year?", "expect": "366", "kind": "short"},
    {"q": "What is 100 divided by 4?", "expect": "25", "kind": "short"},
    {"q": "What year did the first Moon landing happen?", "expect": "1969", "kind": "short"},
]

HELD_OUT = [
    {"q": "What is 15 * 3?", "expect": "45", "kind": "short"},
    {"q": "What is the capital of Japan?", "expect": "tokyo", "kind": "short"},
    {
        "q": "Why do experiment tracking systems record parameters alongside metrics?",
        "expect": "reproduc",
        "kind": "explain",
    },
    {
        "q": "Explain in a sentence why a model registry is useful to a team.",
        "expect": "version",
        "kind": "explain",
    },
    {
        "q": "Describe what happens during a train/test split and why it matters.",
        "expect": "generaliz",
        "kind": "explain",
    },
]

CANDIDATE_PROMPTS = {
    "baseline": "You are a helpful assistant. Answer the user's question.",
    "terse": (
        "You are a helpful assistant. Answer with the shortest possible answer -- "
        "a single number or word. Never explain."
    ),
    "balanced": (
        "You are a helpful assistant. Give the direct answer first. If the question "
        "asks why or to explain, follow with one short sentence of justification."
    ),
}


def build_llm(system_prompt: str):
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    ), system_prompt


def grade(answer: str, task: dict) -> int:
    """Correct AND appropriately scoped for the question type.

    One rule, applied identically to both splits -- it keys off the TASK kind,
    never off which split the task came from. Scoring format compliance is normal
    in real benchmarks (GAIA's exact-match grading punishes rambling the same
    way), and it is what makes the dev/held-out divergence visible here:
    a "short" task wants a short answer, an "explain" task wants a reason.
    """
    text = answer.lower().strip()
    if task["expect"].lower() not in text:
        return 0
    words = len(text.split())
    if task["kind"] == "short":
        return int(words <= 8)
    return int(words >= 12)


def score_split(system_prompt: str, split: list[dict], split_name: str) -> dict:
    """Score one prompt against one split."""
    llm, _ = build_llm(system_prompt)
    correct = 0
    per_kind: dict[str, list[int]] = {}

    for task in split:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["q"]},
            ]
        )
        # .content is typed str | list[...] because some providers return content
        # blocks; these prompts always produce plain text, so coerce for grading.
        hit = grade(str(response.content or ""), task)
        correct += hit
        per_kind.setdefault(task["kind"], []).append(hit)

    metrics = {f"{split_name}_accuracy": correct / len(split)}
    for kind, hits in per_kind.items():
        metrics[f"{split_name}_accuracy_{kind}"] = statistics.mean(hits)
    return metrics


# ---------------------------------------------------------------------------
# Part 2: optimize on DEV only
# ---------------------------------------------------------------------------
def optimize_on_dev() -> tuple[str, dict]:
    print("\n" + "=" * 60)
    print("  Part 2: Optimize against the DEV split ONLY")
    print("=" * 60)
    print("  The held-out split is not touched here. Not looked at, not scored.\n")

    dev_scores = {}
    for name, prompt in CANDIDATE_PROMPTS.items():
        with mlflow.start_run(run_name=f"dev/{name}", nested=True):
            # Tagging the split on every run is what keeps the distinction alive
            # in review. A run whose split is unrecorded is a run nobody can
            # trust six months later.
            mlflow.set_tags({"split": "dev", "prompt_variant": name})
            mlflow.log_param("system_prompt", prompt)
            metrics = score_split(prompt, DEV, "dev")
            mlflow.log_metrics(metrics)

        dev_scores[name] = metrics["dev_accuracy"]
        print(f"    {name:<10} dev_accuracy={metrics['dev_accuracy']:.0%}")

    winner = max(dev_scores, key=lambda k: dev_scores[k])
    print(f"\n  DEV winner: '{winner}' at {dev_scores[winner]:.0%}")
    print("  If we stopped here and published this number, it would be a claim")
    print("  about our tuning loop, not about the agent.")
    return winner, dev_scores


# ---------------------------------------------------------------------------
# Part 3: score on HELD-OUT
# ---------------------------------------------------------------------------
def evaluate_held_out(dev_scores: dict) -> list[dict]:
    print("\n" + "=" * 60)
    print("  Part 3: Score every candidate on the HELD-OUT split")
    print("=" * 60)
    print("  Scoring all of them (not just the winner) makes the gap visible.\n")

    rows = []
    for name, prompt in CANDIDATE_PROMPTS.items():
        with mlflow.start_run(run_name=f"heldout/{name}", nested=True):
            mlflow.set_tags({"split": "held_out", "prompt_variant": name})
            mlflow.log_param("system_prompt", prompt)
            metrics = score_split(prompt, HELD_OUT, "held_out")
            gap = dev_scores[name] - metrics["held_out_accuracy"]
            mlflow.log_metrics({**metrics, "dev_minus_heldout": gap})

        rows.append(
            {
                "variant": name,
                "dev": dev_scores[name],
                "held_out": metrics["held_out_accuracy"],
                "gap": gap,
                "short": metrics.get("held_out_accuracy_short", float("nan")),
                "explain": metrics.get("held_out_accuracy_explain", float("nan")),
            }
        )
        print(f"    {name:<10} dev={dev_scores[name]:.0%}  held_out={metrics['held_out_accuracy']:.0%}  gap={gap:+.0%}")
    return rows


# ---------------------------------------------------------------------------
# Part 4: read the gap
# ---------------------------------------------------------------------------
def report(rows: list[dict], dev_winner: str) -> None:
    print("\n" + "=" * 60)
    print("  Part 4: What the gap tells you")
    print("=" * 60)

    print(f"\n  {'variant':<12}{'dev':>7}{'held-out':>11}{'gap':>8}{'short':>8}{'explain':>9}")
    print("  " + "-" * 55)
    for r in rows:
        print(
            f"  {r['variant']:<12}{r['dev']:>6.0%}{r['held_out']:>10.0%}{r['gap']:>+8.0%}"
            f"{r['short']:>8.0%}{r['explain']:>9.0%}"
        )

    held_out_winner = max(rows, key=lambda r: r["held_out"])["variant"]
    worst_gap = max(rows, key=lambda r: r["gap"])

    print(f"\n  DEV winner      : {dev_winner}")
    print(f"  HELD-OUT winner : {held_out_winner}")
    if dev_winner != held_out_winner:
        print("\n  These disagree -- which is the entire lesson. The dev winner was")
        print("  selected partly for a quirk of the dev split, not for being better.")
    else:
        print("\n  These agree, so the dev selection held up. That is the outcome you")
        print("  want, and you only know it because a held-out split existed.")

    print(f"\n  Largest dev-minus-held-out gap: '{worst_gap['variant']}' at {worst_gap['gap']:+.0%}")
    print("  Track that gap across optimization iterations. A rising gap means you")
    print("  are fitting the split, not improving the agent -- stop, or enlarge dev.")

    print("\n  Applying this to real benchmarks:")
    print("    GAIA               tune on validation, report the leaderboard test score")
    print("    SWE-Bench Verified no held-out half exists -- carve your own dev subset")
    print("                       out of the 500 and never tune on the rest")


def main() -> None:
    print("=" * 60)
    print("  L2-M3.3 — Optimizing Against Benchmarks Without Destroying Them")
    print("=" * 60)
    print("\n  Part 1: the benchmark")
    print(f"    dev      : {len(DEV)} tasks (tuning is allowed here)")
    print(f"    held-out : {len(HELD_OUT)} tasks (reporting only, never tuned on)")
    print(f"    candidates: {list(CANDIDATE_PROMPTS)}")

    with mlflow.start_run(run_name="benchmark_optimization"):
        mlflow.log_params(
            {
                "model": MODEL_ALIAS,
                "dev_size": len(DEV),
                "held_out_size": len(HELD_OUT),
                "candidates": ",".join(CANDIDATE_PROMPTS),
            }
        )
        dev_winner, dev_scores = optimize_on_dev()
        rows = evaluate_held_out(dev_scores)

        held_out_winner = max(rows, key=lambda r: r["held_out"])
        mlflow.log_metrics(
            {
                "best_held_out_accuracy": held_out_winner["held_out"],
                "max_dev_minus_heldout": max(r["gap"] for r in rows),
            }
        )
        mlflow.set_tags({"dev_winner": dev_winner, "held_out_winner": held_out_winner["variant"]})

    report(rows, dev_winner)

    print("\n" + "=" * 60)
    print("  Done! View results in MLflow UI: http://127.0.0.1:5555")
    print(f"  Experiment: {EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
