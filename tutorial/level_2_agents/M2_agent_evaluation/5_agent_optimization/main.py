"""
L2-M2.5 — Agent Optimization

Systematically optimize a ReAct agent across three dimensions and track
the improvement trajectory in MLflow:

  1. System prompt — minimal vs detailed vs structured
  2. Temperature  — 0.0, 0.3, 0.7, 1.0
  3. Tool descriptions — original (terse) vs improved (detailed)

Each variant is evaluated on the same 5-question benchmark.  Results are
logged as nested MLflow runs and summarized in a final comparison table
that identifies the best configuration.
"""

import os
import time
from typing import Any, cast

import mlflow
import mlflow.langchain
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.optimize import MetaPromptOptimizer
from mlflow.genai.scorers import scorer
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
# The LiteLLM gateway from infra/, not a provider directly. "gemma-large" is an
# alias defined in infra/litellm/config.yaml -- swapping model or provider is a
# change there, never here. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-large"

# The optimizer's reflection model resolves through LiteLLM, which reads these.
# Assignments, not setdefault -- a real OPENAI_API_KEY in the environment would
# win and every reflection call would be rejected by the gateway. See L2-M2.2.
os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("L2/M2_agent_evaluation/4_agent_optimization")
mlflow.langchain.autolog(log_traces=True)

# ---------------------------------------------------------------------------
# Shared tool logic
# ---------------------------------------------------------------------------
CONVERSIONS = {
    "km_to_miles": 0.621371,
    "miles_to_km": 1.60934,
    "kg_to_lbs": 2.20462,
    "lbs_to_kg": 0.453592,
    "c_to_f": lambda c: c * 9 / 5 + 32,
    "f_to_c": lambda f: (f - 32) * 5 / 9,
}

FACTS = {
    "python": "Python was created by Guido van Rossum and released in 1991.",
    "earth": "Earth is the third planet from the Sun with a diameter of 12,742 km.",
    "water": "Water boils at 100C (212F) at standard atmospheric pressure.",
    "light": "The speed of light in vacuum is 299,792,458 meters per second.",
    "moon": "The Moon orbits Earth at an average distance of 384,400 km.",
}


def _do_calc(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))  # nosec: arithmetic-only, builtins stripped
    except Exception as exc:
        return f"Error: {exc}"


def _do_convert(query: str) -> str:
    q = query.lower().strip()
    for key, factor in CONVERSIONS.items():
        if key.replace("_", " ") in q or key in q:
            nums = [float(s) for s in q.split() if s.replace(".", "").replace("-", "").isdigit()]
            if nums:
                val = nums[0]
                res = factor(val) if callable(factor) else val * factor
                src, tgt = key.split("_to_")
                return f"{val} {src} = {res:.2f} {tgt}"
    return "Supported: '10 km_to_miles', '100 c_to_f', '5 kg_to_lbs'"


def _do_lookup(topic: str) -> str:
    key = topic.strip().lower()
    for k, v in FACTS.items():
        if k in key or key in k:
            return v
    return f"No fact found for '{topic}'. Known: {', '.join(FACTS)}."


# ---- Original tools (terse descriptions) ----
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return _do_calc(expression)


@tool
def unit_converter(query: str) -> str:
    """Convert between units."""
    return _do_convert(query)


@tool
def fact_lookup(topic: str) -> str:
    """Look up a fact."""
    return _do_lookup(topic)


ORIGINAL_TOOLS = [calculator, unit_converter, fact_lookup]


# ---- Improved tools (detailed descriptions) ----
@tool
def calculator_v2(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Use this for ANY arithmetic: addition, subtraction, multiplication,
    division, exponents. Pass a valid Python math expression.

    Args:
        expression: e.g. '2 + 3 * 4' or '100 / 7'.
    """
    return _do_calc(expression)


@tool
def unit_converter_v2(query: str) -> str:
    """Convert a numeric value between measurement units.

    Supported: km<->miles, kg<->lbs, C<->F. Include number and type.

    Args:
        query: e.g. '10 km_to_miles' or '100 c_to_f'.
    """
    return _do_convert(query)


@tool
def fact_lookup_v2(topic: str) -> str:
    """Look up a factual piece of information about a well-known topic.

    Use for facts about planets, programming languages, or constants.
    Do NOT use for math or unit conversions.

    Args:
        topic: e.g. 'python', 'earth', 'water'.
    """
    return _do_lookup(topic)


IMPROVED_TOOLS = [calculator_v2, unit_converter_v2, fact_lookup_v2]

# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
EVAL_CASES = [
    {"question": "What is 15 * 24 + 100?", "expected": "460", "needs_tool": "calculator"},
    {"question": "Convert 42 km to miles.", "expected": "26.1", "needs_tool": "converter"},
    {"question": "What is the speed of light?", "expected": "299", "needs_tool": "fact"},
    {"question": "How many pounds is 10 kg?", "expected": "22.0", "needs_tool": "converter"},
    {"question": "Who created Python?", "expected": "Guido", "needs_tool": "fact"},
    # The two below exist to leave the optimizer some headroom. The five above
    # are all answerable from memory -- the model knows who created Python and
    # can do 15*24 in its head -- so a bare "You are a helpful assistant" already
    # scores 1.0 on them, and an optimizer cannot improve on a saturated
    # baseline. These two need the tool's precision to hit the expected string.
    {"question": "What is 4839 * 271?", "expected": "1311369", "needs_tool": "calculator"},
    {"question": "Convert 137 km to miles.", "expected": "85.13", "needs_tool": "converter"},
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "minimal": "You are a helpful assistant.",
    "detailed": (
        "You are a helpful assistant with access to tools for math calculations, "
        "unit conversions, and factual lookups. Always use the appropriate tool "
        "rather than guessing. Provide concise, accurate answers."
    ),
    "structured": (
        "You are a precise assistant. Follow these rules:\n"
        "1. For math questions, ALWAYS use the calculator tool.\n"
        "2. For unit conversions, ALWAYS use the unit_converter tool.\n"
        "3. For factual questions, ALWAYS use the fact_lookup tool.\n"
        "4. Report the tool result directly — do not guess or recalculate.\n"
        "5. Keep your final answer concise (one sentence)."
    ),
}

# ---------------------------------------------------------------------------
# Scoring and evaluation helpers
# ---------------------------------------------------------------------------
TOOL_MAP = {"calculator": "calculator", "converter": "unit_converter", "fact": "fact_lookup"}


def score_answer(answer: str, expected: str) -> float:
    return 1.0 if expected.lower() in answer.lower() else 0.0


def score_tool_selection(messages: list, needs_tool: str) -> float:
    expected_kw = TOOL_MAP.get(needs_tool, needs_tool)
    for msg in messages:
        if msg.type == "tool":
            name = getattr(msg, "name", "").lower()
            if expected_kw in name or expected_kw.split("_")[0] in name:
                return 1.0
    return 0.0


def run_agent_eval(agent, cases: list[dict], label: str) -> list[dict]:
    rows = []
    for i, case in enumerate(cases, 1):
        start = time.time()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": case["question"]}]},
                config={"recursion_limit": 10},
            )
            messages = result["messages"]
            answer = messages[-1].content if messages else ""
        except Exception as exc:
            answer, messages = f"Error: {exc}", []
        latency = time.time() - start
        rows.append(
            {
                "variant": label,
                "case": i,
                "question": case["question"],
                "answer": answer[:120],
                "correctness": score_answer(answer, case["expected"]),
                "tool_selection": score_tool_selection(messages, case["needs_tool"]),
                "latency_s": round(latency, 3),
            }
        )
    return rows


def log_and_print(label: str, rows: list[dict], params: dict) -> dict:
    avg_c = sum(r["correctness"] for r in rows) / len(rows)
    avg_t = sum(r["tool_selection"] for r in rows) / len(rows)
    avg_l = sum(r["latency_s"] for r in rows) / len(rows)
    quality = (avg_c + avg_t) / 2
    mlflow.log_params(params)
    mlflow.log_metrics(
        {
            "avg_correctness": round(avg_c, 3),
            "avg_tool_selection": round(avg_t, 3),
            "avg_latency_s": round(avg_l, 3),
            "quality_score": round(quality, 3),
        }
    )
    for r in rows:
        status = "PASS" if r["correctness"] == 1.0 else "FAIL"
        print(f"    [{status}] Q{r['case']}: {r['question'][:45]}")
        print(f"           ToolSel={r['tool_selection']:.0f}  Latency={r['latency_s']:.2f}s")
    agg = {
        "correctness": round(avg_c, 3),
        "tool_selection": round(avg_t, 3),
        "latency_s": round(avg_l, 3),
        "quality": round(quality, 3),
    }
    print(
        f"    ---- Aggregates: correctness={agg['correctness']:.3f}  "
        f"tool_sel={agg['tool_selection']:.3f}  quality={agg['quality']:.3f}  "
        f"latency={agg['latency_s']:.3f}s"
    )
    return agg


def run_variant(label: str, agent, dimension: str, params: dict, all_results: list[dict]) -> None:
    print(f"\n  >> {label}")
    with mlflow.start_run(run_name=label, nested=True):
        rows = run_agent_eval(agent, EVAL_CASES, label)
        agg = log_and_print(label, rows, params)
        all_results.append({"variant": label, "dimension": dimension, **agg})


# ---------------------------------------------------------------------------
# Part 4: automated prompt optimization
#
# Parts 1-3 are a hand-built grid: a human wrote three candidate prompts, picked
# the winner, then swept temperature and tool descriptions around it. That is
# the honest baseline, and it has the same ceiling as a hand-written test suite --
# it only ever explores what somebody thought to type.
#
# `mlflow.genai.optimize_prompts` searches instead. Two things make it work:
#   * the prompt must live in the PROMPT REGISTRY, not in a Python string. The
#     optimizer rewrites registered versions, so it takes `prompt_uris`.
#   * `predict_fn` must call `PromptVersion.format()` at run time. That call is
#     the hook -- it is how a candidate template reaches the agent. Inlining the
#     prompt text instead means the optimizer rewrites something nothing reads.
# ---------------------------------------------------------------------------
PROMPT_NAME = "agent_system_prompt"
PROMPT_URI = ""  # set by register_base_prompt() at run time


def register_base_prompt() -> str:
    """Put the weakest prompt in the registry as the optimizer's starting point."""
    version = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=SYSTEM_PROMPTS["minimal"],
        commit_message="L2-M2.5 starting point: the minimal prompt",
    )
    return f"prompts:/{PROMPT_NAME}/{version.version}"


def optimize_predict_fn(question: str) -> str:
    """The function being optimized. Loads the prompt rather than hardcoding it."""
    prompt = mlflow.genai.load_prompt(PROMPT_URI)
    # `format()` is typed as returning PromptVersion | str | list[dict] because a
    # prompt can be chat-shaped; this one is a plain template, so narrow to str.
    system_prompt = str(prompt.format())
    agent = create_agent(
        model=ChatOpenAI(
            model=MODEL_ALIAS,
            base_url=GATEWAY_URL,
            api_key=SecretStr(GATEWAY_KEY),
            temperature=0.0,
        ),
        tools=ORIGINAL_TOOLS,
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)


def _expected_from(expectations: Any) -> str:
    """Dig the expected answer out of whatever shape the harness passes."""
    if isinstance(expectations, dict):
        for key in ("expected", "expected_response", "outputs", "value"):
            if key in expectations and expectations[key]:
                return str(expectations[key])
        return " ".join(str(v) for v in expectations.values())
    return str(expectations or "")


@scorer
def answer_correct(outputs: Any, expectations: Any = None) -> Feedback:
    """Did the answer contain the expected value?

    Deliberately the same cheap check Part 1 uses. The point of this lesson is
    whether an optimizer beats a human at writing the prompt -- holding the
    yardstick constant is what makes that comparison mean anything.
    """
    predicted = outputs if isinstance(outputs, str) else str(outputs)
    expected = _expected_from(expectations)
    hit = bool(expected) and expected.lower() in predicted.lower()
    return Feedback(
        value=1.0 if hit else 0.0,
        rationale=f"expected {expected!r} in answer" if expected else "no expectation supplied",
        source=AssessmentSource(source_type="CODE", source_id="answer_correct"),
    )


def run_automated_optimization() -> dict[str, Any]:
    """Register a prompt, let MetaPrompt rewrite it, and report what changed."""
    global PROMPT_URI
    PROMPT_URI = register_base_prompt()
    print(f"  registered {PROMPT_URI}")
    print(f"  starting template: {SYSTEM_PROMPTS['minimal']!r}")

    train_data = [{"inputs": {"question": c["question"]}, "outputs": c["expected"]} for c in EVAL_CASES]

    # MetaPrompt, not GEPA: GepaPromptOptimizer defaults to 100 metric calls,
    # which on a free tier is a very long lesson for the same teaching point.
    optimizer = MetaPromptOptimizer(reflection_model=f"openai:/{MODEL_ALIAS}")
    print("  optimizing (the reflection model rewrites the prompt, then re-scores)...")
    result = mlflow.genai.optimize_prompts(
        predict_fn=optimize_predict_fn,
        train_data=train_data,
        prompt_uris=[PROMPT_URI],
        optimizer=optimizer,
        scorers=[answer_correct],
    )
    return {"result": result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  L2-M2.5 — Agent Optimization")
    print("  A hand-built grid, then an optimizer, on the same yardstick")
    print("=" * 70)

    all_results: list[dict] = []

    with mlflow.start_run(run_name="agent_optimization") as parent:
        mlflow.set_tags(
            {
                "optimization_type": "systematic",
                "model": MODEL_ALIAS,
                "num_test_cases": str(len(EVAL_CASES)),
            }
        )

        # Part 1: System prompt optimization
        print(f"\n{'=' * 70}")
        print("  Part 1: System Prompt Optimization")
        print(f"{'=' * 70}")
        for name, text in SYSTEM_PROMPTS.items():
            agent = create_agent(
                model=ChatOpenAI(
                    model=MODEL_ALIAS,
                    base_url=GATEWAY_URL,
                    api_key=SecretStr(GATEWAY_KEY),
                    temperature=0.0,
                ),
                tools=ORIGINAL_TOOLS,
                system_prompt=text,
            )
            run_variant(
                f"prompt_{name}",
                agent,
                "system_prompt",
                {
                    "dimension": "system_prompt",
                    "variant": name,
                    "model": MODEL_ALIAS,
                    "temperature": "0.0",
                },
                all_results,
            )

        # Part 2: Temperature tuning (using structured prompt)
        print(f"\n{'=' * 70}")
        print("  Part 2: Temperature Tuning")
        print(f"{'=' * 70}")
        best_prompt = SYSTEM_PROMPTS["structured"]
        for temp in [0.0, 0.3, 0.7, 1.0]:
            agent = create_agent(
                model=ChatOpenAI(
                    model=MODEL_ALIAS,
                    base_url=GATEWAY_URL,
                    api_key=SecretStr(GATEWAY_KEY),
                    temperature=temp,
                ),
                tools=ORIGINAL_TOOLS,
                system_prompt=best_prompt,
            )
            run_variant(
                f"temp_{temp}",
                agent,
                "temperature",
                {
                    "dimension": "temperature",
                    "variant": str(temp),
                    "model": MODEL_ALIAS,
                    "temperature": str(temp),
                },
                all_results,
            )

        # Part 3: Tool description optimization
        print(f"\n{'=' * 70}")
        print("  Part 3: Tool Description Optimization")
        print(f"{'=' * 70}")
        for tlabel, tools in [
            ("tools_original", ORIGINAL_TOOLS),
            ("tools_improved", IMPROVED_TOOLS),
        ]:
            agent = create_agent(
                model=ChatOpenAI(
                    model=MODEL_ALIAS,
                    base_url=GATEWAY_URL,
                    api_key=SecretStr(GATEWAY_KEY),
                    temperature=0.0,
                ),
                tools=tools,
                system_prompt=best_prompt,
            )
            run_variant(
                tlabel,
                agent,
                "tool_descriptions",
                {
                    "dimension": "tool_descriptions",
                    "variant": tlabel,
                    "model": MODEL_ALIAS,
                    "temperature": "0.0",
                },
                all_results,
            )

        # Part 4: hand off to the optimizer, same yardstick
        print(f"\n{'=' * 70}")
        print("  Part 4: Automated prompt optimization (mlflow.genai.optimize_prompts)")
        print(f"{'=' * 70}")
        # Compare CORRECTNESS, not quality. The optimizer is scored by
        # `answer_correct` -- correctness alone -- while the grid's "quality" is a
        # blend of correctness and tool selection. Putting 0.929 (quality) beside
        # the optimizer's 1.000 (correctness) would be two different scales
        # pretending to be one, and would flatter whichever side you wanted.
        hand_built_best = max((r["correctness"] for r in all_results), default=0.0)
        hand_built_best_quality = max((r["quality"] for r in all_results), default=0.0)
        optimized = run_automated_optimization()
        opt_result = optimized["result"]

        optimized_prompt = ""
        for prompt in getattr(opt_result, "optimized_prompts", []) or []:
            optimized_prompt = str(getattr(prompt, "template", prompt))
        before = getattr(opt_result, "initial_eval_score", None)
        after = getattr(opt_result, "final_eval_score", None)

        print(f"\n  optimizer ({getattr(opt_result, 'optimizer_name', '?')}):")
        print(f"    score before : {before}")
        print(f"    score after  : {after}")
        print(f"  hand-built grid, best correctness : {hand_built_best:.3f}  (same scale as above)")
        print(f"  hand-built grid, best quality     : {hand_built_best_quality:.3f}  (blended, not comparable)")
        if optimized_prompt:
            print(f"\n  the optimizer rewrote the prompt as:\n    {' '.join(optimized_prompt.split())[:400]}")

        # Report the measured outcome, including the boring one. An optimizer
        # that changed nothing is a real result -- and the usual cause is a
        # saturated baseline, not a broken optimizer.
        if before is not None and after is not None:
            delta = float(after) - float(before)
            if delta > 0.01:
                print(f"\n  The optimizer improved the prompt by {delta:+.3f} without anyone editing it.")
            elif float(before) >= 0.999:
                print(
                    "\n  No improvement -- but look at the starting score: the baseline was"
                    "\n  already perfect, so there was nothing left to win. A saturated"
                    "\n  benchmark cannot rank prompts. Fix the benchmark before reaching"
                    "\n  for an optimizer; that is the real lesson here."
                )
            elif delta < -0.01:
                print(f"\n  The optimizer made it WORSE by {delta:.3f} -- too few examples to generalise from.")
            else:
                print(f"\n  The optimizer moved the score by {delta:+.3f}: no meaningful change.")

        mlflow.log_metrics(
            {
                "hand_built_best_correctness": hand_built_best,
                "hand_built_best_quality": hand_built_best_quality,
                **({"optimizer_score_before": float(before)} if before is not None else {}),
                **({"optimizer_score_after": float(after)} if after is not None else {}),
            }
        )

        # Part 5: Optimization summary
        print(f"\n{'=' * 70}")
        print("  Optimization Summary")
        print(f"{'=' * 70}")
        df = pd.DataFrame(all_results)

        print(f"\n  {'Variant':<22} {'Dimension':<18} {'Correct':>8} {'ToolSel':>8} {'Quality':>8} {'Latency':>8}")
        print("  " + "-" * 74)
        for _, row in df.iterrows():
            print(
                f"  {row['variant']:<22} {row['dimension']:<18} "
                f"{row['correctness']:>8.3f} {row['tool_selection']:>8.3f} "
                f"{row['quality']:>8.3f} {row['latency_s']:>7.3f}s"
            )

        best = df.loc[df["quality"].idxmax()]
        print(f"\n  BEST CONFIGURATION: {best['variant']}")
        print(
            f"    Quality={best['quality']:.3f}  Correctness={best['correctness']:.3f}  "
            f"ToolSelection={best['tool_selection']:.3f}  Latency={best['latency_s']:.3f}s"
        )

        for step, (_, row) in enumerate(df.iterrows()):
            mlflow.log_metric("opt_quality", float(row["quality"]), step=step)
            mlflow.log_metric("opt_correctness", float(row["correctness"]), step=step)
            mlflow.log_metric("opt_tool_selection", float(row["tool_selection"]), step=step)

        mlflow.log_params(
            {
                "best_variant": best["variant"],
                "best_quality": best["quality"],
                "num_variants_tested": len(all_results),
            }
        )

        for dim in df["dimension"].unique():
            dim_best = cast(pd.DataFrame, df[df["dimension"] == dim]).sort_values("quality", ascending=False).iloc[0]
            mlflow.set_tag(f"best_{dim}", dim_best["variant"])
            print(f"  Best {dim}: {dim_best['variant']} (quality={dim_best['quality']:.3f})")

        csv_path = "optimization_summary.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        print(f"\n  Parent Run ID: {parent.info.run_id}")
        print("  View in MLflow UI: http://127.0.0.1:5555")

    if os.path.exists(csv_path):
        os.remove(csv_path)
    print(f"\n{'=' * 70}")
    print("  Done! Check nested runs in the MLflow UI for full details.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
