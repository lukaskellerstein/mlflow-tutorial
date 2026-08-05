"""
L2-M3.2 — Agent Configuration Optimization

M3.1 used `mlflow.genai.optimize_prompts()`, a real optimizer with a real
algorithm. This lesson covers everything MLflow has NO optimizer for:

  - model selection      -- the highest-leverage knob in practice
  - tool / MCP budget    -- which tools to expose at all
  - delegation topology  -- subagents, skills, handoffs

There is exactly one pattern for all of them, and MLflow's role in it is to
**track the search, not run it**: a parent run per sweep, a child run per
configuration, every child scored by the same judge. That is the transferable
part -- it works for any knob anyone invents later.

  Part 1: define the search space
  Part 2: sweep it, one nested run per configuration
  Part 3: read a Pareto frontier rather than a single winner
  Part 4: know when to stop -- variance vs. the size of the improvement

Builds on L2-M2.1.3 (Quality Metrics) and L2-M3.1 (Prompt Optimization).
"""

import itertools
import statistics
import time

import mlflow
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# The LiteLLM gateway from infra/. Swapping a model is a change to the alias
# list below, never to provider code -- which is what makes a model sweep cheap
# to express. See L2-M1.1.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin

EXPERIMENT = "L2/M3_agent_optimization/2_configuration_optimization"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment(EXPERIMENT)
mlflow.langchain.autolog(log_traces=True)


# ---------------------------------------------------------------------------
# Tools -- the pool the "tool budget" axis selects from
# ---------------------------------------------------------------------------
KNOWLEDGE = {
    "mlflow": "MLflow is an open-source platform for the ML lifecycle: tracking, "
    "model registry, evaluation, and deployment.",
    "langgraph": "LangGraph builds stateful multi-actor LLM applications using "
    "graph-based workflows with nodes, edges, and state.",
    "docker": "Docker packages applications and dependencies into portable containers.",
}


@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information on a topic."""
    q = query.lower()
    hits = [v for k, v in KNOWLEDGE.items() if k in q]
    return hits[0] if hits else f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3' or '10 * 5'."""
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        return str(eval(expression))  # nosec: whitelisted arithmetic characters only
    return "Invalid expression — only basic arithmetic is supported."


@tool
def get_current_year() -> str:
    """Return the current year."""
    return "2026"


@tool
def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))


ALL_TOOLS = [search_knowledge, calculate, get_current_year, word_count]
MINIMAL_TOOLS = [search_knowledge, calculate]


# ---------------------------------------------------------------------------
# Part 1: the search space
# ---------------------------------------------------------------------------
# Free-tier aliases keep the sweep cheap. `gemma-small` is deliberately absent:
# it is served by LMStudio, and a sweep that dies when LMStudio is asleep teaches
# the wrong lesson about reproducibility.
MODELS = ["gemma-26b-free", "gemma-31b-free"]
TOOL_BUDGETS = {"minimal": MINIMAL_TOOLS, "full": ALL_TOOLS}

EVAL_CASES = [
    {"input": "What is MLflow used for?", "expect": "lifecycle"},
    {"input": "What is 25 * 4?", "expect": "100"},
    {"input": "Explain LangGraph in one sentence.", "expect": "stateful"},
    {"input": "What is 144 / 12?", "expect": "12"},
    {"input": "What is Docker?", "expect": "container"},
]


def build_agent(model_alias: str, tools: list):
    llm = ChatOpenAI(
        model=model_alias,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )
    return create_agent(llm, tools)


def score_config(model_alias: str, budget_name: str, tools: list) -> dict:
    """Run every eval case against one configuration and aggregate."""
    agent = build_agent(model_alias, tools)
    correct, latencies, tool_calls, errors = 0, [], 0, 0

    for case in EVAL_CASES:
        started = time.time()
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})
            latencies.append(time.time() - started)
            answer = response["messages"][-1].content or ""
            # A deterministic scorer keeps the sweep fast and free. Swap in a
            # registered judge from M2.1.2 when the answers are open-ended.
            if case["expect"].lower() in answer.lower():
                correct += 1
            tool_calls += sum(1 for m in response["messages"] if getattr(m, "name", None))
        except Exception as exc:  # noqa: BLE001 -- a failing config is data, not a crash
            errors += 1
            latencies.append(time.time() - started)
            print(f"      ! {model_alias}/{budget_name} failed on {case['input']!r}: {exc}")

    return {
        "accuracy": correct / len(EVAL_CASES),
        "avg_latency_s": round(statistics.mean(latencies), 2),
        "tool_calls": tool_calls,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Part 2: sweep the space, one nested run per configuration
# ---------------------------------------------------------------------------
def run_sweep() -> list[dict]:
    print("\n" + "=" * 60)
    print("  Part 2: Sweep the configuration space")
    print("=" * 60)

    space = list(itertools.product(MODELS, TOOL_BUDGETS.items()))
    print(f"  {len(MODELS)} models x {len(TOOL_BUDGETS)} tool budgets = {len(space)} configurations\n")

    results = []
    with mlflow.start_run(run_name="configuration_sweep"):
        mlflow.log_params(
            {
                "models": ",".join(MODELS),
                "tool_budgets": ",".join(TOOL_BUDGETS),
                "eval_cases": len(EVAL_CASES),
            }
        )

        for model_alias, (budget_name, tools) in space:
            label = f"{model_alias}/{budget_name}"
            print(f"    [{label}] running {len(EVAL_CASES)} cases...")

            # One child run per configuration IS the search log. This is the
            # whole MLflow contribution here -- it does not choose the next
            # configuration, it makes the comparison auditable afterwards.
            with mlflow.start_run(run_name=label, nested=True):
                mlflow.log_params(
                    {
                        "model": model_alias,
                        "tool_budget": budget_name,
                        "n_tools": len(tools),
                    }
                )
                metrics = score_config(model_alias, budget_name, tools)
                mlflow.log_metrics(metrics)

            print(
                f"      accuracy={metrics['accuracy']:.0%} "
                f"latency={metrics['avg_latency_s']}s "
                f"tool_calls={metrics['tool_calls']} errors={metrics['errors']}"
            )
            results.append({"config": label, "n_tools": len(tools), **metrics})

        best = max(results, key=lambda r: r["accuracy"])
        mlflow.log_metrics({"best_accuracy": best["accuracy"], "configs_tried": len(results)})
        mlflow.set_tag("best_config", best["config"])

    return results


# ---------------------------------------------------------------------------
# Part 3: the Pareto frontier
# ---------------------------------------------------------------------------
def pareto_frontier(results: list[dict]) -> list[dict]:
    """A config is on the frontier if nothing beats it on BOTH axes.

    Reporting one winner hides the tradeoff. The frontier is the honest answer:
    these are the configurations worth their cost, and picking between them is a
    business decision, not a measurement.
    """
    frontier = []
    for candidate in results:
        dominated = any(
            other["accuracy"] >= candidate["accuracy"]
            and other["avg_latency_s"] <= candidate["avg_latency_s"]
            and other != candidate
            and (other["accuracy"] > candidate["accuracy"] or other["avg_latency_s"] < candidate["avg_latency_s"])
            for other in results
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def report(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("  Part 3: Pareto frontier (quality vs. latency)")
    print("=" * 60)

    print(f"\n  {'configuration':<28}{'accuracy':>10}{'latency':>10}{'tools':>8}")
    print("  " + "-" * 56)
    for r in sorted(results, key=lambda r: -r["accuracy"]):
        print(f"  {r['config']:<28}{r['accuracy']:>9.0%}{r['avg_latency_s']:>9.1f}s{r['n_tools']:>8}")

    frontier = pareto_frontier(results)
    print(f"\n  On the frontier ({len(frontier)} of {len(results)}):")
    for r in frontier:
        print(f"    {r['config']}  --  {r['accuracy']:.0%} at {r['avg_latency_s']}s")

    print("\n  Part 4: when to stop")
    accs = [r["accuracy"] for r in results]
    spread = max(accs) - min(accs)
    per_case = 1 / len(EVAL_CASES)
    print(f"    accuracy spread across configs : {spread:.0%}")
    print(f"    one test case is worth          : {per_case:.0%}")
    if spread <= per_case:
        print("    -> The spread is within a single test case. This sweep has NOT")
        print("       found a real difference; add cases before trusting a winner.")
    else:
        print("    -> The spread exceeds one test case, so the ranking means something.")
        print("       Confirm it by repeating the best two configs and comparing variance.")


def main() -> None:
    print("=" * 60)
    print("  L2-M3.2 — Agent Configuration Optimization")
    print("=" * 60)
    print("\n  Part 1: the search space")
    print(f"    models       : {MODELS}")
    print(f"    tool budgets : {list(TOOL_BUDGETS)}")
    print("\n  MLflow has no optimizer for any of these. It tracks the search;")
    print("  you run it. That is the entire point of this lesson.")

    results = run_sweep()
    report(results)

    print("\n" + "=" * 60)
    print("  Done! View the sweep in MLflow UI: http://127.0.0.1:5555")
    print(f"  Experiment: {EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
