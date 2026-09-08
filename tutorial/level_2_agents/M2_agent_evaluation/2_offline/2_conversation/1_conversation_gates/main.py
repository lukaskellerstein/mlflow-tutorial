"""L2-M2.2.2.1 -- Offline Gates on Whole Conversations.

1_turn/2_offline_gates answers "is this version good enough to ship?" one
question at a time. This lesson asks the same question of a whole conversation.

Why that is not the same gate. A support agent can answer every single turn
correctly and still fail the only thing the business cares about: the customer
did not get their problem solved and phoned in anyway. "Was the task ever
finished?" is a property of the session, so a turn gate is structurally unable
to see it -- it can be all green while the product is failing.

The pipeline is the same shape as the turn gate. Only the UNIT changes:

    turn gate         case      -> agent -> turn scorers    -> threshold
    conversation gate scenario  -> agent -> SESSION scorers -> threshold

One thing genuinely changes, and it is the hard part: a simulated conversation
is driven by a second model, so the SAME candidate scores differently on
consecutive runs. A conversation gate has to be built for that. See Step 4.
"""

from __future__ import annotations

import os
import statistics
import uuid
from typing import Any, cast

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Trace
from mlflow.genai.scorers import ConversationCompleteness, KnowledgeRetention, UserFrustration
from pydantic import SecretStr

GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"
JUDGE_ALIAS = "gemma-judge"

os.environ["OPENAI_API_KEY"] = GATEWAY_KEY
os.environ["OPENAI_BASE_URL"] = GATEWAY_URL
SESSION_MODEL = f"openai:/{JUDGE_ALIAS}"

EXPERIMENT = "L2/M2_agent_evaluation/2_offline/2_conversation/1_conversation_gates"
BASELINE_TAG = "conversation_gate_baseline"

mlflow.set_tracking_uri("http://127.0.0.1:5555")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT).experiment_id
mlflow.langchain.autolog()

ORDERS = {
    "A1001": "shipped, arriving Thursday",
    "A1002": "held at the warehouse, payment not confirmed",
    "A1003": "delivered on Monday",
}
POLICIES = {
    "returns": "30 days from delivery, receipt required (P-101).",
    "warranty": "12 months against manufacturing defects (P-204).",
    "shipping": "Return shipping is free for faulty items (P-330).",
}


@tool
def order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A1001."""
    return ORDERS.get(order_id.strip().upper(), f"No order found with id {order_id}.")


@tool
def policy_lookup(topic: str) -> str:
    """Look up store policy. Topics: returns, warranty, shipping."""
    return POLICIES.get(topic.strip().lower(), f"No policy on file for '{topic}'.")


SYSTEM_PROMPT = (
    "You are a retail support agent. Use the tools to answer questions about orders and policy. "
    "Never invent an order status or a policy -- look it up. Keep replies under three sentences."
)

# The curated conversation set. This is the offline half of "offline": input you
# own, chosen deliberately, run when YOU decide -- not sampled from live traffic.
# Each entry is a scripted user side, so the gate itself stays reproducible.
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "delayed_order",
        "turns": [
            "What is the status of order A1002?",
            "So when will it actually ship?",
            "Fine. And remind me which order we were discussing?",
        ],
    },
    {
        "name": "faulty_return",
        "turns": [
            "I bought something two months ago and it is faulty.",
            "How long do I have to return it?",
            "And who pays the return shipping?",
        ],
    },
]

# The gate. `op` carries the POLARITY, which is the trap: user_frustration is
# good at 0.0 while the other two are good at 1.0. A gate that compared every
# metric with >= would pass the most frustrated agent it could build.
GATES = [
    ("conversation_completeness", "min", 0.5),
    ("knowledge_retention", "min", 0.5),
    ("user_frustration", "max", 0.5),
]


def build(scorer_cls: Any, name: str) -> Any:
    """Construct a built-in session scorer bound to the gateway model.

    `model` is a real pydantic field on these classes, but basedpyright resolves
    a narrower __init__ than pydantic synthesises and reports "No parameter named
    model". The factory keeps the checker useful without inline suppressions.
    """
    return scorer_cls(name=name, model=SESSION_MODEL)


SESSION_SCORERS = [
    build(ConversationCompleteness, "conversation_completeness"),
    build(KnowledgeRetention, "knowledge_retention"),
    build(UserFrustration, "user_frustration"),
]

TRUTHY = {"yes", "true", "pass", "y"}


def build_agent(system_prompt: str = SYSTEM_PROMPT) -> Any:
    """The candidate under test. `system_prompt` is the knob we break in Step 5."""
    llm = ChatOpenAI(base_url=GATEWAY_URL, api_key=SecretStr(GATEWAY_KEY), model=MODEL_ALIAS, temperature=0.0)
    return create_agent(llm, tools=[order_status, policy_lookup], system_prompt=system_prompt)


@mlflow.trace(name="conversation_turn")
def run_turn(agent: Any, history: list[Any], session_id: str) -> dict:
    """One turn, wrapped so we own the root span and can stamp the session id."""
    mlflow.update_current_trace(session_id=session_id)
    return agent.invoke({"messages": history})


def run_scenario(agent: Any, scenario: dict[str, Any]) -> list[Trace]:
    """Drive one scripted conversation and return its traces, in order."""
    session_id = f"gate-{scenario['name']}-{uuid.uuid4().hex[:8]}"
    trace_ids: list[str] = []
    history: list[Any] = []
    for turn in scenario["turns"]:
        history.append({"role": "user", "content": turn})
        result = run_turn(agent, history, session_id)
        history = result["messages"]
        if trace_id := mlflow.get_last_active_trace_id():
            trace_ids.append(trace_id)
    mlflow.flush_trace_async_logging()
    return [t for tid in trace_ids if (t := mlflow.get_trace(tid))]


def to_metric(value: Any) -> float:
    """Session judges answer with STRINGS. bool('no') is True, so never cast."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in TRUTHY else 0.0
    return 0.0


def score_conversations(agent: Any, label: str) -> dict[str, float]:
    """Run every scenario, score each session, and average per metric."""
    print(f"\n  [{label}] simulating {len(SCENARIOS)} conversations...")
    per_metric: dict[str, list[float]] = {s.name: [] for s in SESSION_SCORERS}
    for scenario in SCENARIOS:
        traces = run_scenario(agent, scenario)
        print(f"    {scenario['name']:<16} {len(traces)} turns", end="")
        for session_scorer in SESSION_SCORERS:
            try:
                feedback = session_scorer(session=traces)
            except Exception:
                # A judge that could not parse its own answer contributes
                # NOTHING. Recording it as 0.0 would fail the gate on a judge
                # bug rather than on agent quality.
                print(f"  {session_scorer.name}=ERR", end="")
                continue
            value = to_metric(getattr(feedback, "value", feedback))
            per_metric[session_scorer.name].append(value)
            print(f"  {session_scorer.name.split('_')[0]}={value:.0f}", end="")
        print()
    return {k: (statistics.mean(v) if v else float("nan")) for k, v in per_metric.items()}


def apply_gates(scores: dict[str, float]) -> tuple[bool, list[str]]:
    """Check every gate. Returns (shipped, per-gate report lines)."""
    lines, shipped = [], True
    for metric, op, threshold in GATES:
        value = scores.get(metric, float("nan"))
        if value != value:  # NaN -- every judge for this metric failed
            lines.append(f"  {metric:<28} {'NO DATA':>8}   gate SKIPPED (all judges failed)")
            continue
        ok = value >= threshold if op == "min" else value <= threshold
        shipped = shipped and ok
        sign = ">=" if op == "min" else "<="
        lines.append(f"  {metric:<28} {value:>8.2f}   {sign} {threshold}   {'PASS' if ok else 'FAIL'}")
    return shipped, lines


def load_baseline() -> dict[str, float] | None:
    """The last shipped scores, stored as a run tag rather than a file on disk."""
    # `search_runs` is typed `list[Run] | DataFrame`, and truth-testing a
    # DataFrame raises. We asked for a list, so say so -- do not write
    # `if not runs` against the union.
    runs: list[Any] = cast(
        "list[Any]",
        mlflow.search_runs(
            experiment_ids=[EXPERIMENT_ID],
            filter_string=f"tags.{BASELINE_TAG} = 'true'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        ),
    )
    if not runs:
        return None
    return {m: v for m, v in runs[0].data.metrics.items() if m in {g[0] for g in GATES}}


def main() -> None:
    print("=" * 70)
    print("  L2-M2.2.2.1 -- Offline Gates on Whole Conversations")
    print("=" * 70)

    # -- 1-3. Candidate: simulate, score, gate ------------------------------- #
    print("\nStep 1-3: run the candidate through the curated conversations")
    agent = build_agent()
    with mlflow.start_run(run_name="candidate") as run:
        scores = score_conversations(agent, "candidate")
        mlflow.log_params({"model": MODEL_ALIAS, "judge": JUDGE_ALIAS, "scenarios": len(SCENARIOS)})
        mlflow.log_metrics({k: v for k, v in scores.items() if v == v})

        shipped, lines = apply_gates(scores)
        print(f"\n{'=' * 70}\n  Gate report\n{'=' * 70}")
        print(f"  {'metric':<28} {'value':>8}   rule          verdict")
        print(f"  {'-' * 28} {'-' * 8}   {'-' * 12}  {'-' * 7}")
        for line in lines:
            print(line)
        print(f"\n  VERDICT: {'SHIP' if shipped else 'BLOCK'}")
        mlflow.set_tag("gate_verdict", "SHIP" if shipped else "BLOCK")

        # -- 4. Compare against the last shipped version --------------------- #
        print(f"\n{'=' * 70}\n  Step 4: regression against the stored baseline\n{'=' * 70}")
        baseline = load_baseline()
        if baseline is None:
            print("  no baseline yet -- freezing this run as the baseline")
            mlflow.set_tag(BASELINE_TAG, "true")
        else:
            print(f"  {'metric':<28} {'baseline':>9} {'now':>8} {'delta':>8}")
            print(f"  {'-' * 28} {'-' * 9} {'-' * 8} {'-' * 8}")
            for metric, _op, _t in GATES:
                was, now = baseline.get(metric), scores.get(metric)
                if was is None or now is None or now != now:
                    continue
                print(f"  {metric:<28} {was:>9.2f} {now:>8.2f} {now - was:>+8.2f}")
            print("\n  A delta here is NOT automatically a regression. The simulated")
            print("  user is a model, so the same candidate scores differently on")
            print("  consecutive runs. Read Step 5 before acting on a small drop.")

        print(f"\n  run: {run.info.run_id}")

    # -- 5. Why a single run is not a gate ----------------------------------- #
    print(f"\n{'=' * 70}\n  Step 5: the thing that makes a conversation gate hard\n{'=' * 70}")
    print("  A turn gate replays a FIXED dataset, so two runs of the same")
    print("  candidate give the same score. A conversation gate does not: the")
    print("  user side is generated, and every judge is a model too.")
    print("\n  Three ways to build a gate you can trust on top of that:")
    print("    1. threshold on a MEAN over many conversations, not on one")
    print("    2. run the suite k times and gate on the WORST run (pass^k),")
    print("       the same trick tau-bench uses -- see 2_multi_turn_benchmark")
    print("    3. give the gate a tolerance band, and only investigate a drop")
    print("       that exceeds the run-to-run spread you measured")
    print("\n  This lesson does (1). Doing (2) properly is the next lesson.")
    print(f"\n  MLflow UI: http://localhost:5555 -> experiment '{EXPERIMENT}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
