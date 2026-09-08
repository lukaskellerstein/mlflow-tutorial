"""
L2-M1.1.2 — DeepAgents with MLflow

DeepAgents is LangChain-AI's agent harness: `create_deep_agent()` wraps
`create_agent()` and hands the model a built-in toolkit — planning
(`write_todos`), a virtual filesystem (`ls`/`read_file`/`write_file`/`edit_file`/
`glob`/`grep`) and sub-agent delegation (`task`).

It is LangGraph underneath, so `mlflow.langchain.autolog()` traces all of it with
no custom integration.

This is a TURN lesson. Every task below starts from an empty message list, and
nothing is carried from one task to the next. Memory that survives a turn
boundary is `2_conversation/2_deepagents`.

Parts:
  1. One deep agent, three tasks — built-in toolkit, custom tools, sub-agents
  2. Backends — where a file lives once the turn ends, and how MLflow logs it
  3. Trace analysis — the span tree, and what a sub-agent hid from its caller
"""

import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.langchain
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.entities import Span, Trace
from pydantic import SecretStr

# The MLflow AI Gateway -- the tracking server itself. Which model "gemma-agent" resolves to lives
# in infra/mlflow/gateway/seed_gateway.py. It is a local Unsloth model with no fallback, so this
# lesson fails loudly rather than answering from somewhere else.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/1_turn/2_deepagents"

# deepagents defaults to recursion_limit=9999, sized for frontier models. Capped
# so a confused run fails fast instead of looping for an hour.
RUN_CONFIG: RunnableConfig = {"recursion_limit": 50}

WORKSPACE = Path(__file__).parent / "workspace"


BUILT_IN_TOOLS = "write_todos, ls, read_file, write_file, edit_file, glob, grep, task"

# ── Custom tools — ADDED to the built-in suite, never replacing it ─────────


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for facts about a topic."""
    knowledge = {
        "microservices": (
            "Microservices architecture decomposes applications into small, independent "
            "services that communicate over APIs. Benefits: independent deployment, "
            "technology flexibility, fault isolation, team autonomy. Challenges: "
            "distributed complexity, data consistency, operational overhead."
        ),
        "monolith": (
            "Monolithic architecture bundles all application logic into a single "
            "deployable unit. Benefits: simpler development, easier debugging, "
            "single deployment. Challenges: scaling limits, tight coupling, "
            "slower release cycles as the codebase grows."
        ),
        "event-driven": (
            "Event-driven architecture uses events to trigger communication between "
            "decoupled services. Benefits: loose coupling, scalability, real-time "
            "processing. Challenges: eventual consistency, debugging complexity, "
            "event ordering."
        ),
    }
    for key, text in knowledge.items():
        if key in query.lower():
            return text
    return "No results found for that query."


@tool
def get_industry_stats(topic: str) -> str:
    """Get industry statistics and adoption data for a technology topic."""
    stats = {
        "microservices": (
            "Adoption: 85% of enterprises use microservices (2024 survey). "
            "Average team size per service: 5-8 engineers. "
            "Deployment frequency: 10-100x more frequent than monoliths. "
            "Incident rate: 23% higher initially, 40% lower after 18 months."
        ),
        "monolith": (
            "Still used by: 60% of startups for initial launch. "
            "Migration rate: 35% of monoliths begin microservices migration within 3 years. "
            "Average codebase size at migration trigger: 500K+ lines."
        ),
    }
    for key, text in stats.items():
        if key in topic.lower():
            return text
    return f"No statistics available for '{topic}'."


CUSTOM_TOOLS = [search_knowledge_base, get_industry_stats]


# ── Sub-agents — each one gets its own context window ─────────────────────

RESEARCHER: SubAgent = {
    "name": "researcher",
    "description": "Researches ONE topic using the knowledge base and statistics tools.",
    "system_prompt": (
        "You are a research specialist. Use search_knowledge_base and "
        "get_industry_stats to gather information about the topic you are given. "
        "Reply with structured bullet points. Your reply goes to another agent, "
        "not a human."
    ),
    "tools": CUSTOM_TOOLS,
}

ANALYST: SubAgent = {
    "name": "analyst",
    "description": "Analyzes research findings and produces a summary with recommendations.",
    "system_prompt": (
        "You are an analysis specialist. Given research findings, identify the "
        "most important patterns, trade-offs and recommendations. Structure your "
        "analysis as: Key Findings, Trade-offs, Recommendations. Reply with the "
        "analysis only."
    ),
}


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Chat model pointed at the MLflow AI Gateway."""
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=temperature,
    )


# ── Shared helpers ────────────────────────────────────────────────────────


def count_tool_calls(messages: list[Any]) -> dict[str, int]:
    """Count tool calls by name from the message history."""
    counts: dict[str, int] = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            counts[tc["name"]] = counts.get(tc["name"], 0) + 1
    return counts


def print_conversation(messages: list[Any]) -> None:
    """Print a condensed view of the agent's conversation."""
    for msg in messages:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc["name"] == "task":
                    print(f"    [{kind}] -> task -> sub-agent '{tc['args'].get('subagent_type', '?')}'")
                else:
                    print(f"    [{kind}] -> {tc['name']}({str(tc['args'])[:70]})")
        elif isinstance(msg, ToolMessage):
            print(f"    [ToolResult] {str(msg.content)[:100]}")
        elif getattr(msg, "content", None):
            print(f"    [{kind}] {str(msg.content)[:120]}")


def state_files(state: dict[str, Any]) -> dict[str, str]:
    """The virtual filesystem a StateBackend leaves in the returned state."""
    return {path: data.get("content", "") for path, data in (state.get("files") or {}).items()}


def disk_files() -> list[Path]:
    """The real files a FilesystemBackend wrote into ./workspace."""
    return [p for p in sorted(WORKSPACE.rglob("*")) if p.is_file()]


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Print a fixed-width table, so the console shows what MLflow logged."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [max([len(h)] + [len(row[i]) for row in cells]) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in cells:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))


# ── Part 1: one agent, three tasks ────────────────────────────────────────

TASKS = [
    (
        "plan_and_research",
        "Use write_todos to plan your work, then research microservices architecture "
        "with search_knowledge_base and get_industry_stats. Save a short summary to "
        "/research.md.",
    ),
    (
        "delegate",
        "Use the task tool to delegate. Ask the 'researcher' sub-agent for facts about "
        "monolith architecture, then give those facts to the 'analyst' sub-agent for a "
        "recommendation. Save the analysis to /analysis.md. Do not research yourself.",
    ),
    (
        "write_then_grep",
        "Look up event-driven architecture with search_knowledge_base and write what you "
        "find to /notes.md. Then grep /notes.md for 'coupling' and report the line that "
        "matched.",
    ),
]


def part1_toolkit_and_subagents() -> list[tuple[str, str]]:
    """ONE deep agent: custom tools, the built-in toolkit and sub-agents together.

    Returns (label, trace_id) per task, for the trace analysis in Part 3.
    """
    print("\n" + "=" * 60)
    print("Part 1: One deep agent — toolkit, custom tools and sub-agents")
    print("=" * 60)

    # tools= and subagents= on the SAME agent. The model decides, per task, whether
    # to use a tool itself or hand the work to a sub-agent.
    agent = create_deep_agent(
        model=get_llm(),
        tools=CUSTOM_TOOLS,
        subagents=[RESEARCHER, ANALYST],
        system_prompt=(
            "You are a technology research assistant. You have your own tools and two "
            "sub-agents ('researcher', 'analyst') reachable with the task tool. Plan "
            "with write_todos, then do exactly what the task asks for. Save results "
            "with write_file."
        ),
    )

    rows: list[list[Any]] = []
    traced: list[tuple[str, str]] = []

    with mlflow.start_run(run_name="deep_agent"):
        mlflow.set_tags({"framework": "deepagents", "model_alias": MODEL_ALIAS, "gateway": "mlflow-ai-gateway"})
        mlflow.log_params(
            {
                "agent_type": "deep_agent",
                "model_alias": MODEL_ALIAS,
                "custom_tools": ", ".join(t.name for t in CUSTOM_TOOLS),
                "built_in_tools": BUILT_IN_TOOLS,
                "subagents": "researcher, analyst",
                "backend": "StateBackend (default)",
            }
        )

        for idx, (name, task) in enumerate(TASKS, start=1):
            print(f"\n  Task {idx} ({name}): {task[:80]}...")

            start = time.time()
            # An empty message list every time: this is a turn, not a conversation.
            result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config=RUN_CONFIG)
            duration = round(time.time() - start, 2)

            # autolog opened a trace for that invoke. Keep its id: Part 3 reads
            # the spans back and asks what the run actually cost.
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                traced.append((name, trace_id))

            tool_calls = count_tool_calls(result["messages"])
            steps = len(result["messages"])
            handoffs = tool_calls.get("task", 0)
            files = state_files(result)

            with mlflow.start_run(run_name=f"task_{idx}_{name}", nested=True):
                mlflow.log_params({"task": task, "task_index": idx})
                mlflow.log_metrics(
                    {
                        "duration_s": duration,
                        "total_steps": steps,
                        "total_tool_calls": sum(tool_calls.values()),
                        "subagent_handoffs": handoffs,
                        "files_written": len(files),
                    }
                )
                for tool_name, count in tool_calls.items():
                    mlflow.log_metric(f"tool_{tool_name}", count)

            print("\n    Conversation:")
            print_conversation(result["messages"])

            todos = result.get("todos", [])
            if todos:
                print("\n    Todos (write_todos -> agent state):")
                for todo in todos:
                    print(f"      [{todo['status']}] {todo['content']}")
            else:
                # Worth seeing rather than hiding: `write_todos` is offered but not
                # forced, and gemma-4 often plans into a plain file instead. The
                # planning HAPPENED — it just landed in the filesystem below.
                print("\n    Todos: (empty — this model planned into a file instead)")

            print("\n    Files (StateBackend — they live in the returned state, not on disk):")
            for path, content in files.items():
                print(f"      {path} ({len(content)} chars)")
            if not files:
                print("      (none)")

            print(f"\n    {duration}s | steps {steps} | tools {sum(tool_calls.values())} | handoffs {handoffs}")

            rows.append([idx, name, steps, sum(tool_calls.values()), handoffs, len(files), duration])

        table: dict[str, list[Any]] = {
            "task_index": [r[0] for r in rows],
            "task": [r[1] for r in rows],
            "total_steps": [r[2] for r in rows],
            "tool_calls": [r[3] for r in rows],
            "subagent_handoffs": [r[4] for r in rows],
            "files_written": [r[5] for r in rows],
            "duration_s": [r[6] for r in rows],
        }
        mlflow.log_table(data=table, artifact_file="tasks.json")
        mlflow.log_metrics(
            {
                "avg_duration_s": round(sum(r[6] for r in rows) / len(rows), 2),
                "total_tool_calls": sum(r[3] for r in rows),
                "total_subagent_handoffs": sum(r[4] for r in rows),
            }
        )

    print("\n  Summary — one agent, three independent turns:")
    print_table(["task", "name", "steps", "tools", "handoffs", "files", "duration"], rows)

    return traced


# ── Part 2: backends ──────────────────────────────────────────────────────

BACKEND_TASK = (
    "Research microservices architecture with search_knowledge_base and save a short summary to /research.md."
)


def part2_backends() -> list[tuple[str, str]]:
    """Where a "file" lives once the turn ends — and what MLflow can log.

    Same task, same tool call, ONE invoke each. Both agents behave identically.
    The difference only appears after the turn is over.

    Returns (label, trace_id) per backend, for the trace analysis in Part 3.
    """
    print("\n" + "=" * 60)
    print("Part 2: Backends — what write_file actually wrote")
    print("=" * 60)

    # ./workspace is gitignored, so files left by an earlier run would make the
    # disk column lie. Start from nothing.
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    backends = [
        ("StateBackend", StateBackend()),
        ("FilesystemBackend", FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True)),
    ]

    rows: list[list[Any]] = []
    traced: list[tuple[str, str]] = []

    for label, backend in backends:
        agent = create_deep_agent(
            model=get_llm(),
            tools=[search_knowledge_base],
            backend=backend,
            system_prompt="You are a research assistant. Use your tools, then save the result with write_file.",
        )

        with mlflow.start_run(run_name=f"backend_{label}"):
            mlflow.log_params({"backend": label, "model_alias": MODEL_ALIAS, "root_dir": str(WORKSPACE)})

            start = time.time()
            result = agent.invoke({"messages": [{"role": "user", "content": BACKEND_TASK}]}, config=RUN_CONFIG)
            duration = round(time.time() - start, 2)

            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                traced.append((label, trace_id))

            in_state = state_files(result)
            on_disk = disk_files()

            # The whole point, in four lines. A real path can be logged as an
            # artifact. A file that exists only in the state dict cannot — you have
            # to pull the text out and log that instead.
            for path, content in in_state.items():
                mlflow.log_text(content, f"from_state{path}")
            for path in on_disk:
                mlflow.log_artifact(str(path))

            how = "mlflow.log_artifact(path)" if on_disk else "mlflow.log_text(state)"
            mlflow.log_metrics({"duration_s": duration, "files_in_state": len(in_state), "files_on_disk": len(on_disk)})
            mlflow.set_tag("logged_with", how)

            print(f"\n  {label}: {duration}s")
            print(f"    files in the returned state : {list(in_state) or '(none)'}")
            print(f"    files in ./workspace        : {[p.name for p in on_disk] or '(none)'}")

            rows.append([label, len(in_state), len(on_disk), how])

    table: dict[str, list[Any]] = {
        "backend": [r[0] for r in rows],
        "files_in_state": [r[1] for r in rows],
        "files_on_disk": [r[2] for r in rows],
        "mlflow_logs_it_with": [r[3] for r in rows],
    }
    with mlflow.start_run(run_name="backend_comparison"):
        mlflow.set_tag("run_type", "comparison")
        mlflow.log_table(data=table, artifact_file="backends.json")

    print("\n  After ONE turn — the agent did the same thing, the file did not:")
    print_table(["backend", "in state", "on disk", "MLflow logs it with"], rows)
    print("\n  StoreBackend and CompositeBackend carry files ACROSS turns.")
    print("  They belong to a conversation, so they live in 2_conversation/2_deepagents.")

    return traced


# ── Part 3: trace analysis ────────────────────────────────────────────────


def span_depth(span: Span, by_id: dict[str, Span]) -> int:
    """How deep this span sits under the root."""
    depth = 0
    while span.parent_id and span.parent_id in by_id:
        span = by_id[span.parent_id]
        depth += 1
    return depth


def inside_subagent(span: Span, by_id: dict[str, Span]) -> bool:
    """True if any ancestor of this span is a `task` tool call.

    Those spans are the work a sub-agent did. The caller never saw them — only
    the sub-agent's final answer came back. That is context isolation, and this
    is where you can measure it.
    """
    while span.parent_id and span.parent_id in by_id:
        span = by_id[span.parent_id]
        if span.name == "task":
            return True
    return False


def print_span_tree(trace: Trace) -> None:
    """Indent every span under its parent, so delegation is visible as nesting."""
    by_id = {s.span_id: s for s in trace.data.spans}
    for span in trace.data.spans:
        marker = "  " * span_depth(span, by_id)
        duration_ms = ((span.end_time_ns or 0) - (span.start_time_ns or 0)) / 1e6
        print(f"      {marker}[{span.span_type}] {span.name} ({duration_ms:.0f}ms)")


def part3_trace_analysis(traced: list[tuple[str, str]]) -> None:
    """Read back what autolog captured for every turn above.

    No agent runs here. Everything comes from traces the parts above produced.
    """
    print("\n" + "=" * 60)
    print("Part 3: Trace analysis — spans, nesting and the token bill")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  No experiment found — skipping trace analysis.")
        return

    # flush=True waits for the async exporter, so the turns just run are there.
    found = {
        trace.info.trace_id: trace
        for trace in cast(
            list[Trace],
            mlflow.search_traces(locations=[experiment.experiment_id], max_results=50, return_type="list", flush=True),
        )
    }

    rows: list[list[Any]] = []
    deepest: tuple[int, str, Trace] | None = None

    for label, trace_id in traced:
        trace = found.get(trace_id)
        if trace is None:
            print(f"  {label}: trace {trace_id} is not exported yet — skipped")
            continue

        by_id = {s.span_id: s for s in trace.data.spans}
        hidden = sum(1 for s in trace.data.spans if inside_subagent(s, by_id))
        usage = trace.info.token_usage or {}
        rows.append(
            [
                label,
                len(trace.data.spans),
                hidden,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                trace.info.execution_time_ms or 0,
            ]
        )
        if hidden and (deepest is None or hidden > deepest[0]):
            deepest = (hidden, label, trace)

    if not rows:
        print("  No traces to analyse.")
        return

    print("\n  One row per turn. 'hidden' spans ran inside a sub-agent:")
    print_table(["turn", "spans", "hidden", "input_tok", "output_tok", "ms"], rows)

    if deepest:
        hidden, label, trace = deepest
        print(f"\n  Span tree of '{label}' — {hidden} of {len(trace.data.spans)} spans ran inside a sub-agent:")
        print_span_tree(trace)
        print("\n  Everything indented under [TOOL] task cost tokens and time, but never")
        print("  entered the caller's message list. It got one ToolMessage back.")

    table: dict[str, list[Any]] = {
        "turn": [r[0] for r in rows],
        "spans": [r[1] for r in rows],
        "spans_inside_subagents": [r[2] for r in rows],
        "input_tokens": [r[3] for r in rows],
        "output_tokens": [r[4] for r in rows],
        "execution_time_ms": [r[5] for r in rows],
    }
    with mlflow.start_run(run_name="trace_analysis"):
        mlflow.set_tag("run_type", "analysis")
        mlflow.log_table(data=table, artifact_file="traces.json")
        mlflow.log_metrics(
            {
                "traces_analysed": len(rows),
                "total_spans": sum(r[1] for r in rows),
                "spans_inside_subagents": sum(r[2] for r in rows),
                "total_input_tokens": sum(r[3] for r in rows),
                "total_output_tokens": sum(r[4] for r in rows),
            }
        )


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("L2-M1.1.2 — DeepAgents with MLflow")
    print("=" * 60)
    print("\nDeepAgents is built on LangGraph, so mlflow.langchain.autolog()")
    print("captures every agent step, tool call and sub-agent trace.\n")

    mlflow.langchain.autolog()

    traced = part1_toolkit_and_subagents()
    traced += part2_backends()
    part3_trace_analysis(traced)

    print("\n" + "=" * 60)
    print("Done. View traces in the MLflow UI:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print("  Open the 'delegate' trace: the sub-agent's work nests under [TOOL] task.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT)
    main()
