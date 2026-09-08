"""L2-M1.2.2 — everything the two approaches hold fixed.

`main_decorator.py` and `main_context.py` both import this module. They do the
same job and differ in ONE thing: how a turn gets stamped with the session id.
The deep agent, the four turns, the run-level logging, the control, the
FilesystemBackend trap, the StoreBackend share and the read-back all live here.

That is not tidiness. If each script carried its own copy, the two could drift,
and a difference in the MLflow UI would stop being evidence about the stamping
API — it could just be two files doing different work. Sharing this module is
what makes the comparison mean something.

L2-M1.2.1 made the same split for a plain LangChain agent. Repeating it here is
the point: the choice between the two stamping APIs belongs to MLflow, not to
the framework, so it lands the same way on a deep agent that also carries a
todo list, a filesystem and two sub-agents.

**The agent is the one from L2-M1.1.2**, unchanged: the same two custom tools,
the same `researcher` and `analyst` sub-agents, the same system prompt, plus a
checkpointer. That is deliberate. The turn lesson showed what this agent can do
in one shot; this lesson asks what survives when the shot ends, and holding the
agent fixed is what makes the two lessons comparable.
"""

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import mlflow.langchain
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import BackendProtocol, CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import SecretStr

TRACKING_URI = "http://127.0.0.1:5555"

# The MLflow AI Gateway -- the tracking server itself. Which model "gemma-agent" resolves to lives
# in infra/mlflow/gateway/seed_gateway.py — see L2-M1.1.1.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/2_conversation/2_deepagents"

# MLflow stores a user alongside the session. Both are trace metadata, both are
# filterable, and in production this is where the real account id goes.
USER = "tutorial-user"

# deepagents defaults to recursion_limit=9999, sized for frontier models. Capped
# so a confused run fails fast instead of burning an hour of quota.
RECURSION_LIMIT = 50

# Gemma 4 reasons before it answers, and the reasoning is charged against
# max_tokens. On some prompts at temperature 0 it never stops, and the server's
# own ceiling is the whole 262k context — measured here as a 40-minute turn. Cap
# the output so a runaway turn fails in about a minute, and stop the client from
# retrying a 20-minute gateway timeout three times over.
MAX_TOKENS = 4096
REQUEST_TIMEOUT_S = 300

WORKSPACE = Path(__file__).parent / "workspace"

RESEARCH_FILE = "/research.md"
ANALYSIS_FILE = "/analysis.md"

# A fixed first line the agent is told to write verbatim. The rest of the file is
# whatever the model produces, which cannot be asserted on — this one line can.
# The control and the leak test both key off it, so the evidence in Part 2 and
# Part 3 is a string match rather than a judgement about prose.
MARKER = "RESEARCH-NOTE-7413"

# Part 3b: one path prefix is routed to a store and shared on purpose, everything
# else stays thread-scoped. Two files, one under each, each with its own marker —
# so thread B can prove it read one and could not read the other.
MEMORY_DIR = "/memories/"
MEMORY_FILE = f"{MEMORY_DIR}prefs.md"
SCRATCH_FILE = "/scratch.md"
MEMORY_MARKER = "MEMORY-NOTE-2291"
SCRATCH_MARKER = "SCRATCH-NOTE-5806"

# The store namespace is a tuple YOU choose, and it is the whole scope of a
# StoreBackend. This one has no thread in it, so every thread sees the same
# files. Put a user id in it and the files become per-user instead.
MEMORY_NAMESPACE: tuple[str, ...] = ("memories",)


# ── Custom tools — the same two as L2-M1.1.2 ──────────────────────


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


# ── Sub-agents — each one gets its own context window ─────────────

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

# L2-M1.1.2's prompt, with one clause added: this agent is now in a
# conversation, so it is told that earlier turns exist.
SYSTEM_PROMPT = (
    "You are a technology research assistant in an ongoing conversation. You have "
    "your own tools and two sub-agents ('researcher', 'analyst') reachable with the "
    "task tool. Plan with write_todos, then do exactly what the task asks for. Save "
    "results with write_file and read earlier results with read_file. Refer back to "
    "earlier turns when the user does. Keep your final reply to a few short sentences."
)

# Four turns that each need the one before it, in a different way:
#   turn 2  needs the FILE       turn 1 wrote
#   turn 3  needs that file too, and hands it to a sub-agent
#   turn 4  needs the MESSAGES   turn 1 left behind
# A stateless agent answers turn 1 and fails the other three.
CONVERSATION = [
    (
        "Use write_todos to plan your work, then research microservices architecture "
        "with search_knowledge_base and get_industry_stats. Save what you find to "
        f"{RESEARCH_FILE}, and make its very first line exactly: {MARKER}"
    ),
    # The exit clause is not politeness. Without it, a fresh thread has no file,
    # and gemma-4 hunts for one — 13 tool calls, then a model call that never
    # stops. Part 2 depends on this turn being answerable when the file is gone.
    f"Read {RESEARCH_FILE} and tell me its very first line, word for word. If the file does not exist, say so.",
    (
        f"Ask the 'analyst' sub-agent to turn the findings in {RESEARCH_FILE} into a "
        f"recommendation, then save that analysis to {ANALYSIS_FILE}."
    ),
    "What was the very first thing I asked you to do?",
]

# Part 3b's two turns. Thread A writes one file into each scope. Thread B, which
# has never written anything, is asked for both — and the answer it gives is the
# evidence, because it cannot know either marker without reading the file.
SHARE_WRITE = (
    f"Write a file at {MEMORY_FILE} whose first line is exactly: {MEMORY_MARKER}. "
    f"Then write a file at {SCRATCH_FILE} whose first line is exactly: {SCRATCH_MARKER}. "
    "Write nothing else, and reply with one sentence."
)
SHARE_READ = (
    f"Read {MEMORY_FILE} and {SCRATCH_FILE}. For each file, tell me its first line "
    "word for word, or say that the file does not exist."
)

# One user message in, one turn summary out. Both scripts supply one of these,
# and the difference between the two IS the lesson.
TurnFn = Callable[[str, str], dict]

# Part 3 builds its own agent, so it cannot be handed a finished runner. It takes
# the recipe instead, and each script passes its own.
RunnerFactory = Callable[[Any], TurnFn]


def get_llm() -> ChatOpenAI:
    """Chat model pointed at the MLflow AI Gateway."""
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
        # The field is `max_tokens`; this is its declared alias, and the key
        # ChatOpenAI actually sends. Unsloth honours both, the type checker one.
        max_completion_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=1,
    )


def turn_config(session_id: str) -> RunnableConfig:
    """`thread_id` is what the checkpointer keys the saved state on."""
    return {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": session_id}}


def state_files(state: dict[str, Any]) -> dict[str, str]:
    """The virtual filesystem, as it sits in the graph state right now."""
    return {path: data.get("content", "") for path, data in (state.get("files") or {}).items()}


def count_tool_calls(messages: list[Any]) -> dict[str, int]:
    """Count tool calls by name. `task` is a delegation to a sub-agent."""
    counts: dict[str, int] = {}
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            counts[call["name"]] = counts.get(call["name"], 0) + 1
    return counts


# ── The deep agent, given memory ──────────────────────────────────


def build_agent(
    checkpointer: InMemorySaver,
    backend: BackendProtocol | None = None,
    store: BaseStore | None = None,
):
    """L2-M1.1.2's agent, plus one argument.

    `tools=` and `subagents=` are exactly what the turn lesson used, so the two
    lessons compare directly. `checkpointer=` is the whole difference, and it
    carries the messages, the todos AND the StateBackend files together.

    `backend=None` is the real default and means StateBackend. Part 3a passes a
    FilesystemBackend here to break the thread boundary; Part 3b passes a
    CompositeBackend, plus the LangGraph `store=` its StoreBackend route writes to.
    """
    return create_deep_agent(
        model=get_llm(),
        tools=CUSTOM_TOOLS,
        subagents=[RESEARCHER, ANALYST],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        store=store,
    )


def make_plain_runner(agent: Any) -> TurnFn:
    """The turn itself, with no MLflow stamping at all.

    Both scripts build on this, and each wraps it in its own way. The agent is
    bound HERE, in the closure, rather than passed as an argument — see the
    decorator in `main_decorator.py` for why that matters.

    `seen` records how long each thread's history was at the end of its last
    turn, so a turn counts only its OWN tool calls. It is a dict rather than an
    int because one runner can drive several threads — Part 3 does exactly
    that — and a history length only means something per thread.
    """
    seen: dict[str, int] = {}

    def run_turn(text: str, session_id: str) -> dict:
        """Note what is NOT passed to invoke: the history, the todos and the
        files. Only the new message goes in. The checkpointer restores all three
        from `thread_id`.
        """
        state = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config=turn_config(session_id),
        )
        messages = state["messages"]
        fresh = messages[seen.get(session_id, 0) :]
        seen[session_id] = len(messages)

        calls = count_tool_calls(fresh)
        files = state_files(state)
        return {
            "answer": str(messages[-1].content).strip(),
            "files": sorted(files),
            "file_chars": sum(len(c) for c in files.values()),
            "todos": len(state.get("todos") or []),
            "tool_calls": sum(calls.values()),
            "handoffs": calls.get("task", 0),
            "messages": len(messages),
        }

    return run_turn


def print_turn(idx: int, text: str, result: dict, elapsed: float) -> None:
    """One turn, in four lines: what was asked, what came back, what state holds."""
    print(f"\n  Turn {idx}  user : {text[:88]}")
    print(f"          agent: {result['answer'][:100]}")
    print(f"          files: {result['files'] or 'none'} ({result['file_chars']} chars) | todos {result['todos']}")
    print(
        f"          {result['tool_calls']} tool call(s), {result['handoffs']} handoff(s) | "
        f"{result['messages']} messages | {elapsed:.2f}s"
    )


# ── Part 1: one thread carries messages, todos AND files ──────────


def run_conversation(run_turn: TurnFn, session_id: str, label: str) -> list[dict]:
    """Drive every turn of CONVERSATION through one runner on one thread."""
    print(f"\n  session_id / thread_id: {session_id}")
    rows: list[dict] = []

    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"variant": label, "session_id": session_id, "user": USER})
        mlflow.log_params(
            {
                "turns": len(CONVERSATION),
                "model_alias": MODEL_ALIAS,
                "custom_tools": ", ".join(t.name for t in CUSTOM_TOOLS),
                "subagents": "researcher, analyst",
                "backend": "StateBackend (default)",
                "checkpointer": "InMemorySaver",
            }
        )

        for idx, text in enumerate(CONVERSATION, start=1):
            start = time.time()
            result = run_turn(text, session_id)
            elapsed = time.time() - start
            print_turn(idx, text, result, elapsed)

            rows.append(
                {
                    "turn": idx,
                    "user": text,
                    "agent": result["answer"],
                    "files": len(result["files"]),
                    "file_chars": result["file_chars"],
                    "tool_calls": result["tool_calls"],
                    "handoffs": result["handoffs"],
                    "messages": result["messages"],
                    "latency_seconds": round(elapsed, 3),
                }
            )

        mlflow.log_metrics(
            {
                "final_files": rows[-1]["files"],
                "final_file_chars": rows[-1]["file_chars"],
                "final_messages": rows[-1]["messages"],
                "total_tool_calls": sum(r["tool_calls"] for r in rows),
                "total_handoffs": sum(r["handoffs"] for r in rows),
                "avg_latency": round(sum(r["latency_seconds"] for r in rows) / len(rows), 3),
            }
        )
        mlflow.log_table(
            data={k: [r[k] for r in rows] for k in rows[0]},
            artifact_file="conversation.json",
        )

    return rows


# ── Part 2: the control — a fresh thread has no files ─────────────


def run_control(run_turn: TurnFn, session_id: str, remembered: str) -> None:
    """Ask turn 2's question on a thread that has never been used.

    Same agent, same checkpointer, same question. The file was written into the
    graph state, and the graph state is keyed by thread — so on a new thread it
    is not there.
    """
    question = CONVERSATION[1]
    result = run_turn(question, session_id)
    quoted = MARKER in result["answer"]

    print(f"\n  Question       : {question}")
    print(f"  On the thread  : {remembered[:100]}")
    print(f"  On a new thread: {result['answer'][:100]}")
    print(f"\n  Files on the new thread : {result['files'] or 'none'}")
    print(f"  New thread quoted {MARKER}: {quoted}")
    print("\n  StateBackend keeps files IN the checkpointed state, so they are")
    print("  scoped to the thread exactly like the messages are.")


# ── Part 3a: the trap — a real file has no thread ─────────────────


def run_filesystem_backend_trap(
    make_runner: RunnerFactory,
    session_a: str,
    session_b: str,
    label: str,
) -> None:
    """The same two threads, against a backend that writes to disk.

    Nothing about the agent changes except `backend=`. But a file on disk is not
    part of any thread's state, so thread B reads what thread A wrote. In a
    product that is one user reading another user's notes.

    This part takes a runner FACTORY rather than a runner, because the agent it
    needs does not exist until this function builds it. It also crosses a session
    boundary in the middle — two threads, two session ids — which is where the
    two stamping APIs stop looking alike. See `main_context.py`.
    """
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    # virtual_mode=True keeps the agent's paths rooted at "/" while the bytes
    # land under WORKSPACE — the tool calls look identical to Part 1.
    agent = build_agent(
        InMemorySaver(),
        backend=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
    )
    run_turn = make_runner(agent)

    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"variant": label, "user": USER})
        mlflow.log_params(
            {
                "backend": "FilesystemBackend",
                "checkpointer": "InMemorySaver",
                "root_dir": str(WORKSPACE),
            }
        )

        written = run_turn(CONVERSATION[0], session_a)
        print(f"\n  Thread A ({session_a}) — researched and wrote the file")
        print(f"    answer: {written['answer'][:100]}")

        read = run_turn(CONVERSATION[1], session_b)
        print(f"\n  Thread B ({session_b}) — never wrote anything")
        print(f"    answer: {read['answer'][:100]}")

        on_disk = sorted(p.name for p in WORKSPACE.rglob("*") if p.is_file())
        leaked = MARKER in read["answer"]
        print(f"\n  Files on disk: {on_disk}")
        print(f"  Thread B quoted thread A's {MARKER}: {leaked}")
        mlflow.log_metric("cross_thread_read", int(leaked))

    print("\n  A checkpointer scopes STATE to a thread. It cannot scope a disk.")
    print("  Give each session its own root_dir, or keep files in StateBackend —")
    print("  or share on purpose, per path, which is 3b.")


# ── Part 3b: the fix — one path prefix shared on purpose ──────────


def run_store_backend_share(
    make_runner: RunnerFactory,
    session_a: str,
    session_b: str,
    label: str,
) -> None:
    """The same two threads, against a backend that shares ONE path on purpose.

    `CompositeBackend` routes by path prefix. Under `/memories/` a `StoreBackend`
    keeps files in a LangGraph store, keyed by a namespace with no thread in it,
    so every thread sees them. Every other path stays in `StateBackend`, where
    the thread scopes it. One agent, two scopes, and the path decides which.

    Part 3a was the accident. This is the same test with the verdict chosen per
    file: thread B reads what thread A left under `/memories/`, and cannot read
    what it left at `/`.
    """
    # The backend gets the store directly, so this function can read it back
    # after the turn. The graph gets it too — that is where LangGraph expects it.
    store = InMemoryStore()
    backend = CompositeBackend(
        default=StateBackend(),
        routes={MEMORY_DIR: StoreBackend(store=store, namespace=lambda _runtime: MEMORY_NAMESPACE)},
    )
    agent = build_agent(InMemorySaver(), backend=backend, store=store)
    run_turn = make_runner(agent)

    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"variant": label, "user": USER})
        mlflow.log_params(
            {
                "backend": "CompositeBackend",
                "route_memories": f"{MEMORY_DIR} -> StoreBackend, namespace {MEMORY_NAMESPACE}",
                "route_default": "StateBackend",
                "checkpointer": "InMemorySaver",
            }
        )

        written = run_turn(SHARE_WRITE, session_a)
        # The route prefix is stripped before the key reaches the store, so
        # /memories/prefs.md sits in the store as /prefs.md under the namespace.
        in_store = sorted(item.key for item in store.search(MEMORY_NAMESPACE))
        print(f"\n  Thread A ({session_a}) — wrote one file into each scope")
        print(f"    answer      : {written['answer'][:100]}")
        print(f"    in the state: {written['files'] or 'none'}   (scoped by thread)")
        print(f"    in the store: {in_store or 'none'}   (scoped by namespace {MEMORY_NAMESPACE})")

        read = run_turn(SHARE_READ, session_b)
        memory_seen = MEMORY_MARKER in read["answer"]
        scratch_seen = SCRATCH_MARKER in read["answer"]
        print(f"\n  Thread B ({session_b}) — never wrote anything")
        print(f"    answer      : {read['answer'][:160]}")
        print(f"    quoted {MEMORY_MARKER} from {MEMORY_FILE}: {memory_seen}")
        print(f"    quoted {SCRATCH_MARKER} from {SCRATCH_FILE}: {scratch_seen}")
        mlflow.log_metrics({"shared_memory_read": int(memory_seen), "cross_thread_read": int(scratch_seen)})

    print("\n  Same two threads as 3a, and now the verdict is per path: /memories/ is")
    print("  shared because it was routed to a store, / is private because it stayed in state.")


# ── Part 4: reading the conversations back ────────────────────────


def read_back(session_ids: list[str]) -> None:
    """`search_sessions` returns one object per conversation, not per trace."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  No experiment found — skipping read-back.")
        return

    # Traces are logged in the background. Without this the newest turns are
    # often missing from the search results.
    mlflow.flush_trace_async_logging()

    sessions = mlflow.search_sessions(locations=[experiment.experiment_id], include_spans=False)
    found = {s.id: s for s in sessions}
    mine = [found[sid] for sid in session_ids if sid in found]

    print(f"  {len(sessions)} session(s) in this experiment, {len(mine)} from this run.\n")
    for session in mine:
        plural = "trace" if len(session) == 1 else "traces"
        print(f"  Session {session.id} — {len(session)} {plural}")
        for pos, trace in enumerate(session, start=1):
            request = " ".join(str(trace.info.request_preview or "").split())
            response = " ".join(str(trace.info.response_preview or "").split())
            print(f"    {pos}. {request[:72]}")
            print(f"       -> {response[:72]}")
        print()


def setup() -> None:
    """The connection every script makes before it logs anything."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    # DeepAgents is LangGraph underneath, so one call traces all of it — the
    # sub-agent work included, nested under the `task` span.
    mlflow.langchain.autolog(log_traces=True)
