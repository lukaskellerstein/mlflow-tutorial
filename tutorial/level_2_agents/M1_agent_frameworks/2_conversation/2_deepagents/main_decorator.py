"""
L2-M1.2.2 — DeepAgents conversations, stamped with @mlflow.trace

APPROACH 1 of 2. `main_context.py` does the same job with
`mlflow.tracing.context`. Run both, then compare the two printouts and the
Sessions tab. Everything they share lives in `conversation.py`, so the stamping
is the only thing that can differ.

This is the same split as L2-M1.2.1, on a different framework, and that is the
point of repeating it. The two stamping APIs are MLflow's, not LangChain's, so
they behave identically on a deep agent that also carries a todo list, a virtual
filesystem and two sub-agents.

The agent is L2-M1.1.2's, unchanged apart from the checkpointer — same custom
tools, same `researcher` and `analyst` sub-agents.

Two keys have to line up, and they are not the same key:

  thread_id   LangGraph's. It decides what the AGENT remembers — here that is
              the messages, the todos AND the files.
  session_id  MLflow's. It decides which traces belong to ONE conversation.

Here you own the root span. `@mlflow.trace` makes `run_turn` the root of the
turn's trace, so every span LangChain autolog produces lands underneath it, and
`update_current_trace` stamps that root with the session id. More code than the
other approach — and in exchange, the trace previews carry the turn's own
summary, including which files the thread was holding.

Parts:
  1. The conversation — messages, todos AND files carried by one thread
  2. The control — a fresh thread starts with an empty filesystem
  3. The backend decides the scope — 3a: a FilesystemBackend leaks a file
     across threads by accident; 3b: a CompositeBackend routes /memories/ to a
     StoreBackend and shares that one prefix on purpose
  4. Reading it back — mlflow.search_sessions()
"""

import uuid
from typing import Any

import mlflow
from conversation import (
    EXPERIMENT,
    USER,
    TurnFn,
    build_agent,
    make_plain_runner,
    read_back,
    run_control,
    run_conversation,
    run_filesystem_backend_trap,
    run_store_backend_share,
    setup,
)
from langgraph.checkpoint.memory import InMemorySaver


def make_turn_runner(agent: Any) -> TurnFn:
    """Wrap the shared runner so the turn owns its trace and stamps the session.

    `@mlflow.trace` records every argument as the span's input. An agent passed
    as a parameter would therefore be serialized into each trace as
    `<CompiledStateGraph object at 0x...>` — noise in every preview, and it
    makes the session unreadable in Part 4. `make_plain_runner` closes over the
    agent, so the two arguments left are the two a reader cares about.

    One factory serves all three parts, including the trap. The session id is an
    ARGUMENT here, so a runner can stamp two different sessions without being
    rebuilt — which is exactly what Part 3 needs.
    """
    plain = make_plain_runner(agent)

    @mlflow.trace(name="turn", span_type="AGENT")
    def run_turn(text: str, session_id: str) -> dict:
        mlflow.update_current_trace(session_id=session_id, user=USER)
        return plain(text, session_id)

    return run_turn


def main() -> None:
    print("=" * 60)
    print("L2-M1.2.2 — DeepAgents conversations stamped with @mlflow.trace")
    print("=" * 60)

    # The default backend is StateBackend: files live in the graph state, so the
    # checkpointer carries them exactly like it carries the messages.
    agent = build_agent(InMemorySaver())
    run_turn = make_turn_runner(agent)

    state_session = f"cnv-dec-state-{uuid.uuid4().hex[:8]}"
    control_session = f"cnv-dec-control-{uuid.uuid4().hex[:8]}"
    disk_a = f"cnv-dec-disk-a-{uuid.uuid4().hex[:8]}"
    disk_b = f"cnv-dec-disk-b-{uuid.uuid4().hex[:8]}"
    memory_a = f"cnv-dec-memory-a-{uuid.uuid4().hex[:8]}"
    memory_b = f"cnv-dec-memory-b-{uuid.uuid4().hex[:8]}"
    sessions = [state_session, control_session, disk_a, disk_b, memory_a, memory_b]

    print("\n" + "=" * 60)
    print("Part 1: the conversation — messages, todos and files on one thread")
    print("=" * 60)
    rows = run_conversation(run_turn, state_session, "decorator_conversation")

    print("\n" + "=" * 60)
    print("Part 2: the control — a fresh thread, empty filesystem")
    print("=" * 60)
    # The same runner, on a new thread. Its per-turn bookkeeping is keyed by
    # session id, so a second thread does not need a second runner.
    run_control(run_turn, control_session, rows[1]["agent"])

    print("\n" + "=" * 60)
    print("Part 3: the backend decides the scope")
    print("=" * 60)
    # Both halves build their own agent, so both take the runner FACTORY. The
    # decorator stamps per trace, so the same factory serves two sessions each.
    print("\n  3a. the trap — FilesystemBackend ignores the thread")
    run_filesystem_backend_trap(make_turn_runner, disk_a, disk_b, "decorator_filesystem_leak")
    print("\n  3b. on purpose — CompositeBackend routes /memories/ to a store")
    run_store_backend_share(make_turn_runner, memory_a, memory_b, "decorator_store_share")

    print("\n" + "=" * 60)
    print("Part 4: reading the conversations back")
    print("=" * 60)
    read_back(sessions)

    print("=" * 60)
    print("Done. In the MLflow UI, the left nav has a Sessions tab — one row per")
    print("conversation, and opening a row replays the turns in order:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print(f"  sessions: {', '.join(sessions)}")
    print("\nNow run main_context.py and compare the trace previews.")
    print("=" * 60)


if __name__ == "__main__":
    setup()
    main()
