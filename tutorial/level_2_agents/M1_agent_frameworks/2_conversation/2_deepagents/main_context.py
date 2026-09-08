"""
L2-M1.2.2 — DeepAgents conversations, stamped with mlflow.tracing.context

APPROACH 2 of 2. `main_decorator.py` does the same job with `@mlflow.trace` and
`mlflow.update_current_trace`. Run both, then compare the two printouts and the
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

Here you own no span at all. One `with` block stamps every trace opened inside
it, so the traces LangChain autolog creates pick the session up on their own.
Fewer moving parts, and no wrapper function — but the root span is autolog's
rather than one you named, so the previews show the raw agent input and the
whole message list back, instead of the turn's file summary. Compare Part 4
with the other script's: that difference is the whole cost of the shorter API.

`context()` stamps a REGION, so the block goes wherever the session boundary
is. Parts 1 and 2 are one session each, so the block wraps the whole part. Both
halves of Part 3 cross a session boundary halfway through, so there the block
has to move inside the turn — see `make_context_runner`.

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


def make_context_runner(agent: Any) -> TurnFn:
    """A runner that opens its own one-turn `context()` block.

    Parts 1 and 2 do not need this: one block around the whole part covers every
    turn in it, because every turn in it shares one session id.

    Part 3 does. Its two turns are two different threads and therefore two
    different sessions, so there is no single region to wrap — the block has to
    follow the `session_id` argument down to one turn. That is the shape of
    `context()`: it stamps a region, and a region cannot hold two session ids.

    The decorator approach needs no equivalent, because it stamps per trace
    already.
    """
    plain = make_plain_runner(agent)

    def run_turn(text: str, session_id: str) -> dict:
        with mlflow.tracing.context(session_id=session_id, user=USER):
            return plain(text, session_id)

    return run_turn


def main() -> None:
    print("=" * 60)
    print("L2-M1.2.2 — DeepAgents conversations stamped with tracing.context")
    print("=" * 60)

    # The default backend is StateBackend: files live in the graph state, so the
    # checkpointer carries them exactly like it carries the messages.
    agent = build_agent(InMemorySaver())
    run_turn = make_plain_runner(agent)

    state_session = f"cnv-ctx-state-{uuid.uuid4().hex[:8]}"
    control_session = f"cnv-ctx-control-{uuid.uuid4().hex[:8]}"
    disk_a = f"cnv-ctx-disk-a-{uuid.uuid4().hex[:8]}"
    disk_b = f"cnv-ctx-disk-b-{uuid.uuid4().hex[:8]}"
    memory_a = f"cnv-ctx-memory-a-{uuid.uuid4().hex[:8]}"
    memory_b = f"cnv-ctx-memory-b-{uuid.uuid4().hex[:8]}"
    sessions = [state_session, control_session, disk_a, disk_b, memory_a, memory_b]

    print("\n" + "=" * 60)
    print("Part 1: the conversation — messages, todos and files on one thread")
    print("=" * 60)
    # No wrapper and no decorator: the runner is the shared one, unmodified. The
    # block around it is the entire stamping mechanism.
    with mlflow.tracing.context(session_id=state_session, user=USER):
        rows = run_conversation(run_turn, state_session, "tracing_context_conversation")

    print("\n" + "=" * 60)
    print("Part 2: the control — a fresh thread, empty filesystem")
    print("=" * 60)
    with mlflow.tracing.context(session_id=control_session, user=USER):
        run_control(run_turn, control_session, rows[1]["agent"])

    print("\n" + "=" * 60)
    print("Part 3: the backend decides the scope")
    print("=" * 60)
    # Two threads, two sessions, one function — in both halves. The block moves
    # inside the turn, which is what `make_context_runner` does.
    print("\n  3a. the trap — FilesystemBackend ignores the thread")
    run_filesystem_backend_trap(make_context_runner, disk_a, disk_b, "tracing_context_filesystem_leak")
    print("\n  3b. on purpose — CompositeBackend routes /memories/ to a store")
    run_store_backend_share(make_context_runner, memory_a, memory_b, "tracing_context_store_share")

    print("\n" + "=" * 60)
    print("Part 4: reading the conversations back")
    print("=" * 60)
    read_back(sessions)

    print("=" * 60)
    print("Done. In the MLflow UI, the left nav has a Sessions tab — one row per")
    print("conversation, and opening a row replays the turns in order:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print(f"  sessions: {', '.join(sessions)}")
    print("\nThe previews above come from autolog's own root span: the raw agent")
    print("input, and the whole message list back. In main_decorator.py they are")
    print("the turn's two arguments and its file summary — that is the trade.")
    print("=" * 60)


if __name__ == "__main__":
    setup()
    main()
