"""
L2-M1.2.1 — Multi-Turn Conversations, stamped with mlflow.tracing.context

APPROACH 2 of 2. `main_decorator.py` does the same job with `@mlflow.trace` and
`mlflow.update_current_trace`. Run both, then compare the two printouts and the
Sessions tab. Everything they share lives in `conversation.py`, so the stamping
is the only thing that can differ.

Two keys have to line up, and they are not the same key:

  thread_id   LangGraph's. It decides what the AGENT remembers.
  session_id  MLflow's. It decides which traces belong to ONE conversation.

Here you own no span at all. One `with` block stamps every trace opened inside
it, so the traces LangChain autolog creates pick the session up on their own.
Fewer moving parts, and no wrapper function — but the root span is autolog's
rather than one you named, because `context()` only writes metadata and cannot
open a span. Compare the previews in Part 3 with the other script's: that
difference is the whole cost of the shorter API.

Parts:
  1. The conversation — four turns, one trace each, one session
  2. The control — the same question on a fresh thread, to prove what carries
     the memory
  3. Reading it back — mlflow.search_sessions()
"""

import uuid

import mlflow
from conversation import (
    EXPERIMENT,
    USER,
    build_agent,
    make_plain_runner,
    read_back,
    run_control,
    run_conversation,
    setup,
)
from langgraph.checkpoint.memory import InMemorySaver


def main() -> None:
    print("=" * 60)
    print("L2-M1.2.1 — Conversations stamped with mlflow.tracing.context")
    print("=" * 60)

    # One saver holds every thread. Swap it for SqliteSaver or PostgresSaver and
    # the conversation survives the process — nothing else in this file changes.
    agent = build_agent(InMemorySaver())

    conversation_session = f"cnv-context-{uuid.uuid4().hex[:8]}"
    control_session = f"cnv-context-control-{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 60)
    print("Part 1: the conversation — create_agent + checkpointer")
    print("=" * 60)
    # No wrapper and no decorator: the runner is the shared one, unmodified. The
    # block around it is the entire stamping mechanism.
    with mlflow.tracing.context(session_id=conversation_session, user=USER):
        rows = run_conversation(make_plain_runner(agent), conversation_session, "tracing_context_conversation")

    print("\n" + "=" * 60)
    print("Part 2: the control — a fresh thread, no memory")
    print("=" * 60)
    # A fresh runner, because a fresh thread starts from an empty history.
    with mlflow.tracing.context(session_id=control_session, user=USER):
        run_control(make_plain_runner(agent), control_session, rows[1]["agent"])

    print("\n" + "=" * 60)
    print("Part 3: reading the conversation back")
    print("=" * 60)
    read_back([conversation_session, control_session])

    print("=" * 60)
    print("Done. In the MLflow UI, the left nav has a Sessions tab — one row per")
    print("conversation, and opening a row replays the turns in order:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print(f"  sessions: {conversation_session}, {control_session}")
    print("\nThe request previews above are the raw LangChain message list. In")
    print("main_decorator.py they are the turn's own inputs — that is the trade.")
    print("=" * 60)


if __name__ == "__main__":
    setup()
    main()
