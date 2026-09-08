"""
L2-M1.2.1 — Multi-Turn Conversations, stamped with @mlflow.trace

APPROACH 1 of 2. `main_context.py` does the same job with
`mlflow.tracing.context`. Run both, then compare the two printouts and the
Sessions tab. Everything they share lives in `conversation.py`, so the stamping
is the only thing that can differ.

Two keys have to line up, and they are not the same key:

  thread_id   LangGraph's. It decides what the AGENT remembers.
  session_id  MLflow's. It decides which traces belong to ONE conversation.

Here you own the root span. `@mlflow.trace` makes `run_turn` the root of the
turn's trace, so every span LangChain autolog produces lands underneath it, and
`update_current_trace` stamps that root with the session id. More code than the
other approach — and in exchange, the trace previews say what the turn did.

Parts:
  1. The conversation — four turns, one trace each, one session
  2. The control — the same question on a fresh thread, to prove what carries
     the memory
  3. Reading it back — mlflow.search_sessions()
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
    setup,
)
from langgraph.checkpoint.memory import InMemorySaver


def make_turn_runner(agent: Any) -> TurnFn:
    """Wrap the shared runner so the turn owns its trace and stamps the session.

    `@mlflow.trace` records every argument as the span's input. An agent passed
    as a parameter would therefore be serialized into each trace as
    `<CompiledStateGraph object at 0x...>` — noise in every preview, and it
    makes the session unreadable in Part 3. `make_plain_runner` closes over the
    agent, so the two arguments left are the two a reader cares about.
    """
    plain = make_plain_runner(agent)

    @mlflow.trace(name="turn", span_type="AGENT")
    def run_turn(text: str, session_id: str) -> dict:
        mlflow.update_current_trace(session_id=session_id, user=USER)
        return plain(text, session_id)

    return run_turn


def main() -> None:
    print("=" * 60)
    print("L2-M1.2.1 — Conversations stamped with @mlflow.trace")
    print("=" * 60)

    # One saver holds every thread. Swap it for SqliteSaver or PostgresSaver and
    # the conversation survives the process — nothing else in this file changes.
    agent = build_agent(InMemorySaver())

    conversation_session = f"cnv-decorator-{uuid.uuid4().hex[:8]}"
    control_session = f"cnv-decorator-control-{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 60)
    print("Part 1: the conversation — create_agent + checkpointer")
    print("=" * 60)
    rows = run_conversation(make_turn_runner(agent), conversation_session, "decorator_conversation")

    print("\n" + "=" * 60)
    print("Part 2: the control — a fresh thread, no memory")
    print("=" * 60)
    # A fresh runner, because a fresh thread starts from an empty history.
    run_control(make_turn_runner(agent), control_session, rows[1]["agent"])

    print("\n" + "=" * 60)
    print("Part 3: reading the conversation back")
    print("=" * 60)
    read_back([conversation_session, control_session])

    print("=" * 60)
    print("Done. In the MLflow UI, the left nav has a Sessions tab — one row per")
    print("conversation, and opening a row replays the turns in order:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print(f"  sessions: {conversation_session}, {control_session}")
    print("\nNow run main_context.py and compare the trace previews.")
    print("=" * 60)


if __name__ == "__main__":
    setup()
    main()
