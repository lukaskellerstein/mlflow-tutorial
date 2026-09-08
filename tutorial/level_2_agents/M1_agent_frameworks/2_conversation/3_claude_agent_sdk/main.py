"""
L2-M1.2.3 — Multi-Turn Conversations with the Claude Agent SDK

The other two frameworks in this group make you supply the session key: you
invent a `thread_id`, hand it to a checkpointer, and reuse it every turn.

The Claude Agent SDK works the other way round. It owns the conversation, it
assigns the session id, and it tells you what that id was AFTER the turn
finished. So the job is not "push an id down" — it is "read the id out and stamp
MLflow with it".

That inversion has a practical payoff in Part 3: because the SDK persists the
session itself, a completely new process can pick the conversation back up with
one option, and there is no checkpoint store to run.

Parts:
  1. One client, four turns — the SDK's session id becomes the MLflow session id
  2. The control — a new client is a new session, with no memory
  3. Resume — a new client that continues the SAME session
  4. Reading it back — mlflow.search_sessions()
"""

import json
from dataclasses import dataclass, field
from typing import Any

import anyio
import mlflow
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

EXPERIMENT = "L2/M1_agent_frameworks/2_conversation/3_claude_agent_sdk"

USER = "tutorial-user"

# The SDK drives the Claude Code CLI, so it authenticates the way your CLI does.
# No MLflow AI Gateway and no local model here — the gateway speaks the OpenAI API,
# this SDK speaks Anthropic's Messages API through the CLI. See L2-M1.1.3.
MODEL = "claude-sonnet-5"

# A conversation is several billed turns, so cap it. The SDK stops rather than
# running past this.
MAX_BUDGET_USD = 1.0

SYSTEM_PROMPT = (
    "You are a helpful assistant in an ongoing conversation. Use the provided "
    "tools when asked. Refer back to earlier turns when the user does. Keep "
    "answers to one short sentence."
)

# Turn 2 and turn 4 are unanswerable without the conversation. Each tool is
# named explicitly: without that Claude does the arithmetic itself and the tool
# spans stay empty.
CONVERSATION = [
    "Use the calculator tool: what is 15 * 23?",
    "Now double that number, using the calculator tool.",
    "Use the string_reverser tool on the word 'MLflow'.",
    "What was the very first thing I asked you?",
]

# Asked in Part 3, from a different client, about a turn it never saw.
RESUMED_QUESTION = "What did the string_reverser tool return earlier?"


# ── Tools — the same three as the sibling lessons ─────────────────


@tool("calculator", "Evaluate a basic arithmetic expression", {"expression": str})
async def calculator(args: dict[str, Any]) -> dict[str, Any]:
    """Runs in this process — no subprocess, no serialization boundary."""
    expression = str(args.get("expression", ""))
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        text = f"Error: expression contains invalid characters: {expression}"
    else:
        try:
            text = f"Result: {eval(expression)}"  # safe: only digits and operators allowed
        except Exception as e:
            text = f"Error evaluating '{expression}': {e}"
    return {"content": [{"type": "text", "text": text}]}


@tool("string_reverser", "Reverse the characters in a string", {"text": str})
async def string_reverser(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    return {"content": [{"type": "text", "text": text[::-1]}]}


@tool("word_counter", "Count the words in a piece of text", {"text": str})
async def word_counter(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    return {"content": [{"type": "text", "text": f"The text contains {len(text.split())} word(s)."}]}


INPROC_SERVER = create_sdk_mcp_server(
    name="inproc",
    version="1.0.0",
    tools=[calculator, string_reverser, word_counter],
)


def build_options(resume: str | None = None) -> ClaudeAgentOptions:
    """Options for one client. `resume` is the only difference in Part 3."""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        mcp_servers={"inproc": INPROC_SERVER},
        allowed_tools=[
            "mcp__inproc__calculator",
            "mcp__inproc__string_reverser",
            "mcp__inproc__word_counter",
        ],
        max_turns=3,
        max_budget_usd=MAX_BUDGET_USD,
        permission_mode="bypassPermissions",
        model=MODEL,
        # REQUIRED. Without it the CLI merges any .mcp.json in the working
        # directory with the servers above, and the tools here silently never
        # become available. See L2-M1.1.3 for the full failure mode.
        strict_mcp_config=True,
        resume=resume,
    )


@dataclass
class TurnResult:
    """What one turn produced, including the id the SDK chose for it."""

    answer: str = ""
    session_id: str = ""
    tool_calls: list[str] = field(default_factory=list)
    duration_ms: int = 0
    cost_usd: float | None = None


# ── Part 1: the SDK owns the id, MLflow borrows it ────────────────


def make_turn_runner(client: ClaudeSDKClient):
    """Bind the client OUTSIDE the traced function, and return the runner.

    `@mlflow.trace` records every argument as the span's input, so a client
    passed as a parameter would be serialized into each trace as an object
    repr — noise in every preview, and Part 4 becomes unreadable.
    """

    @mlflow.trace(name="turn", span_type="AGENT")
    async def run_turn(text: str) -> dict:
        """One user message in, one answer out — over an already-open client.

        Note what this function does NOT take: a session id. It cannot. The id
        arrives in the ResultMessage at the END of the turn, so the trace is
        stamped on the way out rather than on the way in. That is legal —
        `update_current_trace` only has to run before the trace closes.
        """
        result = TurnResult()
        parts: list[str] = []

        await client.query(text)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        # No autolog for this SDK, so the span is hand-built.
                        # It records the decision to call a tool and its input.
                        with mlflow.start_span(name=f"tool_call.{block.name}") as span:
                            span.set_inputs(block.input)
                            span.set_attributes({"tool_name": block.name, "tool_use_id": block.id})
                        result.tool_calls.append(block.name.replace("mcp__inproc__", ""))
            elif isinstance(message, ResultMessage):
                result.session_id = message.session_id
                result.duration_ms = message.duration_ms
                result.cost_usd = message.total_cost_usd

        result.answer = "".join(parts).strip()

        # The whole point of the lesson: the framework's id becomes MLflow's.
        mlflow.update_current_trace(session_id=result.session_id, user=USER)

        return {
            "answer": result.answer,
            "session_id": result.session_id,
            "tool_calls": result.tool_calls,
            "duration_ms": result.duration_ms,
        }

    return run_turn


async def run_conversation(label: str) -> tuple[str, list[dict]]:
    """Four turns over ONE open client. The client is the conversation."""
    rows: list[dict] = []
    session_id = ""

    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"variant": label, "user": USER})
        mlflow.log_params({"turns": len(CONVERSATION), "model": MODEL, "max_budget_usd": MAX_BUDGET_USD})

        async with ClaudeSDKClient(options=build_options()) as client:
            run_turn = make_turn_runner(client)

            for idx, text in enumerate(CONVERSATION, start=1):
                result = await run_turn(text)
                session_id = result["session_id"]

                print(f"\n  Turn {idx}  user : {text}")
                print(f"          agent: {result['answer'][:110]}")
                print(
                    f"          tools: {result['tool_calls'] or 'none'} | "
                    f"session {session_id[:8]}… | {result['duration_ms']}ms"
                )

                rows.append(
                    {
                        "turn": idx,
                        "user": text,
                        "agent": result["answer"],
                        "tools": json.dumps(result["tool_calls"]),
                        "session_id": session_id,
                        "duration_ms": result["duration_ms"],
                    }
                )

        distinct = {r["session_id"] for r in rows}
        print(f"\n  Distinct session ids across {len(rows)} turns: {len(distinct)}")
        print("  One client, one session. The SDK assigned it, not this script.")

        mlflow.set_tag("session_id", session_id)
        mlflow.log_metrics(
            {
                "distinct_session_ids": len(distinct),
                "avg_duration_ms": round(sum(r["duration_ms"] for r in rows) / len(rows), 1),
            }
        )
        mlflow.log_table(data={k: [r[k] for r in rows] for k in rows[0]}, artifact_file="conversation.json")

    return session_id, rows


# ── Part 2: the control — a new client is a new conversation ──────


async def run_control(remembered: str) -> str:
    """Ask turn 2's question from a client that has never been used.

    This is exactly what L2-M1.1.3 did for every one of its queries: a fresh
    `ClaudeSDKClient` per call. That lesson had no memory, and this is why.
    """
    question = CONVERSATION[1]

    with mlflow.start_run(run_name="fresh_client_control"):
        mlflow.set_tags({"variant": "fresh_client", "user": USER})
        async with ClaudeSDKClient(options=build_options()) as client:
            result = await make_turn_runner(client)(question)
        mlflow.set_tag("session_id", result["session_id"])

    print(f"\n  Question        : {question}")
    print(f"  On the session  : {remembered[:100]}")
    print(f"  On a new client : {result['answer'][:100]}")
    print("\n  A ClaudeSDKClient IS the conversation. Close it and it is over.")
    return result["session_id"]


# ── Part 3: resume — a new client, the same conversation ──────────


async def run_resumed(session_id: str) -> str:
    """Continue the Part 1 conversation from a brand-new client.

    `resume=<session_id>` is the whole change. The SDK persists the session
    itself, so nothing here runs a checkpoint store, a database or a saver —
    which is the one thing this framework gives you that the LangGraph-based
    ones do not.
    """
    with mlflow.start_run(run_name="resumed_conversation"):
        mlflow.set_tags({"variant": "resumed", "user": USER})
        mlflow.log_param("resume_from", session_id)

        async with ClaudeSDKClient(options=build_options(resume=session_id)) as client:
            result = await make_turn_runner(client)(RESUMED_QUESTION)

        same = result["session_id"] == session_id
        mlflow.log_metric("session_id_preserved", int(same))
        mlflow.set_tag("session_id", result["session_id"])

    print(f"\n  Resumed from   : {session_id}")
    print(f"  Question       : {RESUMED_QUESTION}")
    print(f"  Answer         : {result['answer'][:100]}")
    print(f"  Same session id: {same}")
    print("\n  A new client, a new process if you like — one continuous session.")
    return result["session_id"]


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
    mine = [found[sid] for sid in dict.fromkeys(session_ids) if sid in found]

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

    print("  The Part 1 session has five traces, not four: Part 3 resumed it,")
    print("  so the resumed turn joined the same conversation in MLflow too.")


# ── Main ──────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("L2-M1.2.3 — Multi-Turn Conversations with the Claude Agent SDK")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Part 1: one client, four turns")
    print("=" * 60)
    session_id, rows = await run_conversation("sdk_conversation")

    print("\n" + "=" * 60)
    print("Part 2: the control — a new client, no memory")
    print("=" * 60)
    control_session = await run_control(rows[1]["agent"])

    print("\n" + "=" * 60)
    print("Part 3: resume — a new client, the same conversation")
    print("=" * 60)
    resumed_session = await run_resumed(session_id)

    print("\n" + "=" * 60)
    print("Part 4: reading the conversations back")
    print("=" * 60)
    read_back([session_id, control_session, resumed_session])

    print("=" * 60)
    print("Done. In the MLflow UI, the left nav has a Sessions tab — one row per")
    print("conversation, and opening a row replays the turns in order:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print(f"  sessions: {session_id}, {control_session}")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT)
    anyio.run(main)
