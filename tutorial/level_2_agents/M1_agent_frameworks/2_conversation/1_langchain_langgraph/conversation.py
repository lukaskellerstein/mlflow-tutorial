"""
L2-M1.2.1 — everything the two approaches hold fixed.

`main_decorator.py` and `main_context.py` both import this module. They do the
same job and differ in ONE thing: how a turn gets stamped with the session id.
The agent, the tools, the four turns, the run-level logging, the control and the
read-back all live here.

That is not tidiness. If each script carried its own copy, the two could drift,
and a difference in the MLflow UI would stop being evidence about the stamping
API — it could just be two files doing different work. Sharing this module is
what makes the comparison mean something.
"""

import time
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

TRACKING_URI = "http://127.0.0.1:5555"

# The MLflow AI Gateway -- the tracking server itself, not a provider directly. Which model
# "gemma-agent" resolves to lives in infra/mlflow/gateway/seed_gateway.py — see L2-M1.1.1.
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys at all
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/2_conversation/1_langchain_langgraph"

# MLflow stores a user alongside the session. Both are trace metadata, both are
# filterable, and in production this is where the real account id goes.
USER = "tutorial-user"

SYSTEM_PROMPT = (
    "You are a helpful assistant in an ongoing conversation. Use the provided "
    "tools for calculation, string reversal, and word counting. Refer back to "
    "earlier turns when the user does. Keep answers to one short sentence."
)

# Turn 2 and turn 4 are unanswerable without memory. That is deliberate: it is
# what makes this a conversation rather than four tasks in a row.
CONVERSATION = [
    "What is 15 * 23?",
    "Now double that number.",
    "Reverse the word 'MLflow' for me.",
    "What was the very first thing I asked you?",
]

# One user message in, one turn summary out. Both scripts supply one of these,
# and the difference between the two IS the lesson.
TurnFn = Callable[[str, str], dict]


# ── Tools — the same three as L2-M1.1.1, so the two lessons compare ───────


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Supports +, -, *, /, and parentheses.

    Args:
        expression: A math expression string, e.g. '15 * 23'.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return f"Error: expression contains invalid characters: {expression}"
    try:
        result = eval(expression)  # safe: only digits and operators allowed
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def string_reverser(text: str) -> str:
    """Reverse the characters in a given string.

    Args:
        text: The string to reverse.
    """
    return text[::-1]


@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text.

    Args:
        text: The text whose words should be counted.
    """
    return f"The text contains {len(text.split())} word(s)."


TOOLS = [calculator, string_reverser, word_counter]


def get_llm() -> ChatOpenAI:
    """Chat model pointed at the MLflow AI Gateway."""
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=0.0,
    )


# ── The agent, given memory ───────────────────────────────────────


def build_agent(checkpointer: InMemorySaver):
    """create_agent takes the checkpointer directly — that is the whole change.

    Without it the agent is stateless and every invoke starts from an empty
    message list, which is exactly what L2-M1.1.1 did.
    """
    return create_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def make_plain_runner(agent: Any) -> TurnFn:
    """The turn itself, with no MLflow stamping at all.

    Both scripts build on this, and each wraps it in its own way. The agent is
    bound HERE, in the closure, rather than passed as an argument — see the
    decorator in `main_decorator.py` for why that matters.

    The closure also carries `seen`, the length of the history at the end of the
    previous turn, so each turn counts only its own tool calls. A fresh runner
    therefore means a fresh count, which is what the control needs.
    """
    seen = 0

    def run_turn(text: str, session_id: str) -> dict:
        """Note what is NOT passed to invoke: the history. Only the new message
        goes in. The checkpointer restores everything else from `thread_id`.
        """
        nonlocal seen
        state = agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": session_id}},
        )
        messages = state["messages"]
        fresh = messages[seen:]
        seen = len(messages)
        return {
            "answer": str(messages[-1].content).strip(),
            "tool_calls": sum(1 for m in fresh if m.type == "tool"),
            "history_messages": len(messages),
        }

    return run_turn


# ── The conversation, and the control ─────────────────────────────


def run_conversation(run_turn: TurnFn, session_id: str, label: str) -> list[dict]:
    """Drive every turn of CONVERSATION through one runner on one thread."""
    print(f"\n  session_id / thread_id: {session_id}")
    rows: list[dict] = []

    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"variant": label, "session_id": session_id, "user": USER})
        mlflow.log_params({"turns": len(CONVERSATION), "model_alias": MODEL_ALIAS})

        for idx, text in enumerate(CONVERSATION, start=1):
            start = time.time()
            result = run_turn(text, session_id)
            elapsed = time.time() - start

            print(f"\n  Turn {idx}  user : {text}")
            print(f"          agent: {result['answer'][:110]}")
            print(
                f"          {result['tool_calls']} tool call(s), "
                f"history now {result['history_messages']} messages, {elapsed:.2f}s"
            )

            rows.append(
                {
                    "turn": idx,
                    "user": text,
                    "agent": result["answer"],
                    "tool_calls": result["tool_calls"],
                    "history_messages": result["history_messages"],
                    "latency_seconds": round(elapsed, 3),
                }
            )

        mlflow.log_metrics(
            {
                "total_tool_calls": sum(r["tool_calls"] for r in rows),
                "final_history_messages": rows[-1]["history_messages"],
                "avg_latency": round(sum(r["latency_seconds"] for r in rows) / len(rows), 3),
            }
        )
        mlflow.log_table(
            data={k: [r[k] for r in rows] for k in rows[0]},
            artifact_file="conversation.json",
        )

    return rows


def run_control(run_turn: TurnFn, session_id: str, remembered: str) -> None:
    """Ask turn 2's question again, on a thread that has never been used.

    Same agent, same checkpointer, same question. The ONLY variable is the
    thread id, so whatever the answer loses is what the thread was carrying.
    Pass a FRESH runner: this turn starts from an empty history.
    """
    question = CONVERSATION[1]
    result = run_turn(question, session_id)

    print(f"\n  Question       : {question}")
    print(f"  On the thread  : {remembered[:100]}")
    print(f"  On a new thread: {result['answer'][:100]}")
    print("\n  Same agent, same checkpointer. The thread id is the memory.")


# ── Reading the conversation back ─────────────────────────────────


def read_back(session_ids: list[str]) -> None:
    """`search_sessions` returns one object per conversation, not per trace.

    Each Session holds its traces already sorted oldest first, so iterating it
    replays the discussion in the order it happened.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("  No experiment found — skipping read-back.")
        return

    # Traces are logged in the background. Without this the newest turns are
    # often missing from the search results.
    mlflow.flush_trace_async_logging()

    sessions = mlflow.search_sessions(locations=[experiment.experiment_id], include_spans=False)
    # Sessions come back newest first. Reorder them into the order this run
    # produced them, so the printout follows the script.
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

    print("  The one-trace session is the control. A session is whatever shares")
    print("  a session id — length is an outcome, not a setting.")


def setup() -> None:
    """The connection every script makes before it logs anything."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    # create_agent returns a compiled StateGraph, so the LangChain autolog covers
    # it — one call instruments every agent in this lesson.
    mlflow.langchain.autolog(log_traces=True)
