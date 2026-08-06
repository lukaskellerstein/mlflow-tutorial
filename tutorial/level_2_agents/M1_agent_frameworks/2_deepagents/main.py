"""
L2-M1.2 — DeepAgents with MLflow

DeepAgents is LangChain-AI's agent harness: `create_deep_agent()` wraps
`create_agent()` and hands the model a built-in toolkit — planning
(`write_todos`), a virtual filesystem (`ls`/`read_file`/`write_file`/`edit_file`/
`glob`/`grep`) and sub-agent delegation (`task`).

It is LangGraph underneath, so `mlflow.langchain.autolog()` traces all of it with
no custom integration.

Parts:
  1. Built-in toolkit — planning and the virtual filesystem
  2. Sub-agent orchestration — delegation with isolated context windows
  3. Backends — where a "file" actually lives, and what survives a conversation
  4. Single-agent baseline vs the orchestrator
"""

import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.langchain
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from pydantic import SecretStr

# The LiteLLM gateway from infra/. "gemma-agent" starts on the free OpenRouter
# tier and the proxy falls back to paid when free rate-limits — see
# infra/litellm/config.yaml.
GATEWAY_URL = "http://localhost:4000/v1"
GATEWAY_KEY = "sk-litellm-master"  # local dev master key, same class as admin/admin
MODEL_ALIAS = "gemma-agent"

EXPERIMENT = "L2/M1_agent_frameworks/2_deepagents"

# deepagents defaults to recursion_limit=9999, sized for frontier models. Capped
# so a confused run fails fast instead of burning an hour of free-tier quota.
RUN_CONFIG: RunnableConfig = {"recursion_limit": 50}

WORKSPACE = Path(__file__).parent / "workspace"


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ALIAS,
        base_url=GATEWAY_URL,
        api_key=SecretStr(GATEWAY_KEY),
        temperature=temperature,
    )


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
                print(f"    [{kind}] -> {tc['name']}({str(tc['args'])[:80]})")
        elif isinstance(msg, ToolMessage):
            print(f"    [ToolResult] {str(msg.content)[:100]}")
        elif getattr(msg, "content", None):
            print(f"    [{kind}] {str(msg.content)[:120]}")


def print_state_files(state: dict[str, Any]) -> None:
    """Print the virtual filesystem held in agent state."""
    files = state.get("files", {})
    if not files:
        print("    (none)")
    for path, file_data in files.items():
        content = file_data.get("content", "")
        print(f"    {path} ({len(content)} chars)")
        for line in content.split("\n")[:5]:
            print(f"      {line}")


# ── Part 1: the built-in toolkit ──────────────────────────────────────────


def part1_builtin_toolkit() -> None:
    """A deep agent plans with write_todos and saves with write_file."""
    print("\n" + "=" * 60)
    print("Part 1: Built-in toolkit — planning and the virtual filesystem")
    print("=" * 60)

    agent = create_deep_agent(
        model=get_llm(),
        tools=[search_knowledge_base, get_industry_stats],
        system_prompt=(
            "You are a technology research assistant. Use the available tools "
            "to gather facts and statistics, then use write_file to save your "
            "findings. Plan your work with write_todos first."
        ),
    )

    with mlflow.start_run(run_name="builtin_toolkit"):
        mlflow.log_params(
            {
                "agent_type": "deep_agent",
                "model_alias": MODEL_ALIAS,
                "custom_tools": "search_knowledge_base, get_industry_stats",
                "built_in_tools": "write_todos, ls, read_file, write_file, edit_file, glob, grep, task",
                "backend": "StateBackend (default)",
            }
        )

        start = time.time()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use write_todos to plan your work, then: "
                            "research microservices architecture using the knowledge base "
                            "and get industry statistics. Save a summary to /research.md."
                        ),
                    },
                ]
            },
            config=RUN_CONFIG,
        )
        duration = round(time.time() - start, 2)

        tool_calls = count_tool_calls(result["messages"])
        steps = len(result["messages"])
        mlflow.log_metrics(
            {
                "duration_s": duration,
                "total_steps": steps,
                "total_tool_calls": sum(tool_calls.values()),
            }
        )
        for name, count in tool_calls.items():
            mlflow.log_metric(f"tool_{name}", count)

        print("\n  Conversation:")
        print_conversation(result["messages"])

        todos = result.get("todos", [])
        print("\n  Todos (write_todos tool -> agent state):")
        for todo in todos:
            print(f"    [{todo['status']}] {todo['content']}")
        if not todos:
            # Worth seeing rather than hiding: `write_todos` is offered but not
            # forced, and gemma-4 usually plans by writing a plain plan file
            # instead. The planning HAPPENED — it just landed in the filesystem
            # below rather than in the todos channel. A stronger model picks the
            # dedicated tool more often.
            print("    (empty — this model planned into a file instead; see Files below)")

        print("\n  Files (StateBackend — ephemeral, lives in agent state):")
        print_state_files(result)

        print(f"\n  Duration: {duration}s | Steps: {steps} | Tools: {tool_calls}")


# ── Part 2: sub-agent orchestration ───────────────────────────────────────


def part2_subagents() -> dict:
    """Delegation via the `task` tool — each sub-agent gets its own context window."""
    print("\n" + "=" * 60)
    print("Part 2: Sub-agent orchestration (task tool)")
    print("=" * 60)

    researcher: SubAgent = {
        "name": "researcher",
        "description": "Researches ONE topic using the knowledge base and statistics tools.",
        "system_prompt": (
            "You are a research specialist. Use search_knowledge_base and "
            "get_industry_stats to gather comprehensive information about the "
            "topic you are given. Reply with structured bullet points. "
            "Your reply goes to another agent, not a human."
        ),
        "tools": [search_knowledge_base, get_industry_stats],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyzes research findings and produces a summary with recommendations.",
        "system_prompt": (
            "You are an analysis specialist. Given research findings, identify "
            "the most important patterns, trade-offs, and recommendations. "
            "Structure your analysis with sections: Key Findings, Trade-offs, "
            "Recommendations. Reply with the analysis only."
        ),
    }

    agent = create_deep_agent(
        model=get_llm(),
        subagents=[researcher, analyst],
        system_prompt=(
            "You are an orchestrator. You NEVER research or analyze yourself. "
            "Delegate research to the 'researcher' sub-agent and analysis to "
            "the 'analyst' sub-agent, using the task tool. Pass the researcher's "
            "findings to the analyst. Save the final analysis with write_file."
        ),
    )

    with mlflow.start_run(run_name="subagent_orchestration"):
        mlflow.log_params(
            {
                "agent_type": "deep_agent_orchestrator",
                "model_alias": MODEL_ALIAS,
                "subagents": "researcher, analyst",
                "pattern": "orchestrator -> researcher -> analyst",
            }
        )

        start = time.time()
        result: dict = {}

        print("\n  Streaming orchestration steps:")
        for step in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Research and analyze microservices vs monolith architecture. "
                            "Have the researcher gather facts, then the analyst produce "
                            "a comparison. Save the final analysis as /analysis.md."
                        ),
                    },
                ]
            },
            stream_mode="values",
            config=RUN_CONFIG,
        ):
            result = step
            message = step["messages"][-1]
            tool_calls_on_message = getattr(message, "tool_calls", None)
            if tool_calls_on_message:
                for tc in tool_calls_on_message:
                    if tc["name"] == "task":
                        subagent = tc["args"].get("subagent_type", "?")
                        desc = str(tc["args"].get("description", ""))[:80]
                        print(f"    [task] -> sub-agent '{subagent}': {desc}")
                    else:
                        print(f"    [tool] -> {tc['name']}({str(tc['args'])[:70]})")
            elif isinstance(message, ToolMessage):
                print(f"    [result] {str(message.content)[:110]}")

        duration = round(time.time() - start, 2)
        if not result:
            raise RuntimeError("Agent stream produced no steps")

        tool_calls = count_tool_calls(result["messages"])
        handoffs = tool_calls.get("task", 0)
        steps = len(result["messages"])
        mlflow.log_metrics(
            {
                "duration_s": duration,
                "total_steps": steps,
                "total_tool_calls": sum(tool_calls.values()),
                "subagent_handoffs": handoffs,
            }
        )

        print("\n  Files:")
        print_state_files(result)
        print(f"\n  Duration: {duration}s | Steps: {steps} | Handoffs: {handoffs}")

    return {"duration": duration, "steps": steps, "tool_calls": tool_calls, "handoffs": handoffs}


# ── Part 3: backends ──────────────────────────────────────────────────────


def part3_backends() -> None:
    """Where a "file" lives decides what survives the conversation.

    StateBackend      (default) -> ephemeral, in agent state — Parts 1 and 2
    FilesystemBackend           -> real files on disk
    StoreBackend                -> LangGraph store, outlives the conversation
    CompositeBackend            -> routes path prefixes to different backends
    """
    print("\n" + "=" * 60)
    print("Part 3: Backends — what a file is, and what outlives the run")
    print("=" * 60)

    WORKSPACE.mkdir(exist_ok=True)
    backend = CompositeBackend(
        # virtual_mode=True: "/" for the agent means the workspace dir on disk
        default=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
        # everything under /memories/ goes to the store instead
        routes={"/memories/": StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("memories",))},
    )

    agent = create_deep_agent(
        model=get_llm(),
        backend=backend,
        system_prompt=(
            "You are a personal assistant. Store notes and documents with write_file. "
            "Anything you should remember about the user long-term belongs under /memories/."
        ),
    )

    with mlflow.start_run(run_name="composite_backend"):
        mlflow.log_params(
            {
                "backend": "CompositeBackend(FilesystemBackend + StoreBackend)",
                "default_route": str(WORKSPACE),
                "store_route": "/memories/",
                "model_alias": MODEL_ALIAS,
            }
        )

        start = time.time()
        agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "My name is Lukas and my favorite language is Python. "
                            "Save a short greeting for me to /greeting.txt, and store what "
                            "you learned about me in /memories/user_profile.md."
                        ),
                    },
                ]
            },
            config=RUN_CONFIG,
        )

        print("\n  Real files now on disk in ./workspace:")
        written = [p for p in sorted(WORKSPACE.rglob("*")) if p.is_file()]
        for path in written:
            print(f"    {path.relative_to(WORKSPACE)}: {path.read_text()[:100]!r}")
        if not written:
            print("    (none — the model chose not to call write_file)")

        # A FRESH conversation: no checkpointer and no shared messages, so the
        # store is the only thing linking the two.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Check /memories/ — what do you know about me?"}]},
            config=RUN_CONFIG,
        )
        duration = round(time.time() - start, 2)

        answer = str(result["messages"][-1].content)
        print("\n  Fresh conversation, answered from the store:")
        print(f"    {answer[:300]}")

        mlflow.log_metrics({"duration_s": duration, "files_on_disk": len(written)})
        mlflow.log_text(answer, "recall_from_store.txt")


# ── Part 4: single-agent baseline vs the orchestrator ─────────────────────


def part4_comparison(multi: dict[str, Any]) -> None:
    """Same task, one agent doing everything — is delegation worth the handoffs?"""
    print("\n" + "=" * 60)
    print("Part 4: Single agent vs multi-agent")
    print("=" * 60)

    agent = create_deep_agent(
        model=get_llm(),
        tools=[search_knowledge_base, get_industry_stats],
        system_prompt=(
            "You are a generalist assistant. Research the topic using the "
            "available tools, analyze the findings, and save a complete "
            "analysis with write_file. Do everything yourself — do not "
            "delegate to sub-agents."
        ),
    )

    with mlflow.start_run(run_name="single_agent_baseline"):
        mlflow.log_params({"agent_type": "deep_agent_single", "model_alias": MODEL_ALIAS, "subagents": "none"})

        start = time.time()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Research and analyze microservices vs monolith architecture. "
                            "Use the tools to gather facts, then produce a comparison "
                            "analysis. Save it as /analysis.md."
                        ),
                    },
                ]
            },
            config=RUN_CONFIG,
        )
        duration = round(time.time() - start, 2)
        tool_calls = count_tool_calls(result["messages"])
        single = {
            "duration": duration,
            "steps": len(result["messages"]),
            "tool_calls": sum(tool_calls.values()),
        }
        mlflow.log_metrics(
            {
                "duration_s": duration,
                "total_steps": single["steps"],
                "total_tool_calls": single["tool_calls"],
                "subagent_handoffs": 0,
            }
        )

    table: dict[str, list[Any]] = {
        "approach": ["multi_agent (orchestrator + subagents)", "single_agent"],
        "duration_s": [multi["duration"], single["duration"]],
        "total_steps": [multi["steps"], single["steps"]],
        "tool_calls": [sum(multi["tool_calls"].values()), single["tool_calls"]],
        "subagent_handoffs": [multi["handoffs"], 0],
    }

    print(f"\n  {'approach':<40} {'steps':<8} {'tools':<8} {'handoffs':<10} duration")
    print("  " + "-" * 78)
    for i in range(2):
        print(
            f"  {table['approach'][i]:<40} {table['total_steps'][i]:<8} "
            f"{table['tool_calls'][i]:<8} {table['subagent_handoffs'][i]:<10} "
            f"{table['duration_s'][i]}s"
        )

    with mlflow.start_run(run_name="approach_comparison"):
        mlflow.set_tag("run_type", "comparison")
        mlflow.log_table(data=table, artifact_file="comparison.json")
        mlflow.log_metrics(
            {
                "multi_agent_duration_s": multi["duration"],
                "single_agent_duration_s": single["duration"],
                "multi_agent_steps": multi["steps"],
                "single_agent_steps": single["steps"],
            }
        )


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("L2-M1.2 — DeepAgents with MLflow")
    print("=" * 60)
    print("\nDeepAgents is built on LangGraph, so mlflow.langchain.autolog()")
    print("captures every agent step, tool call and sub-agent trace.\n")

    mlflow.langchain.autolog()

    part1_builtin_toolkit()
    multi = part2_subagents()
    part3_backends()
    part4_comparison(multi)

    print("\n" + "=" * 60)
    print("Done. View traces in the MLflow UI:")
    print(f"  http://127.0.0.1:5555 — experiment {EXPERIMENT}")
    print("  Compare the single-agent and multi-agent traces side by side.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment(EXPERIMENT)
    main()
