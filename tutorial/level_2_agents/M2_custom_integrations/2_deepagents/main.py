"""
L2-M2.2 — DeepAgents + MLflow: Multi-Agent Orchestration with Tracing

Demonstrates the real DeepAgents library (LangChain-AI's agent harness)
with MLflow integration:
  1. Basic deep agent with custom tools — auto-traced via mlflow.langchain.autolog()
  2. Sub-agents with isolated context windows (task tool delegation)
  3. Orchestration metrics tracked and compared in MLflow
"""

import time

import mlflow
import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-4-26b-a4b",
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Custom tools — added to DeepAgents' built-in suite, never replace it
# ---------------------------------------------------------------------------
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


def count_tool_calls(messages: list) -> dict[str, int]:
    """Count tool calls by name from the message history."""
    counts: dict[str, int] = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc["name"]
            counts[name] = counts.get(name, 0) + 1
    return counts


def print_conversation(messages: list) -> None:
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


# ── Part 1: Basic Deep Agent with MLflow Auto-Tracing ─────────────────────
def part1_basic_agent() -> dict:
    print("\n" + "=" * 60)
    print("Part 1: Basic Deep Agent with Custom Tools")
    print("=" * 60)

    llm = get_llm()

    agent = create_deep_agent(
        model=llm,
        tools=[search_knowledge_base, get_industry_stats],
        system_prompt=(
            "You are a technology research assistant. Use the available tools "
            "to gather facts and statistics, then use write_file to save your "
            "findings. Plan your work with write_todos first."
        ),
    )

    with mlflow.start_run(run_name="basic_deep_agent") as run:
        mlflow.log_params({
            "agent_type": "deep_agent",
            "model": "google/gemma-4-26b-a4b",
            "custom_tools": "search_knowledge_base, get_industry_stats",
            "built_in_tools": "write_todos, ls, read_file, write_file, edit_file, glob, grep, task",
        })

        start = time.time()
        result = agent.invoke(
            {"messages": [
                {"role": "user", "content": (
                    "Use write_todos to plan your work, then: "
                    "research microservices architecture using the knowledge base "
                    "and get industry statistics. Save a summary to /research.md."
                )},
            ]},
            config={"recursion_limit": 50},
        )
        duration = round(time.time() - start, 2)

        tool_calls = count_tool_calls(result["messages"])
        total_steps = len(result["messages"])

        mlflow.log_metrics({
            "duration_s": duration,
            "total_steps": total_steps,
            "total_tool_calls": sum(tool_calls.values()),
        })
        for tool_name, count in tool_calls.items():
            mlflow.log_metric(f"tool_{tool_name}", count)

        print("\n  Conversation:")
        print_conversation(result["messages"])

        print("\n  Todos:")
        for todo in result.get("todos", []):
            print(f"    [{todo['status']}] {todo['content']}")

        print("\n  Files (StateBackend — ephemeral, in agent state):")
        for path, file_data in result.get("files", {}).items():
            content = file_data.get("content", "")
            print(f"    {path} ({len(content)} chars)")
            for line in content.split("\n")[:5]:
                print(f"      {line}")
            if content.count("\n") > 5:
                print(f"      ... ({content.count(chr(10)) - 5} more lines)")

        print(f"\n  Duration: {duration}s | Steps: {total_steps} | Tool calls: {tool_calls}")
        return {"run_id": run.info.run_id, "duration": duration,
                "steps": total_steps, "tool_calls": tool_calls}


# ── Part 2: Sub-agent Orchestration ───────────────────────────────────────
def part2_subagent_orchestration() -> dict:
    print("\n" + "=" * 60)
    print("Part 2: Sub-agent Orchestration (task tool)")
    print("=" * 60)

    llm = get_llm(temperature=0.0)

    researcher = {
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

    analyst = {
        "name": "analyst",
        "description": "Analyzes research findings and produces a clear summary with recommendations.",
        "system_prompt": (
            "You are an analysis specialist. Given research findings, identify "
            "the most important patterns, trade-offs, and recommendations. "
            "Structure your analysis with sections: Key Findings, Trade-offs, "
            "Recommendations. Reply with the analysis only."
        ),
    }

    agent = create_deep_agent(
        model=llm,
        subagents=[researcher, analyst],
        system_prompt=(
            "You are an orchestrator. You NEVER research or analyze yourself. "
            "Delegate research to the 'researcher' sub-agent and analysis to "
            "the 'analyst' sub-agent, using the task tool. Pass the researcher's "
            "findings to the analyst. Save the final analysis with write_file."
        ),
    )

    with mlflow.start_run(run_name="subagent_orchestration") as run:
        mlflow.log_params({
            "agent_type": "deep_agent_orchestrator",
            "model": "google/gemma-4-26b-a4b",
            "subagents": "researcher, analyst",
            "pattern": "orchestrator -> researcher -> analyst",
        })

        start = time.time()
        result = None

        print("\n  Streaming orchestration steps:")
        for step in agent.stream(
            {"messages": [
                {"role": "user", "content": (
                    "Research and analyze microservices vs monolith architecture. "
                    "Have the researcher gather facts, then the analyst produce "
                    "a comparison. Save the final analysis as /analysis.md."
                )},
            ]},
            stream_mode="values",
            config={"recursion_limit": 50},
        ):
            result = step
            message = step["messages"][-1]
            tool_calls_list = getattr(message, "tool_calls", None)
            if tool_calls_list:
                for tc in tool_calls_list:
                    if tc["name"] == "task":
                        subagent = tc["args"].get("subagent_type", "?")
                        desc = str(tc["args"].get("description", ""))[:100]
                        print(f"    [task] -> sub-agent '{subagent}': {desc}")
                    else:
                        print(f"    [tool] -> {tc['name']}({str(tc['args'])[:80]})")
            elif isinstance(message, ToolMessage):
                print(f"    [result] {str(message.content)[:120]}")

        duration = round(time.time() - start, 2)
        tool_calls = count_tool_calls(result["messages"])
        total_steps = len(result["messages"])
        task_calls = tool_calls.get("task", 0)

        mlflow.log_metrics({
            "duration_s": duration,
            "total_steps": total_steps,
            "total_tool_calls": sum(tool_calls.values()),
            "subagent_handoffs": task_calls,
        })

        print(f"\n  Files:")
        for path, file_data in result.get("files", {}).items():
            content = file_data.get("content", "")
            print(f"    {path} ({len(content)} chars)")
            for line in content.split("\n")[:5]:
                print(f"      {line}")

        print(f"\n  Duration: {duration}s | Steps: {total_steps}")
        print(f"  Sub-agent handoffs (task calls): {task_calls}")
        print(f"  Tool calls: {tool_calls}")
        return {"run_id": run.info.run_id, "duration": duration,
                "steps": total_steps, "tool_calls": tool_calls,
                "handoffs": task_calls}


# ── Part 3: Single Agent Comparison ───────────────────────────────────────
def part3_comparison(multi_metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("Part 3: Single Agent vs Multi-Agent Comparison")
    print("=" * 60)

    llm = get_llm(temperature=0.0)

    agent = create_deep_agent(
        model=llm,
        tools=[search_knowledge_base, get_industry_stats],
        system_prompt=(
            "You are a generalist assistant. Research the topic using the "
            "available tools, analyze the findings, and save a complete "
            "analysis with write_file. Do everything yourself — do not "
            "delegate to sub-agents."
        ),
    )

    with mlflow.start_run(run_name="single_agent_baseline") as run:
        mlflow.log_params({
            "agent_type": "deep_agent_single",
            "model": "google/gemma-4-26b-a4b",
            "subagents": "none",
        })

        start = time.time()
        result = agent.invoke(
            {"messages": [
                {"role": "user", "content": (
                    "Research and analyze microservices vs monolith architecture. "
                    "Use the tools to gather facts, then produce a comparison "
                    "analysis. Save it as /analysis.md."
                )},
            ]},
            config={"recursion_limit": 50},
        )
        duration = round(time.time() - start, 2)
        tool_calls = count_tool_calls(result["messages"])
        total_steps = len(result["messages"])

        mlflow.log_metrics({
            "duration_s": duration,
            "total_steps": total_steps,
            "total_tool_calls": sum(tool_calls.values()),
            "subagent_handoffs": 0,
        })

        single_metrics = {"duration": duration, "steps": total_steps,
                          "tool_calls": sum(tool_calls.values()), "handoffs": 0}

    comparison = pd.DataFrame([
        {
            "approach": "multi_agent (orchestrator + subagents)",
            "duration_s": multi_metrics["duration"],
            "total_steps": multi_metrics["steps"],
            "tool_calls": sum(multi_metrics["tool_calls"].values()),
            "subagent_handoffs": multi_metrics["handoffs"],
        },
        {
            "approach": "single_agent",
            "duration_s": single_metrics["duration"],
            "total_steps": single_metrics["steps"],
            "tool_calls": single_metrics["tool_calls"],
            "subagent_handoffs": 0,
        },
    ])
    print("\n  Comparison:")
    print(comparison.to_string(index=False))

    with mlflow.start_run(run_name="approach_comparison"):
        mlflow.set_tag("run_type", "comparison")
        mlflow.log_metrics({
            "multi_agent_duration_s": multi_metrics["duration"],
            "single_agent_duration_s": single_metrics["duration"],
            "multi_agent_steps": multi_metrics["steps"],
            "single_agent_steps": single_metrics["steps"],
        })
        mlflow.log_table(comparison, artifact_file="comparison.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("L2-M2.2 — DeepAgents + MLflow")
    print("=" * 60)
    print("\nDeepAgents is built on LangGraph, so mlflow.langchain.autolog()")
    print("captures all agent steps, tool calls, and sub-agent traces.\n")

    mlflow.langchain.autolog()

    part1_basic_agent()
    multi_metrics = part2_subagent_orchestration()
    part3_comparison(multi_metrics)

    print("\n" + "=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("  Experiment: L2/M2_custom_integrations/2_deepagents")
    print("  Compare traces between single-agent and multi-agent runs.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M2_custom_integrations/2_deepagents")
    main()
