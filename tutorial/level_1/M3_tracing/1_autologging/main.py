"""
L1-M3.1 — Autologging and Auto-Tracing

Demonstrates MLflow's automatic tracing for GenAI calls:
- Part 1: mlflow.openai.autolog() — direct OpenAI SDK with LMStudio
- Part 2: mlflow.langchain.autolog() — LangChain agent (create_agent)
- Part 3: mlflow.autolog() — universal switch enabling all integrations
- Part 4: Searching and inspecting traces programmatically
"""

import time

import mlflow
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
MODEL_NAME = "google/gemma-4-e4b"


# --------------------------------------------------------------------- #
# Part 1: mlflow.openai.autolog() — Direct OpenAI SDK
# --------------------------------------------------------------------- #


def part1_openai_autolog() -> None:
    """Trace a direct OpenAI SDK call to LMStudio."""
    print("=" * 60)
    print("Part 1: mlflow.openai.autolog() — Direct OpenAI SDK")
    print("=" * 60)

    mlflow.openai.autolog()

    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is MLflow? Answer in one sentence."}],
        temperature=0.7,
        max_tokens=1024,
    )

    print(f"  Response: {response.choices[0].message.content[:200]}")
    print()
    print("  [Autolog captured]")
    print("    - Input messages and roles")
    print("    - Output message content")
    print("    - Token usage (prompt, completion, total)")
    print("    - Latency (execution time)")
    print("    - Model name and parameters")
    print("    -> Check the Traces tab in MLflow UI")
    print()

    mlflow.openai.autolog(disable=True)


# --------------------------------------------------------------------- #
# Part 2: mlflow.langchain.autolog() — LangChain Agent
# --------------------------------------------------------------------- #


def get_current_time() -> str:
    """Return the current time as a formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def part2_langchain_autolog() -> None:
    """Trace a LangChain agent built with create_agent."""
    print("=" * 60)
    print("Part 2: mlflow.langchain.autolog() — LangChain Agent")
    print("=" * 60)

    mlflow.langchain.autolog()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        temperature=0.7,
        max_tokens=1024,
    )

    agent = create_agent(
        model=llm,
        tools=[get_current_time],
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What time is it right now?"}]}
    )

    final_message = result["messages"][-1].content
    print(f"  Response: {final_message[:200]}")
    print()
    print("  [Autolog captured]")
    print("    - Full agent execution flow (model calls, tool calls)")
    print("    - Each step as a child span in the trace")
    print("    - Tool inputs and outputs")
    print("    - State transitions in the agent graph")
    print("    -> Expand the trace to see the agent reasoning loop")
    print()

    mlflow.langchain.autolog(disable=True)


# --------------------------------------------------------------------- #
# Part 3: mlflow.autolog() — Universal Switch
# --------------------------------------------------------------------- #


def part3_universal_autolog() -> None:
    """Enable all autologging with a single call."""
    print("=" * 60)
    print("Part 3: mlflow.autolog() — Universal Switch")
    print("=" * 60)

    mlflow.autolog()

    # OpenAI SDK call — automatically traced
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Name three planets. Be brief."}],
        temperature=0.7,
        max_tokens=1024,
    )
    print(f"  OpenAI SDK:  {response.choices[0].message.content[:150]}")

    # LangChain agent call — also automatically traced
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        temperature=0.7,
        max_tokens=1024,
    )
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a concise assistant.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is experiment tracking?"}]}
    )
    print(f"  LangChain:   {result['messages'][-1].content[:150]}")

    print()
    print("  [mlflow.autolog() enabled ALL GenAI integrations at once]")
    print("  Both calls above were traced without separate autolog calls.")
    print()
    print("  Supported GenAI autolog integrations:")
    print("    mlflow.openai.autolog()         - OpenAI / OpenAI-compatible")
    print("    mlflow.langchain.autolog()      - LangChain / LangGraph")
    print("    mlflow.anthropic.autolog()      - Anthropic / Claude")
    print("    mlflow.gemini.autolog()         - Google Gemini")
    print("    mlflow.mistral.autolog()        - Mistral")
    print("    mlflow.bedrock.autolog()        - Amazon Bedrock")
    print("    mlflow.groq.autolog()           - Groq")
    print("    mlflow.litellm.autolog()        - LiteLLM")
    print("    mlflow.crewai.autolog()         - CrewAI")
    print("    mlflow.dspy.autolog()           - DSPy")
    print("    mlflow.llama_index.autolog()    - LlamaIndex")
    print("    mlflow.pydantic_ai.autolog()    - Pydantic AI")
    print("    mlflow.autogen.autolog()        - AutoGen")
    print("    mlflow.smolagents.autolog()     - HF smolagents")
    print("    mlflow.haystack.autolog()       - Haystack")
    print("    mlflow.strands.autolog()        - Strands Agents")
    print()

    mlflow.autolog(disable=True)


# --------------------------------------------------------------------- #
# Part 4: Searching and Inspecting Traces
# --------------------------------------------------------------------- #


def part4_search_traces() -> None:
    """Search and inspect traces programmatically."""
    print("=" * 60)
    print("Part 4: Searching and Inspecting Traces")
    print("=" * 60)

    experiment = mlflow.get_experiment_by_name("L1/M3_tracing/1_autologging")
    if experiment is None:
        print("  No experiment found — run Parts 1-3 first.")
        return

    exp_id = experiment.experiment_id

    traces = mlflow.search_traces(
        locations=[exp_id],
        max_results=5,
        return_type="list",
        flush=True,
    )

    print(f"  Found {len(traces)} trace(s) in the experiment\n")

    for i, trace in enumerate(traces):
        info = trace.info
        duration = info.execution_duration
        print(f"  --- Trace {i + 1} ---")
        print(f"  Trace ID:    {info.trace_id}")
        print(f"  State:       {info.state}")
        print(f"  Duration:    {duration} ms" if duration else "  Duration:    (pending)")

        spans = trace.data.spans
        print(f"  Spans ({len(spans)}):")
        for span in spans:
            parent_tag = "(root)" if span.parent_id is None else f"parent={span.parent_id[:8]}..."
            print(f"    - {span.name}  [{span.span_type}]  {parent_tag}")
        print()

    print("  Key trace concepts:")
    print("    - Trace:  An end-to-end record of one LLM invocation")
    print("    - Span:   A single step within a trace (model call, tool call)")
    print("    - Parent: Spans nest — the root span is parent to its steps")
    print("    - Each span records: inputs, outputs, start/end time, status")
    print()


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main() -> None:
    part1_openai_autolog()
    part2_langchain_autolog()
    part3_universal_autolog()
    part4_search_traces()

    print("=" * 60)
    print("Done! Open the MLflow UI to explore the traces:")
    print("  http://127.0.0.1:5000")
    print()
    print("Navigate to experiment 'L1/M3_tracing/1_autologging'")
    print("and click the Traces tab to see all captured traces.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M3_tracing/1_autologging")

    main()
