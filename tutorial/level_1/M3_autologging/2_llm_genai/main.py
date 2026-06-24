"""
L1-3.2 — LLM and GenAI Autologging

Demonstrates how mlflow.langchain.autolog() automatically captures
traces for LangChain LLM calls, chains, and multi-step pipelines —
no manual logging code required.
"""

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def part1_simple_llm_call(llm: ChatOllama) -> None:
    """A single LLM invoke — autolog captures input, output, and latency."""
    print("=" * 60)
    print("Part 1: Simple LLM Call")
    print("=" * 60)

    response = llm.invoke("What are the three laws of thermodynamics? Be brief.")

    print(f"  Response: {response.content[:200]}")
    print()
    print("  [Autolog captured]")
    print("    - Input message and role")
    print("    - Output message content")
    print("    - Latency (execution time)")
    print("    - Model name and parameters")
    print("    -> Check the Traces tab in MLflow UI")
    print()


def part2_chain_with_prompt(llm: ChatOllama) -> None:
    """A prompt template -> LLM -> parser chain, fully traced."""
    print("=" * 60)
    print("Part 2: Chain with Prompt Template")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise science educator."),
            ("human", "Explain {topic} in exactly two sentences."),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": "black holes"})

    print(f"  Topic:    black holes")
    print(f"  Response: {result[:200]}")
    print()
    print("  [Autolog captured]")
    print("    - Full chain: PromptTemplate -> ChatOllama -> StrOutputParser")
    print("    - Each step as a child span in the trace")
    print("    - Template variables and rendered prompt")
    print("    - Final parsed string output")
    print("    -> Expand the trace to see the three-step pipeline")
    print()


def part3_multi_step_chain(llm: ChatOllama) -> None:
    """Two chained calls: first generate a summary, then generate a title."""
    print("=" * 60)
    print("Part 3: Multi-Step Chain")
    print("=" * 60)

    # Step A — summarize
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical writer."),
            ("human", "Write a one-paragraph summary about {topic}."),
        ]
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # Step B — generate title from summary
    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You create short, catchy titles."),
            ("human", "Create a short title for this text:\n\n{summary}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()

    # Execute both steps
    summary = summarize_chain.invoke({"topic": "quantum computing"})
    title = title_chain.invoke({"summary": summary})

    print(f"  Summary:  {summary[:150]}...")
    print(f"  Title:    {title[:100]}")
    print()
    print("  [Autolog captured]")
    print("    - Two separate traces (one per chain invocation)")
    print("    - Each trace shows prompt rendering -> LLM call -> parsing")
    print("    - Compare latencies between the two calls")
    print("    -> Look at the Traces tab to see both traces side by side")
    print()


def main() -> None:
    """Run all three autologging demonstrations."""

    # Enable LangChain autologging — this is the only line needed
    mlflow.langchain.autolog()
    print()
    print("Enabled mlflow.langchain.autolog()")
    print("All LangChain calls will be automatically traced.")
    print()

    # Create the LLM once and reuse across parts
    llm = ChatOllama(model="gemma4:e2b", temperature=0.7)

    part1_simple_llm_call(llm)
    part2_chain_with_prompt(llm)
    part3_multi_step_chain(llm)

    # Final summary
    print("=" * 60)
    print("Done! Open the MLflow UI to explore the traces:")
    print("  http://127.0.0.1:5000")
    print()
    print("Navigate to experiment 'L1/M3_autologging/2_llm_genai'")
    print("and click the Traces tab to see all captured traces.")
    print()
    print("Supported GenAI autolog integrations:")
    print("  mlflow.langchain.autolog()      - LangChain / LangGraph")
    print("  mlflow.openai.autolog()         - OpenAI")
    print("  mlflow.anthropic.autolog()      - Anthropic / Claude")
    print("  mlflow.transformers.autolog()   - Hugging Face")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M3_autologging/2_llm_genai")

    main()
