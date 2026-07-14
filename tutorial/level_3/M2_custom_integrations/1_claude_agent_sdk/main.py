"""
L3-2.1 — Claude Agent SDK + MLflow Integration

Demonstrates how to build a custom MLflow tracing integration for a
third-party agent SDK (using Claude Agent SDK as the example).

Since the real SDK requires an Anthropic API key, this lesson simulates
the SDK's agent lifecycle with a local LLM (google/gemma-4-26b-a4b via LMStudio) and
focuses on the INTEGRATION PATTERN — the same approach works for any
agent framework that lacks native MLflow support.

Parts:
  1. Simulated Claude Agent class (mimics SDK lifecycle)
  2. MLflow tracing wrappers (@mlflow.trace + start_span)
  3. Reusable TracedClaudeAgent wrapper with automatic metrics
  4. Run the traced agent on example queries (nested MLflow runs)
  5. Trace analysis — query spans, show hierarchy
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Part 1: Simulated Claude Agent SDK classes
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked a question, think step by step. "
    "If a tool is available, decide whether to use it. Always provide a clear, "
    "concise final answer."
)

AVAILABLE_TOOLS = {
    "calculator": "Evaluate a mathematical expression (e.g. '2+3*4')",
    "lookup": "Look up a fact from a knowledge base",
    "summarizer": "Summarize a long piece of text",
}


@dataclass
class ToolCall:
    """Represents a single tool invocation by the agent."""
    name: str
    input: str
    output: str


@dataclass
class AgentResult:
    """Complete result of an agent execution."""
    query: str
    thinking: str
    tool_calls: list[ToolCall]
    response: str
    total_tokens: int
    duration_s: float


class ClaudeAgentSimulator:
    """Simulates the Claude Agent SDK lifecycle using a local LLM.

    Real SDK methods this mirrors:
      - agent.think()   -> internal reasoning / chain-of-thought
      - agent.use_tool() -> tool selection and execution
      - agent.respond()  -> final answer generation
    """

    def __init__(self, model: str = "google/gemma-4-26b-a4b", temperature: float = 0.3):
        self.llm = ChatOpenAI(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=temperature,
        )
        self.model = model
        self.temperature = temperature
        self.tools = AVAILABLE_TOOLS

    def think(self, query: str) -> str:
        """Generate internal reasoning about the query (chain-of-thought)."""
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Analyze this query and explain your reasoning about how to "
                f"answer it. Available tools: {json.dumps(self.tools)}.\n\n"
                f"Query: {query}\n\n"
                f"Provide brief reasoning (2-3 sentences)."
            )),
        ]
        result = self.llm.invoke(messages)
        return result.content

    def use_tool(self, tool_name: str, tool_input: str) -> str:
        """Simulate executing a tool and returning its output."""
        if tool_name == "calculator":
            allowed = set("0123456789+-*/.() ")
            if all(ch in allowed for ch in tool_input):
                try:
                    return str(eval(tool_input))
                except Exception as e:
                    return f"Error: {e}"
            return f"Invalid expression: {tool_input}"
        elif tool_name == "lookup":
            facts = {
                "python": "Python was created by Guido van Rossum in 1991.",
                "mlflow": "MLflow is an open-source MLOps platform by Databricks.",
                "claude": "Claude is an AI assistant made by Anthropic.",
            }
            key = tool_input.lower().strip()
            for k, v in facts.items():
                if k in key:
                    return v
            return f"No fact found for: {tool_input}"
        elif tool_name == "summarizer":
            messages = [
                HumanMessage(content=f"Summarize in one sentence: {tool_input}")
            ]
            return self.llm.invoke(messages).content
        return f"Unknown tool: {tool_name}"

    def select_tool(self, query: str, thinking: str) -> tuple[str, str] | None:
        """Decide which tool to use based on the query and reasoning."""
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Based on this query and reasoning, should a tool be used?\n"
                f"Query: {query}\nReasoning: {thinking}\n\n"
                f"Available tools: {json.dumps(self.tools)}\n\n"
                f"If a tool should be used, respond with ONLY a JSON object: "
                f'{{"tool": "<name>", "input": "<input>"}}\n'
                f"If no tool is needed, respond with: NONE"
            )),
        ]
        result = self.llm.invoke(messages).content.strip()
        if "NONE" in result.upper() or "{" not in result:
            return None
        try:
            start = result.index("{")
            end = result.rindex("}") + 1
            parsed = json.loads(result[start:end])
            return parsed.get("tool", ""), parsed.get("input", "")
        except (json.JSONDecodeError, ValueError):
            return None

    def respond(self, query: str, thinking: str, context: str = "") -> str:
        """Generate the final response to the user."""
        ctx_part = f"\nAdditional context: {context}" if context else ""
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Provide a clear, concise answer.\n"
                f"Query: {query}\n"
                f"Your reasoning: {thinking}{ctx_part}\n\n"
                f"Answer directly and briefly."
            )),
        ]
        result = self.llm.invoke(messages)
        return result.content

    def run(self, query: str) -> AgentResult:
        """Execute the full agent lifecycle: think -> tool -> respond."""
        start = time.time()
        token_estimate = 0

        # Step 1: Think
        thinking = self.think(query)
        token_estimate += len(thinking.split()) * 2  # rough estimate

        # Step 2: Tool selection and execution
        tool_calls: list[ToolCall] = []
        tool_decision = self.select_tool(query, thinking)
        context = ""
        if tool_decision:
            tool_name, tool_input = tool_decision
            tool_output = self.use_tool(tool_name, tool_input)
            tool_calls.append(ToolCall(tool_name, tool_input, tool_output))
            context = f"Tool '{tool_name}' returned: {tool_output}"
            token_estimate += len(tool_output.split()) * 2

        # Step 3: Respond
        response = self.respond(query, thinking, context)
        token_estimate += len(response.split()) * 2

        return AgentResult(
            query=query,
            thinking=thinking,
            tool_calls=tool_calls,
            response=response,
            total_tokens=token_estimate,
            duration_s=round(time.time() - start, 2),
        )


# ---------------------------------------------------------------------------
# Part 2 & 3: TracedClaudeAgent — reusable MLflow tracing wrapper
# ---------------------------------------------------------------------------

class TracedClaudeAgent:
    """Wraps ClaudeAgentSimulator with automatic MLflow tracing.

    This pattern works for ANY third-party agent SDK:
      1. Wrap each lifecycle method with @mlflow.trace or start_span()
      2. Capture inputs, outputs, and metadata on each span
      3. Log aggregate metrics to the MLflow run

    To adapt this for the real Claude Agent SDK, replace the
    ClaudeAgentSimulator calls with actual SDK calls.
    """

    def __init__(self, model: str = "google/gemma-4-26b-a4b", temperature: float = 0.3):
        self.agent = ClaudeAgentSimulator(model=model, temperature=temperature)
        self.model = model
        self.temperature = temperature

    @mlflow.trace(name="claude_agent.run")
    def run(self, query: str) -> AgentResult:
        """Traced agent execution — creates a root span with child spans."""
        start = time.time()
        token_estimate = 0

        # --- Think phase ---
        with mlflow.start_span(name="claude_agent.think") as span:
            span.set_inputs({"query": query})
            span.set_attributes({"phase": "thinking", "model": self.model})
            thinking = self.agent.think(query)
            tokens = len(thinking.split()) * 2
            token_estimate += tokens
            span.set_outputs({"thinking": thinking, "token_estimate": tokens})

        # --- Tool selection and execution ---
        tool_calls: list[ToolCall] = []
        context = ""
        with mlflow.start_span(name="claude_agent.tool_selection") as span:
            span.set_inputs({"query": query, "thinking": thinking[:200]})
            tool_decision = self.agent.select_tool(query, thinking)
            span.set_outputs({"tool_selected": tool_decision is not None,
                              "tool": tool_decision})

        if tool_decision:
            tool_name, tool_input = tool_decision
            with mlflow.start_span(name=f"claude_agent.use_tool.{tool_name}") as span:
                span.set_inputs({"tool": tool_name, "input": tool_input})
                span.set_attributes({"tool_name": tool_name})
                tool_output = self.agent.use_tool(tool_name, tool_input)
                tool_calls.append(ToolCall(tool_name, tool_input, tool_output))
                context = f"Tool '{tool_name}' returned: {tool_output}"
                tokens = len(tool_output.split()) * 2
                token_estimate += tokens
                span.set_outputs({"output": tool_output, "token_estimate": tokens})

        # --- Respond phase ---
        with mlflow.start_span(name="claude_agent.respond") as span:
            span.set_inputs({"query": query, "context": context[:200]})
            span.set_attributes({"phase": "responding", "model": self.model})
            response = self.agent.respond(query, thinking, context)
            tokens = len(response.split()) * 2
            token_estimate += tokens
            span.set_outputs({"response": response, "token_estimate": tokens})

        duration = round(time.time() - start, 2)
        return AgentResult(
            query=query,
            thinking=thinking,
            tool_calls=tool_calls,
            response=response,
            total_tokens=token_estimate,
            duration_s=duration,
        )


# ---------------------------------------------------------------------------
# Part 4: Run traced agent on example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What is 42 * 17 + 3?",
    "Tell me about MLflow.",
    "Explain why testing AI agents is important.",
]


def run_examples(agent: TracedClaudeAgent) -> list[AgentResult]:
    """Run the traced agent on example queries with nested MLflow runs."""
    results: list[AgentResult] = []

    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n  Query {i}: {query}")
        with mlflow.start_run(run_name=f"query_{i}", nested=True):
            mlflow.log_params({
                "query": query[:250],
                "model": agent.model,
                "temperature": agent.temperature,
            })

            result = agent.run(query)
            results.append(result)

            # Log metrics for this query
            mlflow.log_metrics({
                "total_tokens": result.total_tokens,
                "tool_calls": len(result.tool_calls),
                "thinking_length": len(result.thinking),
                "response_length": len(result.response),
                "duration_s": result.duration_s,
            })
            mlflow.set_tags({
                "has_tool_call": str(len(result.tool_calls) > 0),
                "tools_used": json.dumps([tc.name for tc in result.tool_calls]),
            })

            print(f"    Thinking: {result.thinking[:80]}...")
            if result.tool_calls:
                for tc in result.tool_calls:
                    print(f"    Tool: {tc.name}({tc.input}) -> {tc.output[:60]}")
            print(f"    Response: {result.response[:100]}...")
            print(f"    Tokens: ~{result.total_tokens}, Duration: {result.duration_s}s")

    return results


# ---------------------------------------------------------------------------
# Part 5: Trace analysis
# ---------------------------------------------------------------------------

def analyze_traces() -> None:
    """Query traces from MLflow and display the span hierarchy."""
    experiment = mlflow.get_experiment_by_name(
        "L3/M2_custom_integrations/1_claude_agent_sdk"
    )
    if experiment is None:
        print("  Experiment not found.")
        return

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        return_type="list",
        flush=True,
    )
    print(f"  Found {len(traces)} traces")

    for trace in traces[:3]:
        request_id = trace.info.request_id
        status = trace.info.status
        duration_ms = trace.info.execution_time_ms
        print(f"\n  Trace {request_id[:12]}... | Status: {status} | "
              f"Duration: {duration_ms}ms")

        # Show span hierarchy
        spans = trace.data.spans
        for span in spans:
            indent = "    "
            name = span.name
            span_dur = (span.end_time_ns - span.start_time_ns) / 1e6
            print(f"{indent}- {name} ({span_dur:.0f}ms)")

    # Summary statistics
    if traces:
        durations = [t.info.execution_time_ms for t in traces
                     if t.info.execution_time_ms is not None]
        if durations:
            avg_dur = sum(durations) / len(durations)
            print(f"\n  Average trace duration: {avg_dur:.0f}ms")
            print(f"  Total traces: {len(traces)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("L3-2.1 — Claude Agent SDK + MLflow Integration")
    print("=" * 60)
    print("\nThis lesson demonstrates how to build a custom MLflow")
    print("tracing integration for ANY third-party agent SDK.")
    print("We simulate the Claude Agent SDK lifecycle with a local LLM.\n")

    # --- Part 1: Show the simulated agent ---
    print("=" * 60)
    print("Part 1: Simulated Claude Agent (SDK lifecycle)")
    print("=" * 60)
    print(f"  System prompt: {AGENT_SYSTEM_PROMPT[:60]}...")
    print(f"  Available tools: {list(AVAILABLE_TOOLS.keys())}")
    print(f"  Model: google/gemma-4-26b-a4b (local via LMStudio)")

    # --- Part 2-3: Create traced agent ---
    print("\n" + "=" * 60)
    print("Part 2-3: TracedClaudeAgent with MLflow tracing")
    print("=" * 60)
    print("  Integration pattern:")
    print("    1. @mlflow.trace on the top-level run() method")
    print("    2. mlflow.start_span() for each lifecycle phase:")
    print("       - think  (chain-of-thought reasoning)")
    print("       - tool_selection  (decide which tool to use)")
    print("       - use_tool.<name>  (execute the selected tool)")
    print("       - respond  (generate final answer)")
    print("    3. Log inputs/outputs/attributes on every span")

    agent = TracedClaudeAgent(model="google/gemma-4-26b-a4b", temperature=0.3)

    # --- Part 4: Run example queries ---
    print("\n" + "=" * 60)
    print("Part 4: Running traced agent on example queries")
    print("=" * 60)

    with mlflow.start_run(run_name="claude_sdk_integration") as parent_run:
        mlflow.log_params({
            "agent_type": "claude_agent_simulator",
            "model": agent.model,
            "temperature": agent.temperature,
            "num_queries": len(EXAMPLE_QUERIES),
            "tools": json.dumps(list(AVAILABLE_TOOLS.keys())),
        })
        mlflow.set_tags({
            "framework": "claude_agent_sdk",
            "integration_type": "custom_tracing",
        })

        results = run_examples(agent)

        # Log aggregate metrics on parent run
        total_tokens = sum(r.total_tokens for r in results)
        total_tool_calls = sum(len(r.tool_calls) for r in results)
        avg_duration = sum(r.duration_s for r in results) / len(results)
        mlflow.log_metrics({
            "total_tokens_all": total_tokens,
            "total_tool_calls_all": total_tool_calls,
            "avg_duration_s": round(avg_duration, 2),
            "num_queries": len(results),
        })

        # Save results summary as artifact
        summary = {
            "model": agent.model,
            "queries": len(results),
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "results": [
                {
                    "query": r.query,
                    "response": r.response[:200],
                    "tools_used": [tc.name for tc in r.tool_calls],
                    "tokens": r.total_tokens,
                    "duration_s": r.duration_s,
                }
                for r in results
            ],
        }
        summary_path = "/tmp/claude_agent_results.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        parent_run_id = parent_run.info.run_id

    # --- Part 5: Trace analysis ---
    print("\n" + "=" * 60)
    print("Part 5: Trace analysis — span hierarchy")
    print("=" * 60)
    analyze_traces()

    print("\n" + "=" * 60)
    print("Integration Pattern Summary")
    print("=" * 60)
    print("  To integrate ANY third-party agent SDK with MLflow:")
    print("  1. Identify the SDK's lifecycle methods (think/act/respond)")
    print("  2. Create a wrapper class that mirrors the SDK interface")
    print("  3. Use @mlflow.trace on the top-level entry point")
    print("  4. Use mlflow.start_span() for each internal phase")
    print("  5. Set inputs/outputs/attributes on every span")
    print("  6. Log aggregate metrics to MLflow runs")
    print()
    print(f"  Parent run ID: {parent_run_id}")
    print(f"  View in MLflow UI: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L3/M2_custom_integrations/1_claude_agent_sdk")
    main()
