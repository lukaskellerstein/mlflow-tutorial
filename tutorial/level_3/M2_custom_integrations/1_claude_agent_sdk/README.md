# L3-2.1 — Claude Agent SDK + MLflow Integration

**Level:** Expert
**Duration:** 2.5 hours

## Overview

This lesson demonstrates how to build a custom MLflow tracing integration for a third-party agent SDK that lacks native MLflow support. Using the Claude Agent SDK as our example, we create a reusable tracing wrapper pattern that captures the full agent lifecycle — thinking, tool selection, tool execution, and response generation — as structured MLflow spans.

The same pattern applies to any agent framework: Claude Agent SDK, Codex SDK, DeepAgents, or your own custom agent code.

## Prerequisites

- Completed: L1-5.1 (Auto Tracing), L1-5.2 (Manual Tracing)
- Completed: L2-M4 (Advanced Tracing) recommended
- MLFlow server running at http://127.0.0.1:5000
- Ollama running with `gemma4:e2b` model pulled
- No Anthropic API key required — this lesson simulates the SDK lifecycle

## Concepts

### Why Custom Integrations?

MLflow provides autologging for popular frameworks (LangChain, OpenAI, Anthropic), but many agent SDKs and frameworks do not have built-in support. When working with a third-party SDK, you need to manually instrument the code to capture execution traces, metrics, and artifacts.

### The Integration Pattern

The approach works in three layers:

1. **Identify lifecycle phases** — Every agent SDK has a similar lifecycle: receive input, reason about it, optionally use tools, and generate a response. Map the SDK's methods to these phases.

2. **Wrap with tracing** — Use `@mlflow.trace` on the top-level entry point to create a root span. Use `mlflow.start_span()` inside each lifecycle phase to create child spans. Set inputs, outputs, and attributes on every span.

3. **Log aggregate metrics** — After execution, log summary metrics (tokens, latency, tool usage) to the MLflow run for comparison and analysis.

### Claude Agent SDK Lifecycle

The real Claude Agent SDK follows this lifecycle (which we simulate):

```
Query -> think() -> select_tool() -> use_tool() -> respond() -> Result
           |              |               |              |
        reasoning    tool decision    execution     final answer
```

Each phase maps to an MLflow span in our tracing integration.

## Step-by-Step

### Step 1: Simulate the Agent SDK

We create a `ClaudeAgentSimulator` class that mirrors the real SDK's interface. Under the hood, it uses a local LLM (`gemma4:e2b` via Ollama) to generate responses. This lets us demonstrate the integration pattern without needing an API key.

```python
class ClaudeAgentSimulator:
    def think(self, query: str) -> str: ...
    def use_tool(self, tool_name: str, tool_input: str) -> str: ...
    def select_tool(self, query: str, thinking: str) -> tuple | None: ...
    def respond(self, query: str, thinking: str, context: str) -> str: ...
    def run(self, query: str) -> AgentResult: ...
```

### Step 2: Build the Tracing Wrapper

The `TracedClaudeAgent` wraps every lifecycle method with MLflow tracing:

```python
class TracedClaudeAgent:
    @mlflow.trace(name="claude_agent.run")
    def run(self, query: str) -> AgentResult:
        # Think phase
        with mlflow.start_span(name="claude_agent.think") as span:
            span.set_inputs({"query": query})
            thinking = self.agent.think(query)
            span.set_outputs({"thinking": thinking})

        # Tool phase
        with mlflow.start_span(name="claude_agent.use_tool.calculator") as span:
            span.set_inputs({"tool": "calculator", "input": "42*17+3"})
            output = self.agent.use_tool("calculator", "42*17+3")
            span.set_outputs({"output": output})

        # Respond phase
        with mlflow.start_span(name="claude_agent.respond") as span:
            response = self.agent.respond(query, thinking, context)
            span.set_outputs({"response": response})
```

Key practices:
- `@mlflow.trace` on the top-level method creates the root span
- `mlflow.start_span()` for each sub-phase creates child spans automatically nested under the root
- `set_inputs()` / `set_outputs()` / `set_attributes()` capture structured data on each span

### Step 3: Run with Nested MLflow Runs

Each query gets its own nested MLflow run, with metrics logged at both the individual and aggregate levels:

```python
with mlflow.start_run(run_name="claude_sdk_integration") as parent:
    for query in queries:
        with mlflow.start_run(run_name=f"query_{i}", nested=True):
            result = agent.run(query)
            mlflow.log_metrics({"total_tokens": result.total_tokens, ...})
```

### Step 4: Analyze Traces

After execution, we query traces from MLflow to inspect the span hierarchy:

```python
traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    return_type="list",
    flush=True,
)
for trace in traces:
    for span in trace.data.spans:
        print(f"  - {span.name} ({duration}ms)")
```

## Running the Lesson

```bash
cd tutorial/level_3/M2_custom_integrations/1_claude_agent_sdk
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L3-2.1 — Claude Agent SDK + MLflow Integration
============================================================

Part 1: Simulated Claude Agent (SDK lifecycle)
  System prompt: You are a helpful assistant...
  Available tools: ['calculator', 'lookup', 'summarizer']
  Model: gemma4:e2b (local via Ollama)

Part 2-3: TracedClaudeAgent with MLflow tracing
  Integration pattern:
    1. @mlflow.trace on the top-level run() method
    2. mlflow.start_span() for each lifecycle phase
    ...

Part 4: Running traced agent on example queries

  Query 1: What is 42 * 17 + 3?
    Thinking: I need to calculate 42 * 17 + 3...
    Tool: calculator(42 * 17 + 3) -> 717
    Response: The result of 42 * 17 + 3 is 717...
    Tokens: ~180, Duration: 4.2s

  Query 2: Tell me about MLflow.
    Thinking: The user wants to know about MLflow...
    Tool: lookup(mlflow) -> MLflow is an open-source MLOps platform...
    Response: MLflow is an open-source platform...
    Tokens: ~220, Duration: 5.1s

  Query 3: Explain why testing AI agents is important.
    Thinking: This is a general knowledge question...
    Response: Testing AI agents is important because...
    Tokens: ~160, Duration: 3.8s

Part 5: Trace analysis — span hierarchy
  Found 3 traces
  Trace a1b2c3d4e5f6... | Status: OK | Duration: 4200ms
    - claude_agent.run (4200ms)
    - claude_agent.think (1200ms)
    - claude_agent.tool_selection (800ms)
    - claude_agent.use_tool.calculator (100ms)
    - claude_agent.respond (2100ms)
  ...
```

In the MLflow UI:
- **Experiment**: L3/M2_custom_integrations/1_claude_agent_sdk
- **Runs**: Parent run with 3 nested child runs (one per query)
- **Traces tab**: Click any run to see the span hierarchy with inputs/outputs
- **Artifacts**: `claude_agent_results.json` summary

## Key Takeaways

- **Any agent SDK can be integrated** with MLflow using `@mlflow.trace` and `mlflow.start_span()` — no native support required.
- **Map SDK lifecycle to spans**: identify the phases (think, tool, respond) and create a span for each one.
- **Capture structured data**: use `set_inputs()`, `set_outputs()`, and `set_attributes()` on every span for maximum observability.
- **Wrap, don't modify**: create a wrapper class that delegates to the real SDK — this keeps the integration decoupled and reusable.
- **Log aggregate metrics**: span-level detail is for debugging; run-level metrics are for comparison and tracking.

## Adapting for the Real Claude Agent SDK

To use this pattern with the actual SDK, replace `ClaudeAgentSimulator` calls in `TracedClaudeAgent` with real SDK calls:

```python
from claude_agent_sdk import Agent  # hypothetical import

class TracedClaudeAgent:
    def __init__(self):
        self.agent = Agent(api_key="...")  # real SDK

    @mlflow.trace(name="claude_agent.run")
    def run(self, query):
        with mlflow.start_span(name="claude_agent.think") as span:
            thinking = self.agent.think(query)  # real SDK call
            span.set_outputs({"thinking": thinking})
        ...
```

## Next Steps

Continue to **L3-2.2 (Codex SDK + MLflow)** to see the same pattern applied to a TypeScript-based agent framework, including cross-language integration challenges.
