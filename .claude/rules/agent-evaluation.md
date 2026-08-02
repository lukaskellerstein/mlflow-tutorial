---
globs: ["tutorial/level_2_agents/**", "tutorial/level_3_advanced/M4_capstones/**"]
---

# Agent Evaluation — Core Focus Area

This is the most important part of the tutorial. The goal is to show how to systematically evaluate AI agents using MLFlow, with special emphasis on agents built with diverse frameworks. See `syllabus.md` for which lessons cover agents and where they fit.

## Agent Frameworks

### LangChain Agents

- Use `create_agent` from `langchain.agents` (LangChain v1.0+, NOT deprecated APIs)
- Auto-log with `mlflow.langchain.autolog()`
- Track: tool calls, reasoning steps, iterations, final answers

### LangGraph Agents

- Use `StateGraph`, nodes, edges, conditional edges
- Use `ToolNode` for tool execution nodes, `Command` for agent handoffs
- Auto-trace state transitions with `mlflow.langchain.autolog()`
- Patterns: collaboration, supervision, swarm

### Claude Agent SDK

- Anthropic's agent framework for building autonomous agents
- No native MLFlow autolog — build custom tracing integration

### DeepAgents

- LangChain-AI's multi-agent orchestration framework
- Use `create_deep_agent()` with configurable backends and sub-agent delegation
- Explore existing MLFlow integration or build custom

## What to Evaluate in Agents

### Functional Metrics

- **Task completion rate** — did the agent accomplish the goal?
- **Tool selection accuracy** — did it pick the right tools?
- **Reasoning quality** — is the chain-of-thought coherent?
- **Answer correctness** — compare against ground truth
- **Faithfulness** — does the answer match the retrieved context?

### Performance Metrics

- **Latency** — end-to-end and per-step
- **Token usage** — total and per-step
- **Number of iterations** — how many reasoning loops?
- **Cost estimation** — based on token usage

### Agent-Specific Metrics

- **State transition count** (LangGraph) — how many graph transitions?
- **Tool call count** — how many tools invoked?
- **Retry count** — how many retries on failure?
- **Handoff count** (multi-agent) — how many inter-agent handoffs?
- **Plan quality** (plan-and-execute) — is the plan coherent and complete?
- **Collaboration quality** (multi-agent) — effective task delegation?

## Evaluation Approach

1. **Define evaluation datasets** — input/expected_output pairs as pandas DataFrames
2. **Use `mlflow.evaluate()`** with appropriate model_type and metrics
3. **Create custom scorers** for agent-specific behaviors (tool selection, reasoning quality)
4. **Use LLM-as-judge** for open-ended quality assessment
5. **Compare agent variants** — different models, temperatures, prompts, architectures
6. **Use `mlflow.genai.agent_tester`** for automated test generation
7. **Use `mlflow.genai.simulators`** for conversation simulation
8. **Track everything** — every evaluation run should be logged to MLFlow with full parameters
9. **Build CI/CD gates** — automated quality thresholds before deployment

## Custom Integration Pattern

For frameworks without native MLFlow support (Claude Agent SDK, DeepAgents), use this pattern:

```python
import mlflow


@mlflow.trace
def run_agent(input_text: str) -> str:
    with mlflow.start_span(name="agent_call") as span:
        span.set_inputs({"input": input_text})
        result = agent.run(input_text)  # framework-specific call
        span.set_outputs({"output": result})
        return result
```

Wrap the framework's execution in MLFlow tracing decorators and manual spans to capture the full execution flow. For custom autolog lessons, go further and build a reusable autolog implementation that can be published as an MLflow plugin.
