---
globs: ["tutorial/level_2/M5_agent_observability/**", "tutorial/level_3/M1_agent_evaluation/**", "tutorial/level_3/M2_custom_integrations/**", "tutorial/level_3/M5_capstones/**"]
---

# Agent Evaluation — Core Focus Area

This is the most important part of the tutorial. The goal is to show how to systematically evaluate AI agents using MLFlow, with special emphasis on agents built with diverse frameworks.

## Where Agents Appear in the Curriculum

- **Level 2, M5**: Agent Observability — LangChain agents, LangGraph agents, multi-agent systems. Focus on tracing and tracking.
- **Level 3, M1**: Advanced Agent Evaluation — testing, quality metrics, architecture comparison, optimization, end-to-end pipeline. This is the deepest module.
- **Level 3, M2**: Custom Integrations — Claude Agent SDK, Codex SDK, DeepAgents, custom autolog. Integrating non-LangChain frameworks with MLflow.
- **Level 3, M5**: Capstones — full production agent platform and cross-framework benchmark.

## Agent Frameworks Covered

### 1. LangChain Agents (Level 2, M5.1)
- Use `create_react_agent` from LangChain v1.0+ (NOT deprecated APIs)
- Auto-log with `mlflow.langchain.autolog()`
- Track: tool calls, reasoning steps, iterations, final answers
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent`

### 2. LangGraph Agents (Level 2, M5.2 and M5.3)
- Use `StateGraph`, nodes, edges, conditional edges
- Auto-trace state transitions with `mlflow.langchain.autolog()`
- Single agents (M5.2) and multi-agent systems (M5.3)
- Patterns: collaboration, supervision, swarm
- Reference code (agents): `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/5_agent`
- Reference code (multi-agent): `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/6_agents`

### 3. Claude Agent SDK (Level 3, M2.1)
- Anthropic's agent framework for building autonomous agents
- No native MLFlow autolog — build custom tracing integration
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- Source code: `~/Projects/github/anthropics/claude-agent-sdk-python`

### 4. Codex SDK (Level 3, M2.2)
- OpenAI's code generation agent framework (TypeScript)
- Custom MLFlow integration needed — wrap calls with manual tracing
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`
- Source code: `~/Projects/github/openai/codex/sdk`

### 5. DeepAgents (Level 3, M2.3)
- LangChain-AI's multi-agent orchestration framework
- Explore existing MLFlow integration or build custom
- Source code: `~/Projects/github/lanchain-ai/deepagents`

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

## Custom Integration Pattern (for non-LangChain frameworks)

For frameworks without native MLFlow support (Claude Agent SDK, Codex SDK, DeepAgents), use this pattern:

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

Wrap the framework's execution in MLFlow tracing decorators and manual spans to capture the full execution flow.

For Level 3, M2.4 (Custom Autolog), go further and build a reusable autolog implementation that can be published as an MLflow plugin.
