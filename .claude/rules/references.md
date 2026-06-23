# Reference Sources

Always consult these sources when building lessons. Do NOT guess at APIs — read the source code and docs first.

## MLFlow
- **Source code**: `~/Projects/github/mlflow/mlflow`
  - Python API: `~/Projects/github/mlflow/mlflow/mlflow/` (the inner `mlflow/` package)
  - GenAI module: `~/Projects/github/mlflow/mlflow/mlflow/genai/`
  - LangChain integration: `~/Projects/github/mlflow/mlflow/mlflow/langchain/`
  - Evaluation: `~/Projects/github/mlflow/mlflow/mlflow/metrics/`
  - Tracing: `~/Projects/github/mlflow/mlflow/mlflow/tracing/`
- **Documentation**: `/Users/lkellers/Projects/github/mlflow/mlflow/docs/docs`
  - Look here for API reference, tutorials, and guides
  - Check for the latest API signatures before writing code

## LangChain + LangGraph
- **Code samples**: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai`
  - LangChain basics: `.../1_langchain/`
  - LangChain agents: `.../1_langchain/10_agent/`
  - LangGraph basics: `.../2_langgraph/`
  - LangGraph agents: `.../2_langgraph/5_agent/`
  - LangGraph multi-agent: `.../2_langgraph/6_agents/`
- Use these as inspiration for agent implementations in the tutorial
- Verify that LangChain APIs used in reference code are still current (v1.0+ only)

## Temporal.io
- **Code samples**: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python`
  - AI workflows: `.../MY/5_AI/`
- Used for Module 3.3 (multi-step workflow tracing, optional path)

## Claude Agent SDK
- **Code samples**: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- **Source code**: `~/Projects/github/anthropics/claude-agent-sdk-python`
  - Read the SDK source to understand the agent lifecycle and available hooks
  - Look for existing MLFlow integration before building custom
- Used in Bonus B.2 (custom integration patterns)

## Codex SDK
- **Code samples**: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`
- **Source code**: `~/Projects/github/openai/codex/sdk`
  - Note: Codex SDK is TypeScript — the integration lesson may use a Python wrapper or subprocess calls
- Used in Bonus B.2 (custom integration patterns)

## DeepAgents
- **Source code**: `~/Projects/github/lanchain-ai/deepagents`
  - Multi-agent orchestration framework by LangChain AI
  - Read the source to understand agent patterns and how to instrument with MLFlow
- Used in Bonus B.2 (custom integration patterns)

## How to Use References

1. **Before implementing a lesson**: read the corresponding reference code to understand patterns
2. **Before using an API**: check the MLFlow source/docs to verify it exists and get the correct signature
3. **Adapt, don't copy**: reference code is inspiration — tutorial code should be clean, self-contained, and educational
4. **Check for deprecations**: LangChain APIs change frequently. Always verify against current docs
5. **Cross-reference**: when building an agent eval lesson, check both the agent framework reference AND the MLFlow eval reference
