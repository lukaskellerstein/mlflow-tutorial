# MLFlow

Source code: /Users/lkellers/Projects/github/mlflow/mlflow

## LMStudio

LMS CLI - <https://lmstudio.ai/docs/cli>

LMS as service (headless) - <https://lmstudio.ai/docs/developer/core/headless>

Selected model: Gemma4-26B-A4B (MoE, ~4B active) - served behind every
`gemma-*` alias (`gemma-chat`, `gemma-judge`, `gemma-agent`). Chosen because it is the fastest of the local options and the
only fast one that also exists on OpenRouter, so a fallback cannot change which
model answered.

Lessons never call LMStudio directly — they go through the LiteLLM gateway on
`localhost:4000`, which maps those aliases. See `infra/litellm/config.yaml`.

## AI Agents

### Langchain

Source code: /Users/lkellers/Projects/github/langchain-ai/langchain

Examples: /Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain

### Langgraph

Source code: /Users/lkellers/Projects/github/langchain-ai/langgraph

Examples: /Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph

### Deepagents

Source code: /Users/lkellers/Projects/github/langchain-ai/deepagents

Examples: /Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/3_deepagents

### Claude Agents SDK

Source code: /Users/lkellers/Projects/github/anthropics/claude-agent-sdk-python

Examples: /Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python

## RAG

Qdrant

## Evaluation

SWE-Bench - <https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified>
