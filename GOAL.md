# MLFlow

Source code: /Users/lkellers/Projects/github/mlflow/mlflow

## Unsloth Studio

Unsloth Studio - <https://unsloth.ai/>

It serves an OpenAI-compatible API on :8888, needs a key on every route, and
holds ONE model at a time (Settings -> API -> Model auto-switch must be ON).

Selected model: Gemma4-26B-A4B (MoE, ~4B active) - served behind every
`gemma-*` alias (`gemma-chat`, `gemma-judge`, `gemma-agent`). Chosen because it
is the fastest of the local options.

Every alias in the gateway is served by Unsloth. There is no hosted provider and
no fallback, so nothing can change which model answered.

Lessons never call Unsloth directly — they go through the MLflow AI Gateway on
`localhost:5555/gateway/mlflow/v1`, which maps those aliases. There is no
separate proxy: the tracking server IS the gateway. See
`infra/mlflow/gateway/seed_gateway.py`.

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
