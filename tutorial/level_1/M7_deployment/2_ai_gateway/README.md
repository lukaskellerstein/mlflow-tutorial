# L1-M7.2 — AI Gateway Overview

**Level:** Essentials
**Duration:** 20 min

## Overview

The MLflow AI Gateway provides a unified, centralized interface for interacting with multiple LLM providers (OpenAI, Anthropic, Mistral, and more) through a single API endpoint. This lesson introduces the Gateway's purpose, configuration format, and when to use it instead of direct API calls.

## Prerequisites

- Completed: L1-M7.1 (Model Serving Basics)
- MLflow server running at http://127.0.0.1:5000

## Concepts

### Why an AI Gateway?

When teams work with LLMs, they typically face several challenges:

1. **API key sprawl** — every application needs its own set of provider API keys
2. **Provider lock-in** — switching from OpenAI to Anthropic requires code changes everywhere
3. **No rate limiting** — individual apps can exhaust API quotas
4. **No fallbacks** — if one provider is down, the app fails

The MLflow AI Gateway solves these by acting as a proxy between your applications and LLM providers. Applications call the gateway with a standard API, and the gateway routes requests to the configured provider.

### Supported Providers

| Provider | Route Types |
|----------|-------------|
| OpenAI | chat, completions, embeddings |
| Anthropic | chat, completions |
| Mistral | chat, completions, embeddings |
| Google (Gemini) | chat, completions |
| AWS Bedrock | chat, completions |
| Azure OpenAI | chat, completions, embeddings |
| Hugging Face TGI | chat, completions |

### Route Types

The gateway supports three standard route types:

- **`llm/v1/chat`** — conversational endpoints (messages in, message out)
- **`llm/v1/completions`** — text completion endpoints (prompt in, text out)
- **`llm/v1/embeddings`** — embedding endpoints (text in, vector out)

## Step-by-Step

### Step 1: Understand the Gateway Configuration

The gateway is configured via a YAML file that defines routes. Each route maps a name to a provider and model:

```yaml
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_key: $OPENAI_API_KEY
```

- **`name`** — the route identifier your apps use to call this endpoint
- **`route_type`** — determines the API contract (chat, completions, or embeddings)
- **`provider`** — which LLM provider handles requests
- **`model.name`** — the provider-specific model ID
- **`config`** — API keys referenced via environment variables (never hardcoded)

### Step 2: Learn the CLI Commands

Start the gateway server:

```bash
mlflow gateway start --config-path config.yaml --port 7000
```

Call a route:

```bash
curl http://localhost:7000/gateway/chat/invocations \
    -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Step 3: Compare Gateway vs Direct API

The lesson logs a comparison table as an artifact. Key takeaway: use the Gateway for team and production environments; use direct API calls for quick prototyping.

### Step 4: Review Artifacts in MLflow

The script saves both the gateway configuration YAML and the comparison table as MLflow artifacts, so you can see how to document infrastructure decisions alongside your experiments.

## Running the Lesson

```bash
cd tutorial/level_1/M7_deployment/2_ai_gateway
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: What is the MLflow AI Gateway?
============================================================

The MLflow AI Gateway (formerly MLflow Deployments for LLMs)
provides a unified interface for interacting with multiple
LLM providers through a single API endpoint.
...

============================================================
Part 2: Gateway Route Configuration
============================================================
Example gateway config (config.yaml):
...
  Saved gateway config as artifact (run: <run_id>)

============================================================
Part 3: Gateway CLI Commands
============================================================
...

============================================================
Part 4: When to Use Gateway vs Direct API
============================================================
| Feature                | AI Gateway                | Direct API                |
...
  Saved comparison table as artifact (run: <run_id>)

============================================================
Done! View artifacts in the MLflow UI:
...
============================================================
```

In the MLflow UI, open the experiment "L1/M7_deployment/2_ai_gateway" to see:
- **gateway_config_example** run with the YAML config artifact
- **gateway_vs_direct_comparison** run with the JSON comparison table

## Key Takeaways

- The AI Gateway acts as a proxy between your apps and LLM providers, centralizing key management and routing.
- Routes are configured in YAML with a name, type, provider, model, and API key reference.
- Three standard route types cover chat, completions, and embeddings use cases.
- The Gateway is best suited for teams and production; direct API calls are fine for solo prototyping.
- Provider switching only requires a config change, not code changes.

## Next Steps

Continue to **L1-M8.1 (Authentication and Permissions)** to learn how to secure your MLflow server for multi-user environments. In **Level 2 (L2-M7)**, we'll set up a live gateway with real provider routing, fallback chains, and load balancing.
