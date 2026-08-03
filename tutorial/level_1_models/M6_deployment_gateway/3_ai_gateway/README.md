# L1-M6.3 — AI Gateway

**Level:** Essentials
**Duration:** 30 min

## Overview

The MLflow AI Gateway provides a unified, centralized interface for interacting with multiple LLM providers through a single API endpoint. This lesson covers the gateway's purpose, route configuration, fallback chains for high availability, traffic splitting for A/B testing, budget policies for cost control, and a comparison of supported providers.

## Prerequisites

- Completed: L1-M6.1 (Model Serving)
- MLflow server running at <http://127.0.0.1:5555>
- No cloud API keys required (this lesson is educational / config-focused)

## Concepts

### Why an AI Gateway?

When teams work with LLMs, they face several challenges:

1. **API key sprawl** -- every application needs its own provider API keys
2. **Provider lock-in** -- switching providers requires code changes everywhere
3. **No rate limiting** -- individual apps can exhaust API quotas
4. **No fallbacks** -- if one provider is down, the app fails

The AI Gateway solves these by acting as a proxy between applications and LLM providers.

### Route Types

| Route Type | Purpose |
|---|---|
| `llm/v1/chat` | Conversational endpoints (messages in, message out) |
| `llm/v1/completions` | Text completion endpoints (prompt in, text out) |
| `llm/v1/embeddings` | Embedding endpoints (text in, vector out) |

### Fallback Chains

Each endpoint can have an ordered list of alternative models. If the primary model fails (errors, rate limits, outages), the gateway tries fallbacks sequentially -- keeping applications available without code changes.

### Traffic Splitting

Distributes requests across multiple models by weight percentage (must sum to 100%). Enables A/B testing, gradual migration, and load distribution.

### Budget Policies

Set spending thresholds with two actions:
- **ALERT** -- fires a webhook, requests continue
- **REJECT** -- blocks requests with HTTP 429

## Step-by-Step

### Step 1: Understand Route Configuration

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

### Step 2: Configure Fallback Chains

```python
endpoint = {
    "name": "team-chat",
    "provider": "openai",
    "model": "gpt-4o",
    "fallbacks": [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    ],
}
```

### Step 3: Set Up Traffic Splitting

```python
traffic = {
    "endpoint": "production-chat",
    "traffic_split": [
        {"provider": "openai", "model": "gpt-4o", "weight": 60},
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "weight": 30},
        {"provider": "google", "model": "gemini-2.5-pro", "weight": 10},
    ],
}
```

### Step 4: Define Budget Policies

```python
policies = [
    {"name": "daily-alert", "amount_usd": 50, "period": "daily", "action": "alert"},
    {"name": "monthly-cap", "amount_usd": 2000, "period": "monthly", "action": "reject"},
]
```

### Step 5: Start the Gateway

```bash
mlflow gateway start --config-path config.yaml --port 7000
```

## Running the Lesson

```bash
cd tutorial/level_1_models/M6_deployment_gateway/3_ai_gateway
uv sync
uv run python main.py
```

## Expected Output

```text
============================================================
L1-M6.3 — AI Gateway
============================================================
  Run ID: <run-id>

============================================================
Part 1: What is the MLflow AI Gateway?
============================================================
  The AI Gateway provides a unified interface...

============================================================
Part 3: Fallback Chains for High Availability
============================================================
  team-chat             primary=openai/gpt-4o
                        fallbacks=[openai/gpt-4o-mini, anthropic/claude-sonnet-4-20250514]
  ...

============================================================
Part 5: Budget Policies and Usage Tracking
============================================================
  Simulated 7-day usage:
    Requests: 8,860  |  Cost: $211.30  |  Avg latency: 256.4 ms

============================================================
Part 6: Gateway vs Direct API
============================================================
  | Feature                | AI Gateway                | Direct API                |
  ...
```

In the MLflow UI, check the experiment for:
- Gateway configuration YAML artifact
- Endpoint, traffic split, and budget policy JSON configs
- Usage metrics and provider comparison table

## Key Takeaways

- The AI Gateway centralizes API key management and routing for all LLM providers
- Fallback chains provide high availability -- if one provider fails, the next is tried automatically
- Traffic splitting enables A/B testing and gradual provider migration on live traffic
- Budget policies prevent unexpected cost overruns with alert or reject actions
- Provider switching requires only a config change, not code changes
- Use the Gateway for teams and production; direct API calls are fine for prototyping

## Next Steps

This completes the Deployment and Gateway module. Continue to Level 2 (Agents) to build and observe LLM agents with LangChain and LangGraph.
