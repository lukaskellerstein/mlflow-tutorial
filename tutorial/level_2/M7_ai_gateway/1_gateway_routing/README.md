# L2-7.1 — Gateway Configuration and Routing

**Level:** Practitioner
**Duration:** 1 hour

## Overview

The MLflow AI Gateway provides a unified interface for deploying and managing multiple LLM providers. This lesson explores advanced endpoint configuration with fallback chains, traffic splitting for A/B testing and load balancing, budget policies for cost control, and a reference comparison of supported providers. Since the gateway requires live API keys, this lesson focuses on understanding configurations and logging reference materials to MLflow.

## Prerequisites

- Completed: L1-8.2 (AI Gateway Overview)
- MLflow server running at http://127.0.0.1:5000
- No cloud API keys required (this lesson is educational / config-focused)

## Concepts

### Endpoints and Fallback Chains

An AI Gateway **endpoint** routes requests to an LLM provider. Each endpoint can have a **fallback chain** -- an ordered list of alternative models that the gateway tries sequentially if the primary model fails (due to errors, rate limits, or outages). This keeps applications available without code changes.

### Traffic Splitting

Traffic splitting distributes requests across multiple models by **weight percentage** (weights must sum to 100%). This enables:
- **A/B testing** -- compare model quality on live traffic
- **Gradual migration** -- shift traffic from one provider to another incrementally
- **Load distribution** -- spread requests across providers to avoid rate limits

### Budget Policies

Budget policies set a spending threshold over a time window (daily, weekly, or monthly). When the threshold is exceeded, the policy either **alerts** (fires a webhook, requests continue) or **rejects** (blocks subsequent requests with HTTP 429). Spend resets automatically at each window boundary.

### Usage Tracking

When enabled on an endpoint, the gateway logs all requests as traces, providing dashboards for request volume, latency, token consumption, and cost breakdowns.

## Step-by-Step

### Step 1: Advanced Route Configuration

We define three endpoints with different purposes and fallback strategies:

```python
route_config = {
    "endpoints": [
        {
            "name": "team-chat",
            "provider": "openai",
            "model": "gpt-4o",
            "fallbacks": [
                {"provider": "openai", "model": "gpt-4o-mini"},
                {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            ],
        },
        ...
    ]
}
```

The fallback chain ensures that if `gpt-4o` hits a rate limit, the gateway automatically tries `gpt-4o-mini` (cheaper and faster), then falls back to Anthropic as a cross-provider safety net.

### Step 2: Traffic Splitting

We configure a production endpoint that splits traffic across three providers:

- 60% to OpenAI gpt-4o (primary workhorse)
- 30% to Anthropic Claude Sonnet (comparison candidate)
- 10% to Google Gemini 2.5 Pro (experimental evaluation)

This configuration lets you compare model quality on real production traffic before committing to a migration.

### Step 3: Cost Management

We define budget policies (daily alert at $50, monthly hard cap at $2,000) and simulate a week of usage data, logging metrics like total requests, token counts, cost, and latency to MLflow for tracking.

### Step 4: Provider Comparison

We create a reference table comparing six major providers across chat models, embedding models, rate limit structures, and strengths. This table is logged to MLflow as an artifact for team reference.

## Running the Lesson

```bash
cd tutorial/level_2/M7_ai_gateway/1_gateway_routing
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-7.1 — Gateway Configuration and Routing
============================================================
  Run ID: <run-id>

============================================================
Part 1: Advanced Route Configuration
============================================================
  Endpoint: 'team-chat'
    Primary:    openai / gpt-4o
    Fallback 1: openai / gpt-4o-mini  (cheaper, faster)
    Fallback 2: anthropic / claude-sonnet  (different provider)
  ...

============================================================
Part 2: Load Balancing and Traffic Splitting
============================================================
  Traffic splitting distributes requests by weight percentage.
   60%  openai / gpt-4o
   30%  anthropic / claude-sonnet-4-20250514
   10%  google / gemini-2.5-pro
  ...

============================================================
Part 3: Cost Management
============================================================
  Budget policies control spending per time window.
  ...
  Simulated 7-day usage summary:
    Total requests:      8,860
    Total cost:          $211.30
  ...

============================================================
Part 4: Provider Comparison
============================================================
  | Provider     | Chat Models            | ...
  ...

============================================================
Done!
============================================================
```

In the MLflow UI, check:
- **Artifacts tab**: `configs/` folder with endpoint, traffic split, and budget policy JSON files; `tables/` with usage data and provider comparison
- **Metrics**: total_requests, total_cost_usd, avg_daily_cost_usd, avg_p50_latency_ms

## Key Takeaways

- AI Gateway endpoints support **fallback chains** that try alternative models automatically on failure, providing high availability without code changes.
- **Traffic splitting** distributes requests by weight percentage, enabling A/B testing and gradual provider migrations on live traffic.
- **Budget policies** set spending thresholds with alert or reject actions, preventing unexpected cost overruns.
- **Usage tracking** logs all gateway requests as traces, providing dashboards for volume, latency, tokens, and cost.
- The gateway supports 100+ providers through LiteLLM, with zero-downtime configuration updates.

## Next Steps

In a production setup, you would create endpoints interactively via the MLflow UI at `http://localhost:5000/#/gateway`, configure LLM connections with real API keys, and set up budget policies under AI Gateway > Budgets. For monitoring, see L3-M3 (Production Patterns) where we integrate Grafana dashboards with gateway metrics.
