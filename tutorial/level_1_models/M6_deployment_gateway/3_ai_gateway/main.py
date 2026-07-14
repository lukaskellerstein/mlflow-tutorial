"""L1-M6.3 — AI Gateway

Combines gateway overview with routing configuration:
- What the AI Gateway is and when to use it
- Route configuration with YAML
- Fallback chains for high availability
- Traffic splitting for A/B testing and load balancing
- Budget policies for cost control
- Provider comparison table
"""

import json
import os
import tempfile

import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M6_deployment_gateway/3_ai_gateway")


GATEWAY_CONFIG = """\
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_key: $OPENAI_API_KEY
  - name: completions
    route_type: llm/v1/completions
    model:
      provider: anthropic
      name: claude-3-sonnet-20240229
      config:
        anthropic_api_key: $ANTHROPIC_API_KEY
  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-ada-002
      config:
        openai_api_key: $OPENAI_API_KEY
"""

COMPARISON = [
    ("API key management", "Centralized, secure", "Per-app, in env vars"),
    ("Provider switching", "Change config, no code", "Requires code changes"),
    ("Rate limiting", "Built-in", "Manual implementation"),
    ("Fallback routing", "Built-in", "Manual implementation"),
    ("Usage tracking", "Automatic", "Manual implementation"),
    ("Setup complexity", "Requires config + server", "Minimal"),
    ("Best for", "Teams, production", "Prototyping, single dev"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def log_json(data: dict, filename: str, path: str = "configs") -> None:
    """Log a dict as a JSON artifact."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, filename)
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(fp, artifact_path=path)


def main() -> None:
    print("=" * 60)
    print("L1-M6.3 — AI Gateway")
    print("=" * 60)

    with mlflow.start_run(run_name="ai_gateway_overview") as run:
        print(f"  Run ID: {run.info.run_id}")

        # ── Part 1: Gateway Overview ──────────────────────────────
        section("Part 1: What is the MLflow AI Gateway?")
        print("  The AI Gateway provides a unified interface for multiple LLM")
        print("  providers through a single API endpoint.\n")
        print("  Key capabilities: unified API, centralized API key management,")
        print("  rate limiting, fallback routes, and usage tracking.\n")
        print("  Supported providers: OpenAI, Anthropic, Mistral, Google (Gemini),")
        print("  AWS Bedrock, Azure OpenAI, Hugging Face TGI, Groq\n")

        # ── Part 2: Route Configuration ───────────────────────────
        section("Part 2: Route Configuration")
        print("  Example config.yaml:\n")
        print(GATEWAY_CONFIG)
        print("  Fields: name, route_type (chat/completions/embeddings),")
        print("  provider, model.name, config (API keys via env vars).\n")
        print("  CLI: mlflow gateway start --config-path config.yaml --port 7000")
        print("  Call: curl localhost:7000/gateway/chat/invocations \\")
        print('        -d \'{"messages":[{"role":"user","content":"Hello"}]}\'')

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gateway_config.yaml")
            with open(p, "w") as f:
                f.write(GATEWAY_CONFIG)
            mlflow.log_artifact(p)
        print("\n  Saved gateway config as artifact.")

        # ── Part 3: Fallback Chains ───────────────────────────────
        section("Part 3: Fallback Chains for High Availability")
        route_config = {"endpoints": [
            {"name": "team-chat", "provider": "openai", "model": "gpt-4o",
             "fallbacks": [
                 {"provider": "openai", "model": "gpt-4o-mini"},
                 {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}]},
            {"name": "fast-completions", "provider": "anthropic",
             "model": "claude-haiku-4-20250414",
             "fallbacks": [{"provider": "openai", "model": "gpt-4o-mini"}]},
            {"name": "embeddings", "provider": "openai",
             "model": "text-embedding-3-small",
             "fallbacks": [{"provider": "openai", "model": "text-embedding-3-large"}]},
        ]}
        for ep in route_config["endpoints"]:
            fb = ", ".join(f"{f['provider']}/{f['model']}" for f in ep["fallbacks"])
            print(f"  {ep['name']:20s}  primary={ep['provider']}/{ep['model']}")
            print(f"  {'':20s}  fallbacks=[{fb}]")
        print("\n  Fallbacks trigger on errors/rate limits; tried in order.")
        log_json(route_config, "endpoint_config.json")
        mlflow.log_param("num_endpoints", len(route_config["endpoints"]))

        # ── Part 4: Traffic Splitting ─────────────────────────────
        section("Part 4: Traffic Splitting (A/B Testing)")
        traffic_config = {"endpoint": "production-chat", "traffic_split": [
            {"provider": "openai", "model": "gpt-4o", "weight": 60},
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "weight": 30},
            {"provider": "google", "model": "gemini-2.5-pro", "weight": 10},
        ]}
        print("  Weights must sum to 100%. Example 'production-chat':")
        for e in traffic_config["traffic_split"]:
            print(f"    {e['weight']:>3}%  {e['provider']} / {e['model']}")
        print("  Use cases: A/B testing, gradual migration, load distribution")
        log_json(traffic_config, "traffic_split_config.json")

        # ── Part 5: Cost Management ───────────────────────────────
        section("Part 5: Budget Policies and Usage Tracking")
        budget_config = {"policies": [
            {"name": "daily-alert", "amount_usd": 50, "period": "daily", "action": "alert"},
            {"name": "monthly-hard-cap", "amount_usd": 2000, "period": "monthly", "action": "reject"},
        ]}
        print("  Actions: ALERT (webhook) | REJECT (HTTP 429)")
        for p in budget_config["policies"]:
            print(f"  '{p['name']}': ${p['amount_usd']}/{p['period']} -> {p['action']}")

        usage_df = pd.DataFrame({
            "day": [f"2026-07-{d:02d}" for d in range(7, 14)],
            "requests": [1200, 1350, 980, 1100, 1450, 1280, 1500],
            "cost_usd": [28.50, 32.25, 23.40, 26.50, 34.75, 30.40, 35.50],
            "p50_latency_ms": [245, 260, 230, 250, 275, 255, 280],
        })
        totals = {
            "total_requests": int(usage_df["requests"].sum()),
            "total_cost_usd": round(usage_df["cost_usd"].sum(), 2),
            "avg_daily_cost_usd": round(usage_df["cost_usd"].mean(), 2),
            "avg_p50_latency_ms": round(usage_df["p50_latency_ms"].mean(), 1),
        }
        print(f"\n  Simulated 7-day usage:")
        print(f"    Requests: {totals['total_requests']:,}  |  "
              f"Cost: ${totals['total_cost_usd']}  |  "
              f"Avg latency: {totals['avg_p50_latency_ms']} ms")
        mlflow.log_metrics(totals)
        mlflow.log_table(data=usage_df, artifact_file="tables/daily_usage.json")
        log_json(budget_config, "budget_policies.json")

        # ── Part 6: Gateway vs Direct API + Provider Comparison ───
        section("Part 6: Gateway vs Direct API")
        w = (22, 25, 25)
        print(f"  | {'Feature':<{w[0]}} | {'AI Gateway':<{w[1]}} | {'Direct API':<{w[2]}} |")
        print(f"  |{'-'*(w[0]+2)}|{'-'*(w[1]+2)}|{'-'*(w[2]+2)}|")
        for feat, gw, direct in COMPARISON:
            print(f"  | {feat:<{w[0]}} | {gw:<{w[1]}} | {direct:<{w[2]}} |")

        providers_df = pd.DataFrame({
            "provider": ["OpenAI", "Anthropic", "Google", "AWS Bedrock", "Mistral", "Groq"],
            "chat_models": ["gpt-4o, 4o-mini", "claude-4, haiku",
                            "gemini-2.5-pro/flash", "claude, llama",
                            "large, small", "llama, mixtral"],
            "strengths": ["Broad ecosystem", "Long context", "Multimodal",
                          "Enterprise/VPC", "EU-hosted, open-weight",
                          "Ultra-low latency"],
        })
        print("\n  Provider Overview:")
        for _, r in providers_df.iterrows():
            print(f"    {r['provider']:12s} | {r['chat_models']:20s} | {r['strengths']}")
        mlflow.log_table(data=providers_df, artifact_file="tables/provider_comparison.json")
        mlflow.set_tag("lesson", "L1-M6.3")

    section("Done!")
    print(f"  Experiment: L1/M6_deployment_gateway/3_ai_gateway")
    print(f"  UI: http://127.0.0.1:5000")
    print("  Artifacts: gateway config, endpoint config, traffic split,")
    print("             budget policies, usage data, provider comparison")


if __name__ == "__main__":
    main()
