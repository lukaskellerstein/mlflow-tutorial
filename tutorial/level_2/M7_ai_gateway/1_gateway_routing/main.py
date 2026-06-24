"""L2-7.1 — Gateway Configuration and Routing

Deep dive into MLflow AI Gateway: endpoint configuration with fallback
chains, traffic splitting, cost management, and provider comparison.
"""

import json, os, tempfile
import pandas as pd
import mlflow

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L2/M7_ai_gateway/1_gateway_routing"

def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

def log_json(data: dict, filename: str, path: str = "configs") -> None:
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, filename)
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(fp, artifact_path=path)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("=" * 60)
    print("L2-7.1 — Gateway Configuration and Routing")
    print("=" * 60)

    with mlflow.start_run(run_name="gateway_routing_deep_dive") as run:
        print(f"  Run ID: {run.info.run_id}")

        section("Part 1: Advanced Route Configuration")
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

        section("Part 2: Traffic Splitting (Load Balancing)")
        traffic_config = {"endpoint": "production-chat", "traffic_split": [
            {"provider": "openai", "model": "gpt-4o", "weight": 60},
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "weight": 30},
            {"provider": "google", "model": "gemini-2.5-pro", "weight": 10},
        ], "fallbacks": [{"provider": "openai", "model": "gpt-4o-mini"}]}
        print("  Weights must sum to 100%. Example 'production-chat':")
        for e in traffic_config["traffic_split"]:
            print(f"    {e['weight']:>3}%  {e['provider']} / {e['model']}")
        print("  Use cases: A/B testing, gradual migration, load distribution")
        log_json(traffic_config, "traffic_split_config.json")
        print("  Saved traffic split config as artifact.")

        section("Part 3: Cost Management")
        budget_config = {"policies": [
            {"name": "daily-alert", "amount_usd": 50,
             "period": "daily", "action": "alert"},
            {"name": "monthly-hard-cap", "amount_usd": 2000,
             "period": "monthly", "action": "reject"},
        ]}
        print("  Actions: ALERT (webhook) | REJECT (HTTP 429)")
        print("  Windows: daily | weekly | monthly (auto-reset)\n")
        for p in budget_config["policies"]:
            print(f"  '{p['name']}': ${p['amount_usd']}/{p['period']} -> {p['action']}")

        usage_df = pd.DataFrame({
            "day": [f"2026-06-{d:02d}" for d in range(17, 24)],
            "requests": [1200, 1350, 980, 1100, 1450, 1280, 1500],
            "input_tokens": [450_000, 510_000, 370_000, 420_000, 550_000, 480_000, 560_000],
            "output_tokens": [120_000, 135_000, 98_000, 110_000, 145_000, 128_000, 150_000],
            "cost_usd": [28.50, 32.25, 23.40, 26.50, 34.75, 30.40, 35.50],
            "p50_latency_ms": [245, 260, 230, 250, 275, 255, 280],
        })
        totals = {
            "total_requests": int(usage_df["requests"].sum()),
            "total_input_tokens": int(usage_df["input_tokens"].sum()),
            "total_output_tokens": int(usage_df["output_tokens"].sum()),
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
        print("  Logged usage metrics and budget config.")

        section("Part 4: Provider Comparison")
        providers_df = pd.DataFrame({
            "provider": ["OpenAI", "Anthropic", "Google", "AWS Bedrock", "Mistral", "Groq"],
            "chat_models": ["gpt-4o, 4o-mini", "claude-4, haiku",
                            "gemini-2.5-pro/flash", "claude, llama",
                            "large, small", "llama, mixtral"],
            "embedding_models": ["ada-002, 3-small/large", "voyager-3",
                                 "text-embedding-004", "titan-embed", "embed", "-"],
            "rate_limits": ["Tier-based", "Tier-based", "Per-minute",
                            "Account-based", "Tier-based", "Generous free"],
            "strengths": ["Broad ecosystem", "Long context", "Multimodal",
                          "Enterprise/VPC", "EU-hosted, open-weight",
                          "Ultra-low latency"],
        })
        widths = [12, 20, 22, 14, 22]
        hdrs = ["Provider", "Chat Models", "Embeddings", "Rate Limits", "Strengths"]
        print("  " + " | ".join(f"{h:<{w}}" for h, w in zip(hdrs, widths)))
        print("  " + "-+-".join("-" * w for w in widths))
        for _, r in providers_df.iterrows():
            vals = [r["provider"], r["chat_models"], r["embedding_models"],
                    r["rate_limits"], r["strengths"]]
            print("  " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)))
        mlflow.log_table(data=providers_df, artifact_file="tables/provider_comparison.json")
        mlflow.set_tag("lesson", "L2-7.1")
        print("\n  Logged provider comparison table.")

    section("Done!")
    print(f"  Experiment: {EXPERIMENT_NAME}  |  UI: {TRACKING_URI}")
    print("  Artifacts: configs/ (endpoint, traffic split, budget)")
    print("             tables/ (daily usage, provider comparison)")
    print("  Next: create live endpoints at http://localhost:5000/#/gateway")


if __name__ == "__main__":
    main()
