"""L1-8.2 — AI Gateway Overview: unified LLM endpoint management,
route configuration, CLI commands, and Gateway vs direct API."""

import json, os, tempfile
import mlflow

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
    ("Latency overhead", "Small proxy overhead", "None"),
    ("Best for", "Teams, production", "Prototyping, single dev"),
]

def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")

def main() -> None:
    section("Part 1: What is the MLflow AI Gateway?")
    print("The AI Gateway provides a unified interface for multiple LLM")
    print("providers through a single API endpoint.\n")
    print("Key capabilities: unified API, centralized API key management,")
    print("rate limiting, fallback routes, and usage tracking.\n")

    section("Part 2: Gateway Route Configuration")
    print("Example config.yaml:\n")
    print(GATEWAY_CONFIG)
    print("Fields: name (route ID), route_type (chat/completions/embeddings),")
    print("provider, model.name, config (API keys via env vars).\n")
    with mlflow.start_run(run_name="gateway_config_example"):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gateway_config.yaml")
            with open(p, "w") as f:
                f.write(GATEWAY_CONFIG)
            mlflow.log_artifact(p)
        mlflow.set_tag("lesson", "L1-8.2")
        print("  Saved gateway config as artifact.")

    section("Part 3: Gateway CLI Commands")
    print("Start:  mlflow gateway start --config-path config.yaml --port 7000")
    print("Chat:   curl localhost:7000/gateway/chat/invocations \\")
    print("""          -d '{"messages":[{"role":"user","content":"Hello"}]}'\n""")
    print("Providers: OpenAI, Anthropic, Mistral, Google (Gemini),")
    print("           AWS Bedrock, Azure OpenAI, Hugging Face TGI\n")

    section("Part 4: Gateway vs Direct API")
    w = (22, 25, 25)
    print(f"| {'Feature':<{w[0]}} | {'AI Gateway':<{w[1]}} | {'Direct API':<{w[2]}} |")
    print(f"|{'-'*(w[0]+2)}|{'-'*(w[1]+2)}|{'-'*(w[2]+2)}|")
    for feat, gw, direct in COMPARISON:
        print(f"| {feat:<{w[0]}} | {gw:<{w[1]}} | {direct:<{w[2]}} |")
    print()
    with mlflow.start_run(run_name="gateway_vs_direct_comparison"):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gateway_vs_direct.json")
            rows = [{"feature": f, "gateway": g, "direct": x}
                    for f, g, x in COMPARISON]
            with open(p, "w") as f:
                json.dump(rows, f, indent=2)
            mlflow.log_artifact(p)
        mlflow.set_tag("lesson", "L1-8.2")
        print("  Saved comparison table as artifact.")

    section("Done!")
    print("View artifacts: http://127.0.0.1:5000/#/experiments")
    print("In L2-M7, we'll set up a live gateway with fallbacks.\n")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M8_deployment/2_ai_gateway")
    main()
