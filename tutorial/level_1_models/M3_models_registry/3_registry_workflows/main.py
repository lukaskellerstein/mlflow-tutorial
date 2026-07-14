"""
L1-M3.3 -- Registry Workflows

Full model registry lifecycle in one lesson:
- Part 1: Define and log two LLM configurations as PyFunc models
- Part 2: Register both as versions of the same model
- Part 3: Evaluate both on test prompts
- Part 4: Promote the best to champion, runner-up to challenger
- Part 5: Load champion by alias and demonstrate serving
- Part 6: Comparison summary table
"""

import time

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from openai import OpenAI

MODEL_NAME = "L1-llm-assistant"
LLM_MODEL = "google/gemma-4-e4b"

MODEL_CONFIGS = [
    {
        "name": "concise_assistant",
        "system_prompt": (
            "You are a concise assistant. Answer questions in 1-2 "
            "sentences maximum. Be direct and brief."
        ),
        "temperature": 0.3,
    },
    {
        "name": "detailed_assistant",
        "system_prompt": (
            "You are a thorough assistant. Provide detailed, "
            "comprehensive answers with examples when helpful."
        ),
        "temperature": 0.7,
    },
]

TEST_PROMPTS = [
    "What is machine learning?",
    "Explain the concept of overfitting.",
    "What is the purpose of cross-validation?",
    "How does gradient descent work?",
]


class LLMAssistant(mlflow.pyfunc.PythonModel):
    """An LLM assistant with a configurable system prompt and temperature."""

    def __init__(self, system_prompt: str, temperature: float = 0.7):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def predict(self, context, model_input, params=None):
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:1234/v1", api_key="lm-studio"
        )
        questions = (
            model_input["question"].tolist()
            if isinstance(model_input, pd.DataFrame)
            else [str(model_input)]
        )
        results = []
        for question in questions:
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
            results.append(resp.choices[0].message.content)
        return results


def build_and_register(llm_client: OpenAI) -> list[dict]:
    """Build two model variants, log them, and register as versions."""
    print("=" * 60)
    print("Part 1: Defining and logging two LLM configurations")
    print("=" * 60)

    for cfg in MODEL_CONFIGS:
        print(f"  {cfg['name']}:")
        print(f"    temperature:   {cfg['temperature']}")
        print(f"    system_prompt: {cfg['system_prompt'][:60]}...")
    print()

    sample_input = pd.DataFrame({"question": ["What is AI?"]})
    results = []

    for cfg in MODEL_CONFIGS:
        model = LLMAssistant(cfg["system_prompt"], cfg["temperature"])

        # Get a sample output for signature inference
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": "What is AI?"},
            ],
            temperature=cfg["temperature"],
            max_tokens=1024,
        )
        signature = infer_signature(sample_input, [resp.choices[0].message.content])

        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_param("config_name", cfg["name"])
            mlflow.log_param("temperature", cfg["temperature"])
            mlflow.log_param("system_prompt", cfg["system_prompt"])
            mlflow.pyfunc.log_model(
                name="model",
                python_model=model,
                signature=signature,
                input_example=sample_input,
            )
            results.append({
                "name": cfg["name"],
                "run_id": run.info.run_id,
                "system_prompt": cfg["system_prompt"],
                "temperature": cfg["temperature"],
            })
            print(f"  Logged: {cfg['name']} (run {run.info.run_id[:8]}...)")

    # Register both as versions of the same model
    print()
    print("=" * 60)
    print("Part 2: Registering models in the Model Registry")
    print("=" * 60)

    for entry in results:
        model_uri = f"runs:/{entry['run_id']}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        entry["version"] = mv.version
        print(f"  Registered {entry['name']} as {MODEL_NAME} version {mv.version}")
    print()

    return results


def evaluate_models(llm_client: OpenAI, results: list[dict]) -> list[dict]:
    """Evaluate both models on test prompts."""
    print("=" * 60)
    print("Part 3: Evaluating both models on test prompts")
    print("=" * 60)

    for entry in results:
        total_len, total_lat = 0, 0.0
        print(f"\n  Evaluating: {entry['name']}")

        for prompt in TEST_PROMPTS:
            start = time.time()
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": entry["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=entry["temperature"],
                max_tokens=1024,
            )
            elapsed = time.time() - start
            text = resp.choices[0].message.content
            total_len += len(text)
            total_lat += elapsed
            print(f"    [{elapsed:.1f}s] {prompt[:40]:40s} -> {len(text)} chars")

        n = len(TEST_PROMPTS)
        avg_len = total_len / n
        avg_lat = total_lat / n
        quality = avg_len / (1.0 + avg_lat)
        entry.update(
            avg_response_length=avg_len,
            avg_latency=avg_lat,
            quality_score=quality,
        )

        # Log evaluation metrics back to the original run
        with mlflow.start_run(run_id=entry["run_id"]):
            mlflow.log_metrics({
                "eval_avg_response_length": avg_len,
                "eval_avg_latency": avg_lat,
                "eval_quality_score": quality,
            })

        print(f"    Summary: avg_length={avg_len:.0f}  "
              f"avg_latency={avg_lat:.2f}s  quality={quality:.1f}")
    print()
    return results


def promote_best(client: MlflowClient, results: list[dict]) -> None:
    """Promote the best model to champion, other to challenger."""
    print("=" * 60)
    print("Part 4: Promoting best to champion")
    print("=" * 60)

    ranked = sorted(results, key=lambda r: r["quality_score"], reverse=True)
    champion, challenger = ranked[0], ranked[1]

    client.set_registered_model_alias(MODEL_NAME, "champion", champion["version"])
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger["version"])

    client.update_registered_model(
        MODEL_NAME,
        description="LLM assistant with versioned configurations, "
        "managed in L1-M3 Registry Workflows lesson.",
    )

    for entry in results:
        role = "champion" if entry is champion else "challenger"
        desc = (
            f"{entry['name']} | quality={entry['quality_score']:.1f} | "
            f"avg_latency={entry['avg_latency']:.2f}s | role={role}"
        )
        client.update_model_version(MODEL_NAME, entry["version"], description=desc)
        client.set_model_version_tag(MODEL_NAME, entry["version"], "role", role)
        client.set_model_version_tag(
            MODEL_NAME, entry["version"], "temperature", str(entry["temperature"])
        )

    print(f"  champion   -> v{champion['version']} "
          f"({champion['name']}, quality={champion['quality_score']:.1f})")
    print(f"  challenger -> v{challenger['version']} "
          f"({challenger['name']}, quality={challenger['quality_score']:.1f})")
    print()


def serve_champion() -> None:
    """Load champion by alias and demonstrate serving."""
    print("=" * 60)
    print("Part 5: Loading champion model by alias")
    print("=" * 60)

    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    print(f"  Loaded: {champion_uri}")

    test_df = pd.DataFrame({"question": [
        "What is reinforcement learning?",
        "Why is data preprocessing important?",
    ]})
    predictions = champion_model.predict(test_df)

    for i, (q, a) in enumerate(zip(test_df["question"], predictions)):
        print(f"  Q{i + 1}: {q}")
        print(f"  A{i + 1}: {a[:100].replace(chr(10), ' ')}...")
    print()

    # Also load challenger for comparison
    challenger_uri = f"models:/{MODEL_NAME}@challenger"
    challenger_model = mlflow.pyfunc.load_model(challenger_uri)
    challenger_answer = challenger_model.predict(
        pd.DataFrame({"question": ["What is reinforcement learning?"]})
    )
    print(f"  Challenger answer: {challenger_answer[0][:100].replace(chr(10), ' ')}...")
    print()


def comparison_summary(client: MlflowClient, results: list[dict]) -> None:
    """Print comparison table of all registered versions."""
    print("=" * 60)
    print("Part 6: Lifecycle Summary")
    print("=" * 60)

    rows = []
    for entry in results:
        mv = client.get_model_version(MODEL_NAME, entry["version"])
        aliases = mv.aliases if hasattr(mv, "aliases") else []
        rows.append({
            "Version": f"v{entry['version']}",
            "Style": entry["name"],
            "Avg Length": f"{entry['avg_response_length']:.0f}",
            "Avg Latency": f"{entry['avg_latency']:.2f}s",
            "Quality": f"{entry['quality_score']:.1f}",
            "Alias": ", ".join(aliases) if aliases else "-",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def main() -> None:
    mlflow_client = MlflowClient()
    llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    results = build_and_register(llm_client)
    results = evaluate_models(llm_client, results)
    promote_best(mlflow_client, results)
    serve_champion()
    comparison_summary(mlflow_client, results)

    print("=" * 60)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M3_models_registry/3_registry_workflows")
    main()
