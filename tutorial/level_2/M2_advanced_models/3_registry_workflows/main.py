"""
L2-2.3 -- Model Registry Workflows

Full LLM model registry lifecycle: build two LLM model versions with
different system prompts, register them, evaluate on test prompts,
promote the best to champion, and demonstrate alias-based serving.
"""

import time

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from openai import OpenAI

MODEL_NAME = "L2-llm-assistant"
LLM_MODEL = "google/gemma-4-e4b"

MODEL_CONFIGS = [
    {"name": "precise_assistant",
     "system_prompt": "You are a precise, factual assistant. Give concise, "
                      "accurate answers with specific details. Avoid speculation."},
    {"name": "creative_assistant",
     "system_prompt": "You are a creative, engaging assistant. Use analogies, "
                      "examples, and vivid language to make answers memorable."},
]

TEST_PROMPTS = [
    "What is machine learning?",
    "Explain the concept of overfitting.",
    "What is the purpose of cross-validation?",
    "How does gradient descent work?",
    "What is transfer learning?",
]


class LLMAssistant(mlflow.pyfunc.PythonModel):
    """An LLM assistant with a configurable system prompt."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt

    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        temperature = (params or {}).get("temperature", 0.7)
        questions = (model_input["question"].tolist()
                     if isinstance(model_input, pd.DataFrame) else [str(model_input)])
        results = []
        for question in questions:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": question}],
                temperature=temperature,
            )
            results.append(resp.choices[0].message.content)
        return results


def build_and_log_models(llm: OpenAI) -> list[dict]:
    """Step 1-2: Build two LLM model versions and log them to MLflow."""
    results = []
    sample_input = pd.DataFrame({"question": ["What is AI?"]})
    for config in MODEL_CONFIGS:
        with mlflow.start_run(run_name=f"build_{config['name']}") as run:
            model = LLMAssistant(system_prompt=config["system_prompt"])
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": config["system_prompt"]},
                          {"role": "user", "content": "What is AI?"}],
                temperature=0.7,
            )
            signature = infer_signature(sample_input, [resp.choices[0].message.content])
            mlflow.log_param("model_style", config["name"])
            mlflow.log_param("system_prompt", config["system_prompt"])
            mlflow.pyfunc.log_model(name="model", python_model=model, signature=signature)
            results.append({"name": config["name"], "run_id": run.info.run_id,
                            "system_prompt": config["system_prompt"]})
            print(f"  Logged: {config['name']} (run {run.info.run_id[:8]}...)")
    return results


def register_models(results: list[dict]) -> list[dict]:
    """Step 3: Register both models as versions of MODEL_NAME."""
    for entry in results:
        mv = mlflow.register_model(f"runs:/{entry['run_id']}/model", MODEL_NAME)
        entry["version"] = mv.version
        print(f"  {entry['name']:25s} -> {MODEL_NAME} v{mv.version}")
    return results


def evaluate_models(llm: OpenAI, results: list[dict]) -> list[dict]:
    """Step 4: Evaluate both models on test prompts."""
    for entry in results:
        total_len, total_lat = 0, 0.0
        print(f"\n  Evaluating: {entry['name']}")
        for prompt in TEST_PROMPTS:
            start = time.time()
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": entry["system_prompt"]},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
            )
            elapsed = time.time() - start
            text = resp.choices[0].message.content
            total_len += len(text)
            total_lat += elapsed
            print(f"    [{elapsed:.1f}s] {prompt[:40]:40s} -> {len(text)} chars")

        n = len(TEST_PROMPTS)
        avg_len, avg_lat = total_len / n, total_lat / n
        quality = avg_len / (1.0 + avg_lat)
        entry.update(avg_response_length=avg_len, avg_latency=avg_lat, quality_score=quality)
        with mlflow.start_run(run_id=entry["run_id"]):
            mlflow.log_metrics({"eval_avg_response_length": avg_len,
                                "eval_avg_latency": avg_lat, "eval_quality_score": quality})
        print(f"    Summary: avg_length={avg_len:.0f}  "
              f"avg_latency={avg_lat:.2f}s  quality={quality:.1f}")
    return results


def promote_best(client: MlflowClient, results: list[dict]) -> None:
    """Step 5: Promote the best model to champion, other to challenger."""
    ranked = sorted(results, key=lambda r: r["quality_score"], reverse=True)
    champion, challenger = ranked[0], ranked[1]
    client.set_registered_model_alias(MODEL_NAME, "champion", champion["version"])
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger["version"])
    client.update_registered_model(
        MODEL_NAME, description="LLM assistant with configurable system prompt. "
        "Multiple prompt styles compared; best promoted to champion.")
    for entry in results:
        role = "champion" if entry is champion else "challenger"
        client.update_model_version(MODEL_NAME, entry["version"], description=(
            f"{entry['name']} | quality={entry['quality_score']:.1f} | "
            f"avg_latency={entry['avg_latency']:.2f}s | role={role}"))
        client.set_model_version_tag(MODEL_NAME, entry["version"], "role", role)
        client.set_model_version_tag(MODEL_NAME, entry["version"],
                                     "eval_quality_score", f"{entry['quality_score']:.1f}")
    print(f"  champion   -> v{champion['version']} "
          f"({champion['name']}, quality={champion['quality_score']:.1f})")
    print(f"  challenger -> v{challenger['version']} "
          f"({challenger['name']}, quality={challenger['quality_score']:.1f})")


def serve_champion() -> None:
    """Step 6: Load champion by alias and demonstrate serving."""
    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    test_df = pd.DataFrame({"question": [
        "What is reinforcement learning?",
        "Why is data preprocessing important?"]})
    predictions = champion_model.predict(test_df)
    print(f"  Loaded: {champion_uri}")
    for i, (q, a) in enumerate(zip(test_df["question"], predictions)):
        print(f"  Q{i+1}: {q}")
        print(f"  A{i+1}: {a[:100].replace(chr(10), ' ')}...")


def compare_versions(client: MlflowClient, results: list[dict]) -> None:
    """Print comparison table of all registered versions."""
    rows = []
    for entry in results:
        mv = client.get_model_version(MODEL_NAME, entry["version"])
        aliases = mv.aliases if hasattr(mv, "aliases") else []
        rows.append({"Version": f"v{entry['version']}", "Style": entry["name"],
                      "Avg Length": f"{entry['avg_response_length']:.0f}",
                      "Avg Latency": f"{entry['avg_latency']:.2f}s",
                      "Quality": f"{entry['quality_score']:.1f}",
                      "Alias": ", ".join(aliases) if aliases else "-"})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    """Execute the full LLM model registry workflow."""
    client = MlflowClient()
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    print("=" * 70)
    print("Step 1-2: Build two LLM model versions and log to MLflow")
    print("=" * 70)
    results = build_and_log_models(llm)
    print()
    print("=" * 70)
    print("Step 3: Register both as versions of", MODEL_NAME)
    print("=" * 70)
    results = register_models(results)
    print()
    print("=" * 70)
    print("Step 4: Evaluate both models on test prompts")
    print("=" * 70)
    results = evaluate_models(llm, results)
    print()
    print("=" * 70)
    print("Step 5: Promote best to champion, runner-up to challenger")
    print("=" * 70)
    promote_best(client, results)
    print()
    print("=" * 70)
    print("Step 6: Load champion by alias and demonstrate serving")
    print("=" * 70)
    serve_champion()
    print()
    print("=" * 70)
    print("Lifecycle Summary: All registered versions")
    print("=" * 70)
    compare_versions(client, results)
    print()
    print("=" * 70)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M2_advanced_models/3_registry_workflows")
    main()
