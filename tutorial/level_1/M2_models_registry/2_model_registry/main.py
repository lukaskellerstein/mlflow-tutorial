"""
L1-M2.2 -- Model Registry

Register LLM model versions with different configurations, manage them
with aliases (champion / challenger), and load by alias for inference.
"""

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from openai import OpenAI

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/2_model_registry")

LMSTUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "google/gemma-4-e4b"
MODEL_NAME = "L1-llm-assistant"
TEST_QUESTIONS = [
    "What is machine learning?",
    "Explain what an API is.",
]


class LLMAssistant(mlflow.pyfunc.PythonModel):
    """Wraps an LLM call with a configurable system prompt and temperature."""

    def __init__(self, system_prompt: str, temperature: float):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def predict(self, context, model_input, params=None):
        client = OpenAI(
            base_url="http://localhost:1234/v1", api_key="lm-studio"
        )
        questions = model_input["question"].tolist()
        answers = []
        for q in questions:
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q},
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
            answers.append(resp.choices[0].message.content)
        return answers


def main() -> None:
    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    # ------------------------------------------------------------------
    # Step 1 -- Define two LLM configurations
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Defining two LLM configurations")
    print("=" * 60)

    configs = {
        "concise_assistant": {
            "system_prompt": (
                "You are a concise assistant. Answer questions in 1-2 "
                "sentences maximum. Be direct and brief."
            ),
            "temperature": 0.3,
        },
        "detailed_assistant": {
            "system_prompt": (
                "You are a thorough assistant. Provide detailed, "
                "comprehensive answers with examples when helpful."
            ),
            "temperature": 0.7,
        },
    }

    for name, cfg in configs.items():
        print(f"  {name}:")
        print(f"    temperature:   {cfg['temperature']}")
        print(f"    system_prompt: {cfg['system_prompt'][:60]}...")
    print()

    # ------------------------------------------------------------------
    # Step 2 -- Run and log both models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Running and logging both models")
    print("=" * 60)

    input_df = pd.DataFrame({"question": TEST_QUESTIONS})
    results: dict[str, dict] = {}

    for name, cfg in configs.items():
        model = LLMAssistant(cfg["system_prompt"], cfg["temperature"])
        answers = model.predict(context=None, model_input=input_df)
        signature = infer_signature(input_df, answers)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("config_name", name)
            mlflow.log_param("temperature", cfg["temperature"])
            mlflow.log_param("system_prompt", cfg["system_prompt"])

            avg_len = sum(len(a) for a in answers) / len(answers)
            mlflow.log_metric("avg_response_length", avg_len)

            mlflow.pyfunc.log_model(
                name="model",
                python_model=model,
                signature=signature,
                input_example=input_df,
            )

            results[name] = {
                "run_id": run.info.run_id,
                "avg_response_length": avg_len,
                "answers": answers,
            }
            print(f"  {name:25s}  avg_len={avg_len:.0f} chars"
                  f"  run_id={run.info.run_id}")

        for q, a in zip(TEST_QUESTIONS, answers):
            print(f"    Q: {q}")
            print(f"    A: {a[:100]}...")
            print()

    # ------------------------------------------------------------------
    # Step 3 -- Register both as versions of the same model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Registering models in the Model Registry")
    print("=" * 60)

    versions: dict[str, str] = {}
    for name, info in results.items():
        model_uri = f"runs:/{info['run_id']}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        versions[name] = mv.version
        print(f"  Registered {name} as {MODEL_NAME} version {mv.version}")
    print()

    # ------------------------------------------------------------------
    # Step 4 -- List registered versions
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Listing registered model versions")
    print("=" * 60)

    mlflow_client = MlflowClient()
    all_versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    for mv in all_versions:
        print(f"  Version {mv.version}  |  run_id={mv.run_id}"
              f"  |  status={mv.status}")
    print()

    # ------------------------------------------------------------------
    # Step 5 -- Set aliases (champion / challenger)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 5: Setting aliases (champion / challenger)")
    print("=" * 60)

    champion_ver = versions["detailed_assistant"]
    challenger_ver = versions["concise_assistant"]

    mlflow_client.set_registered_model_alias(MODEL_NAME, "champion", champion_ver)
    mlflow_client.set_registered_model_alias(
        MODEL_NAME, "challenger", challenger_ver
    )
    print(f"  champion   -> v{champion_ver} (detailed_assistant)")
    print(f"  challenger -> v{challenger_ver} (concise_assistant)")
    print()

    # ------------------------------------------------------------------
    # Step 6 -- Add descriptions and tags
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 6: Adding descriptions and tags")
    print("=" * 60)

    mlflow_client.update_registered_model(
        MODEL_NAME,
        description="LLM assistant with versioned configurations, "
        "managed in L1-M2 Model Registry lesson.",
    )
    print(f"  Set model description for '{MODEL_NAME}'")

    for name, version in versions.items():
        cfg = configs[name]
        desc = (
            f"{name} (temp={cfg['temperature']}, "
            f"avg_len={results[name]['avg_response_length']:.0f})"
        )
        mlflow_client.update_model_version(MODEL_NAME, version, description=desc)
        mlflow_client.set_model_version_tag(
            MODEL_NAME, version, "config", name
        )
        mlflow_client.set_model_version_tag(
            MODEL_NAME, version, "temperature", str(cfg["temperature"])
        )
        print(f"  Version {version}: description and tags set")
    print()

    # ------------------------------------------------------------------
    # Step 7 -- Load champion by alias and run inference
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 7: Loading champion model by alias and predicting")
    print("=" * 60)

    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    print(f"  Loaded model: {champion_uri}")

    test_df = pd.DataFrame({"question": ["What is an LLM?"]})
    champion_answer = champion_model.predict(test_df)

    print(f"  Question:  {test_df['question'].iloc[0]}")
    print(f"  Champion:  {champion_answer[0][:200]}...")
    print()

    challenger_uri = f"models:/{MODEL_NAME}@challenger"
    challenger_model = mlflow.pyfunc.load_model(challenger_uri)
    challenger_answer = challenger_model.predict(test_df)

    print(f"  Challenger: {challenger_answer[0][:200]}...")
    print()

    print("=" * 60)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
