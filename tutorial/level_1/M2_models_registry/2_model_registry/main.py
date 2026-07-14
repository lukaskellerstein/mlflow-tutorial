"""
L1-M2.2 -- Model Registry

Register LLM model versions with different configurations, manage them
with aliases (champion / challenger), and load by alias for inference.
"""

import mlflow
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/2_model_registry")

MODEL_NAME = "L1-llm-assistant"
TEST_QUESTIONS = [
    "What is machine learning?",
    "Explain what an API is.",
]


def create_llm_agent(system_prompt: str, temperature: float):
    """Create a LangChain agent with the given configuration."""
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-4-e4b",
        temperature=temperature,
    )
    return create_agent(model=llm, system_prompt=system_prompt)


def run_agent_on_questions(agent, questions: list[str]) -> list[str]:
    """Run the agent on a list of questions, return answers."""
    answers = []
    for q in questions:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]}
        )
        answers.append(result["messages"][-1].content)
    return answers


def main() -> None:
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
    # Step 2 -- Train (run) and log both models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Running and logging both models")
    print("=" * 60)

    results: dict[str, dict] = {}
    for name, cfg in configs.items():
        agent = create_llm_agent(cfg["system_prompt"], cfg["temperature"])
        answers = run_agent_on_questions(agent, TEST_QUESTIONS)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("config_name", name)
            mlflow.log_param("temperature", cfg["temperature"])
            mlflow.log_param("system_prompt", cfg["system_prompt"])

            # Log response length as a proxy metric for comparison
            avg_len = sum(len(a) for a in answers) / len(answers)
            mlflow.log_metric("avg_response_length", avg_len)

            mlflow.langchain.log_model(lc_model=agent, name="model")

            results[name] = {
                "run_id": run.info.run_id,
                "avg_response_length": avg_len,
                "answers": answers,
            }
            print(f"  {name:25s}  avg_len={avg_len:.0f} chars"
                  f"  run_id={run.info.run_id}")

        # Show sample answers
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

    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
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

    # Pick the detailed assistant as champion (more thorough answers)
    champion_ver = versions["detailed_assistant"]
    challenger_ver = versions["concise_assistant"]

    client.set_registered_model_alias(MODEL_NAME, "champion", champion_ver)
    client.set_registered_model_alias(
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

    client.update_registered_model(
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
        client.update_model_version(MODEL_NAME, version, description=desc)
        client.set_model_version_tag(
            MODEL_NAME, version, "config", name
        )
        client.set_model_version_tag(
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
    champion_model = mlflow.langchain.load_model(champion_uri)
    print(f"  Loaded model: {champion_uri}")

    test_q = "What is an LLM?"
    champion_result = champion_model.invoke(
        {"messages": [{"role": "user", "content": test_q}]}
    )
    champion_answer = champion_result["messages"][-1].content

    print(f"  Question: {test_q}")
    print(f"  Champion answer: {champion_answer[:200]}...")
    print()

    # Also load challenger for comparison
    challenger_uri = f"models:/{MODEL_NAME}@challenger"
    challenger_model = mlflow.langchain.load_model(challenger_uri)
    challenger_result = challenger_model.invoke(
        {"messages": [{"role": "user", "content": test_q}]}
    )
    challenger_answer = challenger_result["messages"][-1].content

    print(f"  Challenger answer: {challenger_answer[:200]}...")
    print()

    print("=" * 60)
    print("Done! View the Model Registry in the MLflow UI:")
    print(f"  http://127.0.0.1:5000/#/models/{MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
