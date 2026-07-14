"""
L1-M2.1 -- MLflow Models and Flavors

Learn how MLflow packages models, what flavors are, how signatures
document the expected input/output schema, and how input examples
make models self-documenting -- demonstrated with LLM models.
"""

import mlflow
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/1_models_flavors")


# Define tools at module level so they serialize cleanly with the agent
@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)


@tool
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


def main() -> None:
    print("=" * 60)
    print("L1-M2.1 -- MLflow Models and Flavors")
    print("=" * 60)
    print("""
WHAT IS AN MLFLOW MODEL?
  A standard directory containing:
    MLmodel            YAML manifest listing available flavors
    model artifacts     serialized model (pickle, weights, etc.)
    conda.yaml         Conda environment specification
    requirements.txt   pip dependencies
    input_example.json (optional) sample input for documentation

WHAT ARE FLAVORS?
  Named interfaces through which a model can be loaded.
  Every model gets the generic 'python_function' (pyfunc) flavor.
  Framework-specific flavors provide native access:
    langchain, openai, sklearn, pytorch, transformers, pyfunc ...

WHY SIGNATURES?
  A ModelSignature records input/output schemas so MLflow can:
    - Validate data before inference
    - Generate REST API docs when serving
    - Display schema in the MLflow UI
""")

    # ------------------------------------------------------------------
    # Step 1 -- Create a LangChain agent with tools
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Creating a LangChain agent with tools")
    print("=" * 60)

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="google/gemma-4-e4b",
        temperature=0.7,
    )

    agent = create_agent(
        model=llm,
        tools=[get_word_length, reverse_string],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )
    print("  Agent created with tools: get_word_length, reverse_string")
    print("  LLM: google/gemma-4-e4b via LMStudio")

    # ------------------------------------------------------------------
    # Step 2 -- Run the agent to generate sample I/O for signature
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Running agent to collect sample input/output")
    print("=" * 60)

    sample_input = {
        "messages": [
            {"role": "user", "content": "How many characters are in 'MLflow'?"}
        ]
    }
    result = agent.invoke(sample_input)

    # The last message in the result is the agent's final answer
    final_answer = result["messages"][-1].content
    print(f"  Input:  {sample_input['messages'][0]['content']}")
    print(f"  Output: {final_answer}")

    # ------------------------------------------------------------------
    # Step 3 -- Log with the langchain flavor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Logging agent with mlflow.langchain.log_model()")
    print("=" * 60)

    with mlflow.start_run(run_name="langchain_agent_flavor") as run:
        signature = infer_signature(sample_input, result)
        print(f"  Inferred signature:\n{signature}\n")

        mlflow.langchain.log_model(
            lc_model=agent,
            name="agent_model",
            signature=signature,
            input_example=sample_input,
        )
        mlflow.log_param("flavor", "langchain")
        mlflow.log_param("model_name", "google/gemma-4-e4b")
        mlflow.log_param("num_tools", 2)
        run_id = run.info.run_id
        print(f"  Model logged with 'langchain' flavor")
        print(f"  Run ID: {run_id}")

    # ------------------------------------------------------------------
    # Step 4 -- Load the langchain model and test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Loading langchain model and running inference")
    print("=" * 60)

    model_uri = f"runs:/{run_id}/agent_model"
    loaded_agent = mlflow.langchain.load_model(model_uri)
    print(f"  Loaded from: {model_uri}")

    test_input = {
        "messages": [
            {"role": "user", "content": "Reverse the word 'Python'"}
        ]
    }
    test_result = loaded_agent.invoke(test_input)
    test_answer = test_result["messages"][-1].content
    print(f"  Test input:  {test_input['messages'][0]['content']}")
    print(f"  Test output: {test_answer}")

    # ------------------------------------------------------------------
    # Step 5 -- Log a PyFunc-wrapped LLM for comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Logging a PyFunc-wrapped LLM model")
    print("=" * 60)
    print("  The pyfunc flavor wraps ANY Python code as an MLflow model.")
    print("  Useful when no native flavor exists for your framework.\n")

    class SimpleLLMModel(mlflow.pyfunc.PythonModel):
        """Wraps a raw LLM API call as a reusable MLflow model."""

        def predict(self, context, model_input, params=None):
            from openai import OpenAI

            client = OpenAI(
                base_url="http://localhost:1234/v1", api_key="lm-studio"
            )
            questions = model_input["question"].tolist()
            answers = []
            for q in questions:
                resp = client.chat.completions.create(
                    model="google/gemma-4-e4b",
                    messages=[{"role": "user", "content": q}],
                    temperature=0.7,
                    max_tokens=200,
                )
                answers.append(resp.choices[0].message.content)
            return answers

    pyfunc_input = pd.DataFrame({"question": ["What is MLflow?"]})
    pyfunc_signature = infer_signature(
        pyfunc_input, ["MLflow is an open-source platform..."]
    )

    with mlflow.start_run(run_name="pyfunc_llm_flavor") as run2:
        mlflow.pyfunc.log_model(
            name="pyfunc_llm",
            python_model=SimpleLLMModel(),
            signature=pyfunc_signature,
            input_example=pyfunc_input,
        )
        mlflow.log_param("flavor", "pyfunc")
        mlflow.log_param("model_name", "google/gemma-4-e4b")
        run_id2 = run2.info.run_id
        print(f"  PyFunc model logged")
        print(f"  Run ID: {run_id2}")
        print(f"  Signature:\n{pyfunc_signature}")

    # ------------------------------------------------------------------
    # Step 6 -- Load PyFunc model and test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Loading PyFunc model and running inference")
    print("=" * 60)

    pyfunc_uri = f"runs:/{run_id2}/pyfunc_llm"
    loaded_pyfunc = mlflow.pyfunc.load_model(pyfunc_uri)
    print(f"  Loaded from: {pyfunc_uri}")

    test_df = pd.DataFrame({"question": ["Explain MLflow in one sentence."]})
    pyfunc_result = loaded_pyfunc.predict(test_df)
    print(f"  Input:  {test_df['question'].iloc[0]}")
    print(f"  Output: {pyfunc_result[0][:120]}...")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("Compare the two runs to see different model flavors:")
    print("  - langchain_agent_flavor: uses the 'langchain' flavor")
    print("  - pyfunc_llm_flavor:      uses the 'python_function' flavor")
    print("Inspect each run's Artifacts tab to see the MLmodel file,")
    print("signature, and input example.")
    print("=" * 60)


if __name__ == "__main__":
    main()
