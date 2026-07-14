"""
L1-M2.1 -- MLflow Models and Flavors

Learn how MLflow packages models, what flavors are, how signatures
document the expected input/output schema, and how input examples
make models self-documenting -- demonstrated with two flavors:
pyfunc (wrapping a custom LLM call) and openai (declarative config).
"""

import os

import mlflow
import pandas as pd
from mlflow.models import infer_signature

os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/1_models_flavors")

MODEL = "google/gemma-4-e4b"


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
  Named interfaces through which a model can be saved and loaded.
  Every model gets the generic 'python_function' (pyfunc) flavor.
  Framework-specific flavors provide native access:
    openai, langchain, sklearn, pytorch, transformers, pyfunc ...

WHY SIGNATURES?
  A ModelSignature records input/output schemas so MLflow can:
    - Validate data before inference
    - Generate REST API docs when serving
    - Display schema in the MLflow UI
""")

    # ------------------------------------------------------------------
    # Step 1 -- Log an LLM model with the PyFunc flavor
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Logging an LLM model with the PyFunc flavor")
    print("=" * 60)
    print("  The pyfunc flavor wraps ANY Python code as an MLflow model.")
    print("  Here we wrap a direct OpenAI SDK call to LMStudio.\n")

    class DirectLLMModel(mlflow.pyfunc.PythonModel):
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
                    max_tokens=1024,
                )
                answers.append(resp.choices[0].message.content)
            return answers

    pyfunc_input = pd.DataFrame({"question": ["What is MLflow?"]})
    pyfunc_output = DirectLLMModel().predict(context=None, model_input=pyfunc_input)
    pyfunc_signature = infer_signature(pyfunc_input, pyfunc_output)

    print(f"  Sample input:  {pyfunc_input['question'].iloc[0]}")
    print(f"  Sample output: {pyfunc_output[0][:100]}...")
    print(f"\n  Inferred signature:\n{pyfunc_signature}\n")

    with mlflow.start_run(run_name="pyfunc_llm_flavor") as run:
        mlflow.pyfunc.log_model(
            name="pyfunc_llm",
            python_model=DirectLLMModel(),
            signature=pyfunc_signature,
            input_example=pyfunc_input,
        )
        mlflow.log_param("flavor", "pyfunc")
        mlflow.log_param("model_name", MODEL)
        run_id_pyfunc = run.info.run_id
        print(f"  Model logged with 'python_function' flavor")
        print(f"  Run ID: {run_id_pyfunc}")

    # ------------------------------------------------------------------
    # Step 2 -- Load the PyFunc model and test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Loading PyFunc model and running inference")
    print("=" * 60)

    pyfunc_uri = f"runs:/{run_id_pyfunc}/pyfunc_llm"
    loaded_pyfunc = mlflow.pyfunc.load_model(pyfunc_uri)
    print(f"  Loaded from: {pyfunc_uri}")

    test_df = pd.DataFrame({"question": ["Explain MLflow in one sentence."]})
    pyfunc_result = loaded_pyfunc.predict(test_df)
    print(f"  Input:  {test_df['question'].iloc[0]}")
    print(f"  Output: {pyfunc_result[0][:120]}...")

    # ------------------------------------------------------------------
    # Step 3 -- Log an LLM model with the OpenAI flavor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Logging an LLM with the openai flavor")
    print("=" * 60)
    print("  The openai flavor logs a model declaratively -- just specify")
    print("  the model name, task, and message template.\n")

    with mlflow.start_run(run_name="openai_llm_flavor") as run:
        model_info = mlflow.openai.log_model(
            model=MODEL,
            task="chat.completions",
            name="openai_llm",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
                {"role": "user", "content": "{question}"},
            ],
        )
        mlflow.log_param("flavor", "openai")
        mlflow.log_param("model_name", MODEL)
        run_id_openai = run.info.run_id
        print(f"  Model logged with 'openai' flavor")
        print(f"  Run ID: {run_id_openai}")

    # ------------------------------------------------------------------
    # Step 4 -- Load the OpenAI model natively and test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Loading OpenAI model natively")
    print("=" * 60)

    openai_uri = f"runs:/{run_id_openai}/openai_llm"
    loaded_raw = mlflow.openai.load_model(openai_uri)
    print(f"  Loaded from: {openai_uri}")
    print(f"  Raw model config: {loaded_raw}")
    print("  (Native load returns the saved config as a dict)")

    # ------------------------------------------------------------------
    # Step 5 -- Load OpenAI model via the generic pyfunc interface
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Loading OpenAI model via the pyfunc interface")
    print("=" * 60)
    print("  Every MLflow model can be loaded through the generic pyfunc")
    print("  interface, regardless of its original flavor.\n")

    pyfunc_openai = mlflow.pyfunc.load_model(openai_uri)
    test_df = pd.DataFrame({"question": ["What is an API?"]})
    pyfunc_openai_result = pyfunc_openai.predict(test_df)
    print(f"  Loaded OpenAI model via mlflow.pyfunc.load_model()")
    print(f"  Input:  {test_df['question'].iloc[0]}")
    print(f"  Output: {pyfunc_openai_result[0][:120]}...")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("Compare the two runs to see different model flavors:")
    print(f"  - pyfunc_llm_flavor:  uses the 'python_function' flavor")
    print(f"  - openai_llm_flavor:  uses the 'openai' flavor")
    print("Inspect each run's Artifacts tab to see the MLmodel file,")
    print("signature, and input example.")
    print()
    print("KEY INSIGHT: Both models can be loaded via mlflow.pyfunc.load_model()")
    print("because every flavor includes 'python_function' as a base flavor.")
    print("=" * 60)


if __name__ == "__main__":
    main()
