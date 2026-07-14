"""
L1-M2.3 — PyFunc: The Universal Model Wrapper

Learn how mlflow.pyfunc.PythonModel lets you wrap ANY Python logic
— including LLM calls — into a standard MLflow model that can be
logged, versioned, loaded, and served with a uniform predict() API.
"""

import mlflow
from mlflow.models import infer_signature
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M2_models_registry/3_pyfunc")


class PromptTemplateModel(mlflow.pyfunc.PythonModel):
    """A custom PyFunc model that applies a prompt template and calls LMStudio."""

    def __init__(self, template: str, model_name: str = "google/gemma-4-e4b"):
        self.template = template
        self.model_name = model_name

    def predict(
        self, context, model_input: pd.DataFrame, params=None
    ) -> list[str]:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        results = []
        for _, row in model_input.iterrows():
            prompt = self.template.format(**row.to_dict())
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            results.append(response.choices[0].message.content)
        return results


def main() -> None:
    print("=" * 60)
    print("L1-M2.3 — PyFunc: The Universal Model Wrapper")
    print("=" * 60)

    print("""
WHAT IS PYFUNC?
  mlflow.pyfunc is MLflow's universal model interface.  Every MLflow
  model — sklearn, pytorch, transformers, langchain — exposes a
  'python_function' flavor so it can be loaded and called the same way:

      model = mlflow.pyfunc.load_model(uri)
      predictions = model.predict(data)

  By subclassing mlflow.pyfunc.PythonModel you can wrap ANY custom
  logic (API calls, LLM inference, rule engines, ensembles) into
  this standard interface.

WHY IT MATTERS
  - One API to load and serve any model, regardless of framework
  - Works with mlflow models serve for instant REST endpoints
  - Works with mlflow.evaluate() for automated evaluation
  - Version and register custom models in the Model Registry
""")

    # ------------------------------------------------------------------
    # Step 1 — Define the prompt template
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Defining a prompt-template PyFunc model")
    print("=" * 60)

    template = "Explain {topic} in one sentence for a {audience}."
    model = PromptTemplateModel(template=template)

    print(f"  Template : {template}")
    print(f"  LLM      : {model.model_name}")

    # ------------------------------------------------------------------
    # Step 2 — Log the custom model to MLflow
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Logging the PyFunc model to MLflow")
    print("=" * 60)

    test_input = pd.DataFrame({
        "topic": ["gravity", "photosynthesis"],
        "audience": ["five-year-old", "college student"],
    })

    # Run a quick predict to capture the signature
    sample_output = model.predict(context=None, model_input=test_input)
    signature = infer_signature(test_input, sample_output)

    with mlflow.start_run(run_name="prompt_template_pyfunc") as run:
        mlflow.log_param("template", template)
        mlflow.log_param("llm_model", model.model_name)

        mlflow.pyfunc.log_model(
            name="prompt_model",
            python_model=model,
            signature=signature,
            input_example=test_input,
        )
        run_id = run.info.run_id
        print(f"  Model logged. Run ID: {run_id}")
        print(f"  Signature:\n{signature}\n")

    # ------------------------------------------------------------------
    # Step 3 — Load and predict with the saved model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Loading model back and running predictions")
    print("=" * 60)

    model_uri = f"runs:/{run_id}/prompt_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"  Loaded model from: {model_uri}")

    new_input = pd.DataFrame({
        "topic": ["black holes", "recursion", "democracy"],
        "audience": ["teenager", "beginner programmer", "ten-year-old"],
    })

    print(f"\n  Inputs:")
    for _, row in new_input.iterrows():
        print(f"    topic={row['topic']!r}, audience={row['audience']!r}")

    predictions = loaded_model.predict(new_input)

    print(f"\n  LLM Outputs:")
    for i, pred in enumerate(predictions):
        print(f"    [{i+1}] {pred.strip()}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done!  Check the MLflow UI at http://127.0.0.1:5000")
    print("Navigate to the run to see the logged PyFunc model,")
    print("its signature, and input example.")
    print("=" * 60)


if __name__ == "__main__":
    main()
