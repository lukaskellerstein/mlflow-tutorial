"""
L1-M3.1 -- Models, Flavors, and Signatures

Combines model flavors and signature patterns in one lesson:
- Part 1: PyFunc flavor -- wrap a custom LLM call as an MLflow model
- Part 2: OpenAI flavor -- declarative model logging
- Part 3: Inferred signatures from sample data
- Part 4: Manual signatures with Schema/ColSpec
- Part 5: Signatures with inference-time params (ParamSpec)
- Part 6: Signature enforcement at prediction time
"""

import json
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"

MODEL = "google/gemma-4-e4b"


# ------------------------------------------------------------------- #
# Shared model class
# ------------------------------------------------------------------- #


class LLMModel(mlflow.pyfunc.PythonModel):
    """Wraps a direct LLM call as a reusable MLflow model."""

    def predict(self, context, model_input, params=None):
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        temperature = (params or {}).get("temperature", 0.7)
        max_tokens = (params or {}).get("max_tokens", 1024)
        questions = model_input["question"].tolist()
        answers = []
        for q in questions:
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[{"role": "user", "content": q}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answers.append(resp.choices[0].message.content or "")
        return answers


def part1_pyfunc_flavor() -> str:
    """Log an LLM model with the PyFunc flavor."""
    print("=" * 60)
    print("Part 1: Logging an LLM model with the PyFunc flavor")
    print("=" * 60)
    print("  The pyfunc flavor wraps ANY Python code as an MLflow model.")
    print("  Here we wrap a direct OpenAI SDK call to LMStudio.\n")

    input_df = pd.DataFrame({"question": ["What is MLflow?"]})
    output = LLMModel().predict(context=None, model_input=input_df)
    signature = infer_signature(input_df, output)

    print(f"  Sample input:  {input_df['question'].iloc[0]}")
    print(f"  Sample output: {output[0][:100]}...")
    print(f"  Inferred signature:\n{signature}\n")

    with mlflow.start_run(run_name="pyfunc_llm_flavor") as run:
        mlflow.pyfunc.log_model(
            name="pyfunc_llm",
            python_model=LLMModel(),
            signature=signature,
            input_example=input_df,
        )
        mlflow.log_param("flavor", "pyfunc")
        run_id = run.info.run_id
        print("  Model logged with 'python_function' flavor")
        print(f"  Run ID: {run_id}")

    # Load and test
    print("\n  Loading PyFunc model and running inference...")
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc_llm")
    test_df = pd.DataFrame({"question": ["Explain MLflow in one sentence."]})
    result = loaded.predict(test_df)
    print(f"  Input:  {test_df['question'].iloc[0]}")
    print(f"  Output: {result[0][:120]}...")
    print()
    return run_id


def part2_openai_flavor() -> str:
    """Log an LLM model with the OpenAI flavor (declarative)."""
    print("=" * 60)
    print("Part 2: Logging an LLM with the OpenAI flavor")
    print("=" * 60)
    print("  The openai flavor logs a model declaratively -- just specify")
    print("  the model name, task, and message template.\n")

    with mlflow.start_run(run_name="openai_llm_flavor") as run:
        mlflow.openai.log_model(
            model=MODEL,
            task="chat.completions",
            name="openai_llm",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
                {"role": "user", "content": "{question}"},
            ],
        )
        mlflow.log_param("flavor", "openai")
        run_id = run.info.run_id
        print("  Model logged with 'openai' flavor")
        print(f"  Run ID: {run_id}")

    # Load natively
    openai_uri = f"runs:/{run_id}/openai_llm"
    loaded_raw = mlflow.openai.load_model(openai_uri)
    print(f"  Native load returns config dict: {loaded_raw}")

    # Load via pyfunc interface
    pyfunc_model = mlflow.pyfunc.load_model(openai_uri)
    test_df = pd.DataFrame({"question": ["What is an API?"]})
    result = pyfunc_model.predict(test_df)
    print(f"  Pyfunc load + predict: {result[0][:120]}...")
    print()
    print("  KEY INSIGHT: Both flavors can be loaded via mlflow.pyfunc.load_model()")
    print("  because every flavor includes 'python_function' as a base flavor.")
    print()
    return run_id


def part3_inferred_signature() -> None:
    """Demonstrate infer_signature from sample data."""
    print("=" * 60)
    print("Part 3: Inferred Signatures (infer_signature)")
    print("=" * 60)

    input_df = pd.DataFrame({"question": ["What is MLflow?"]})
    output = ["MLflow is an open-source platform for managing ML lifecycles."]
    signature = infer_signature(input_df, output)

    print(f"  Inferred signature:\n{signature}\n")
    print(f"  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")
    print("  infer_signature() automatically captures column names,")
    print("  data types, and structure from sample data.\n")


def part4_manual_signature() -> str:
    """Build signatures manually with Schema, ColSpec, and ModelSignature."""
    print("=" * 60)
    print("Part 4: Manual Signatures (Schema / ColSpec)")
    print("=" * 60)

    input_schema = Schema([ColSpec(DataType.string, "text")])
    output_schema = Schema([ColSpec(DataType.string, "json_output")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    print(f"  Manual signature:\n{signature}\n")
    for label, schema in [("Input", input_schema), ("Output", output_schema)]:
        for col in schema.inputs:
            print(f"  {label} column: {col.name} ({col.type})")
    print(f"\n  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")
    print("  Manual signatures give precise control over column names")
    print("  and types -- useful for structured or non-standard outputs.\n")

    with mlflow.start_run(run_name="manual_signature_model") as run:
        mlflow.pyfunc.log_model(
            name="structured_model",
            python_model=LLMModel(),
            signature=signature,
            input_example=pd.DataFrame({"text": ["Explain gradient descent."]}),
        )
        run_id = run.info.run_id
    return run_id


def part5_param_signature() -> None:
    """Demonstrate ParamSpec for runtime-configurable parameters."""
    print("=" * 60)
    print("Part 5: Signatures with Inference Params (ParamSpec)")
    print("=" * 60)

    input_schema = Schema([ColSpec(DataType.string, "question")])
    output_schema = Schema([ColSpec(DataType.string, "answer")])
    param_schema = ParamSchema(
        [
            ParamSpec("temperature", DataType.double, default=0.7),
            ParamSpec("max_tokens", DataType.long, default=256),
        ]
    )
    signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
        params=param_schema,
    )

    print(f"  Signature with params:\n{signature}\n")
    for p in param_schema.params:
        print(f"  Param: {p.name} ({p.dtype}, default={p.default})")
    print()

    with mlflow.start_run(run_name="configurable_model") as run:
        info = mlflow.pyfunc.log_model(
            name="configurable_model",
            python_model=LLMModel(),
            signature=signature,
            input_example=pd.DataFrame({"question": ["What is MLflow?"]}),
        )
        run_id = run.info.run_id
        print(f"  Logged model with param schema. Run: {run_id}")

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    test_df = pd.DataFrame({"question": ["Explain model signatures briefly."]})
    result = loaded.predict(test_df, params={"temperature": 0.2, "max_tokens": 128})
    print(f"  Test (temp=0.2, max_tokens=128): {result[0][:120]}...")
    print()


def part6_enforcement(pyfunc_run_id: str) -> None:
    """Test how MLflow enforces signatures at prediction time."""
    print("=" * 60)
    print("Part 6: Signature Enforcement")
    print("=" * 60)

    loaded = mlflow.pyfunc.load_model(f"runs:/{pyfunc_run_id}/pyfunc_llm")
    print(f"  Loaded model signature:\n{loaded.metadata.signature}\n")

    # Correct input
    result = loaded.predict(pd.DataFrame({"question": ["What is MLflow?"]}))
    print(f"  Correct input: {result[0][:80]}...")

    # Wrong column name
    print("\n  Testing wrong column name ('query' instead of 'question')...")
    try:
        loaded.predict(pd.DataFrame({"query": ["What is MLflow?"]}))
        print("  Result: succeeded (schema allowed flexible columns)")
    except Exception as e:
        print(f"  Result: {type(e).__name__}: {e}")

    # Wrong data type
    print("\n  Testing wrong data type (integer instead of string)...")
    try:
        r = loaded.predict(pd.DataFrame({"question": [12345]}))
        print(f"  Result: succeeded (type coerced) - {r[0][:60]}...")
    except Exception as e:
        print(f"  Result: {type(e).__name__}: {e}")

    # Extra columns
    print("\n  Testing extra columns...")
    try:
        r = loaded.predict(pd.DataFrame({"question": ["Hi"], "extra": ["x"]}))
        print(f"  Result: succeeded (extra columns ignored) - {r[0][:60]}...")
    except Exception as e:
        print(f"  Result: {type(e).__name__}: {e}")
    print()


def main() -> None:
    pyfunc_run_id = part1_pyfunc_flavor()
    part2_openai_flavor()
    part3_inferred_signature()
    part4_manual_signature()
    part5_param_signature()
    part6_enforcement(pyfunc_run_id)

    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5555")
    print("Compare the runs to see different flavors and signatures.")
    print("Inspect each run's Artifacts tab to see the MLmodel file.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("L1/M3_models_registry/1_models_flavors_signatures")
    main()
