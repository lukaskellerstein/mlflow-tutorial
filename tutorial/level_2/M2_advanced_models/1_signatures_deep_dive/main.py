"""
L2-M2.1 — Model Signatures Deep Dive

Explore MLflow model signatures for LLM applications: chat completion
signatures, structured output signatures, and signatures with
inference-time parameters.
"""

import json

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from openai import OpenAI

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "google/gemma-4-e4b"


def _make_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class ChatModel(mlflow.pyfunc.PythonModel):
    """A simple chat completion model."""
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        results = []
        for _, row in model_input.iterrows():
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[{"role": "user", "content": row["question"]}],
                temperature=0.7,
            )
            results.append(resp.choices[0].message.content)
        return results


class StructuredOutputModel(mlflow.pyfunc.PythonModel):
    """Model that returns structured JSON output."""
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        system_msg = (
            "Return a JSON object with keys: summary (string), "
            "key_points (list of strings), confidence (float 0-1). "
            "Return ONLY the JSON, no other text."
        )
        results = []
        for _, row in model_input.iterrows():
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": row["text"]},
                ],
                temperature=0.3,
            )
            results.append(resp.choices[0].message.content)
        return results


class ConfigurableChatModel(mlflow.pyfunc.PythonModel):
    """Chat model with runtime-configurable parameters."""
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        params = params or {}
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 256)
        results = []
        for _, row in model_input.iterrows():
            resp = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[{"role": "user", "content": row["question"]}],
                temperature=temperature, max_tokens=max_tokens,
            )
            results.append(resp.choices[0].message.content)
        return results


def part1_chat_signature() -> str:
    """Demo 1: Infer a signature from chat-style input/output."""
    print("=" * 60)
    print("Part 1: Chat Completion Signature")
    print("=" * 60)
    input_df = pd.DataFrame({"question": ["What is MLflow?"]})
    output = ["MLflow is an open-source platform for managing ML lifecycles."]
    signature = infer_signature(input_df, output)
    print(f"\n  Inferred signature:\n{signature}\n")
    print(f"  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")
    with mlflow.start_run(run_name="part1_chat_signature"):
        mlflow.log_param("signature_type", "chat completion")
        info = mlflow.pyfunc.log_model(
            name="chat_model", python_model=ChatModel(),
            signature=signature, input_example=input_df,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model. Run: {run_id}")
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        result = loaded.predict(pd.DataFrame({"question": ["What is experiment tracking?"]}))
        print(f"  Test prediction: {result[0][:120]}...")
    return run_id


def part2_structured_output() -> str:
    """Demo 2: Manual signature for structured JSON output."""
    print("\n" + "=" * 60)
    print("Part 2: Structured Output Signature")
    print("=" * 60)
    input_schema = Schema([ColSpec(DataType.string, "text")])
    output_schema = Schema([ColSpec(DataType.string, "json_output")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    print(f"\n  Manual signature:\n{signature}\n")
    for label, schema in [("Input", input_schema), ("Output", output_schema)]:
        for col in schema.inputs:
            print(f"  {label} column: {col.name} ({col.type})")
    print(f"\n  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")
    with mlflow.start_run(run_name="part2_structured_output"):
        mlflow.log_param("signature_type", "structured JSON output")
        info = mlflow.pyfunc.log_model(
            name="structured_model", python_model=StructuredOutputModel(),
            signature=signature,
            input_example=pd.DataFrame({"text": ["Explain gradient descent."]}),
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model. Run: {run_id}")
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        result = loaded.predict(pd.DataFrame({"text": ["Explain neural networks."]}))
        print(f"  Test prediction: {result[0][:120]}...")
    return run_id


def part3_params_signature() -> str:
    """Demo 3: ParamSpec for runtime configuration."""
    print("\n" + "=" * 60)
    print("Part 3: Signature with Inference Params")
    print("=" * 60)
    input_schema = Schema([ColSpec(DataType.string, "question")])
    output_schema = Schema([ColSpec(DataType.string, "answer")])
    param_schema = ParamSchema([
        ParamSpec("temperature", DataType.double, default=0.7),
        ParamSpec("max_tokens", DataType.long, default=256),
    ])
    signature = ModelSignature(
        inputs=input_schema, outputs=output_schema, params=param_schema,
    )
    print(f"\n  Signature with params:\n{signature}\n")
    for p in param_schema.params:
        print(f"  Param: {p.name} ({p.dtype}, default={p.default})")
    print(f"\n  Signature JSON:\n{json.dumps(signature.to_dict(), indent=2)}\n")
    with mlflow.start_run(run_name="part3_params_signature"):
        mlflow.log_param("signature_type", "with inference params")
        info = mlflow.pyfunc.log_model(
            name="configurable_chat_model", python_model=ConfigurableChatModel(),
            signature=signature,
            input_example=pd.DataFrame({"question": ["What is MLflow?"]}),
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  Logged model. Run: {run_id}")
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        test_df = pd.DataFrame({"question": ["Explain model signatures."]})
        result = loaded.predict(test_df, params={"temperature": 0.2, "max_tokens": 64})
        print(f"  Test (temp=0.2, max_tokens=64): {result[0][:120]}...")
    return run_id


def part4_signature_enforcement(chat_run_id: str) -> None:
    """Demo 4: Test how MLflow enforces signatures at prediction time."""
    print("\n" + "=" * 60)
    print("Part 4: Signature Enforcement")
    print("=" * 60)
    loaded = mlflow.pyfunc.load_model(f"runs:/{chat_run_id}/chat_model")
    print(f"\n  Loaded model signature:\n{loaded.metadata.signature}\n")

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
    print("=" * 60)
    print("L2-M2.1 -- Model Signatures Deep Dive")
    print("=" * 60)
    print()
    chat_run_id = part1_chat_signature()
    part2_structured_output()
    part3_params_signature()
    part4_signature_enforcement(chat_run_id)
    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("Explore each run's model artifacts to see the signatures.")
    print("=" * 60)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L2/M2_advanced_models/1_signatures_deep_dive")
    main()
