# L1-M8.1 -- Model Serving Basics

**Level:** Essentials
**Duration:** ~30 minutes

## Overview

MLflow can serve any logged model as a REST API with a single CLI command.
This lesson covers how to wrap an LLM call in a custom `PythonModel`,
log it with MLflow, test it locally, and serve it as a REST endpoint.
Any application -- in any language -- can then call your LLM over HTTP.

## Prerequisites

- Completed: L1-M2 (Models and Registry)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-e4b` loaded

## Concepts

### Why Model Serving?

Building an LLM application is only half the story. To integrate it into
a product you need to make it accessible. MLflow provides several approaches:

1. **Real-time serving** -- expose the model as a REST API so any language
   or service can call it over HTTP.
2. **Batch prediction** -- run the model against a file of inputs from the
   command line.
3. **Programmatic loading** -- load the model in Python for scripts and
   notebooks.

All approaches work with any MLflow model flavor (pyfunc, langchain, openai,
etc.) without writing any serving code.

### Serving Architecture

```
Client (curl / app)       MLflow Serving Process
   |                          |
   |  POST /invocations       |
   | -----------------------> |
   |                          |  Load PythonModel
   |                          |  Call model.predict(input)
   |                          |  (model calls LLM internally)
   |  JSON response           |
   | <----------------------- |
```

The serving process loads the model once at startup and handles requests.
For our LLM model, each prediction request triggers an API call to LMStudio.

### Endpoints

| Endpoint        | Method | Purpose                    |
|-----------------|--------|----------------------------|
| `/invocations`  | POST   | Run predictions            |
| `/ping`         | GET    | Health check (returns 200) |
| `/version`      | GET    | MLflow version info        |

### Input Formats

The `/invocations` endpoint accepts JSON in two formats:

**dataframe_split** (recommended):
```json
{
  "dataframe_split": {
    "columns": ["question"],
    "data": [["What is MLflow?"]]
  }
}
```

**instances**:
```json
{
  "instances": [{"question": "What is MLflow?"}]
}
```

## Step-by-Step

### Step 1: Create a PythonModel wrapper

We define a class that inherits from `mlflow.pyfunc.PythonModel` and wraps
an LLM API call. The `predict` method receives a DataFrame and returns
a list of answers.

```python
class LLMModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        questions = model_input["question"].tolist()
        answers = []
        for question in questions:
            response = client.chat.completions.create(
                model="google/gemma-4-e4b",
                messages=[{"role": "user", "content": question}],
            )
            answers.append(response.choices[0].message.content)
        return answers
```

### Step 2: Log and register the model

We log the model with a signature and input example, and register it for
easy access.

```python
mlflow.pyfunc.log_model(
    name="model",
    python_model=LLMModel(),
    signature=signature,
    input_example=input_example,
    registered_model_name="L1-llm-serving-demo",
    pip_requirements=["openai>=1.0", "mlflow>=2.0"],
)
```

### Step 3: Test locally

Before serving, verify the model works by loading it in-process.

```python
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
predictions = model.predict(pd.DataFrame({"question": ["What is AI?"]}))
```

### Step 4: Serve as REST API

Start the serving process with one command:

```bash
mlflow models serve -m "models:/L1-llm-serving-demo/1" --port 5001 --no-conda
```

### Step 5: Call the API

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["question"], "data": [["What is MLflow?"]]}}'
```

## Running the Lesson

```bash
cd tutorial/level_1/M8_deployment/1_model_serving
uv sync
uv run python main.py
```

Note: The script logs the model, tests it locally, and prints serving
commands. It does not start a server process. Follow the printed instructions
to try serving yourself.

## Expected Output

In the terminal you will see:
- The model signature showing "question" input and string output
- The model logged and registered as `L1-llm-serving-demo`
- Local test predictions for 3 questions
- CLI commands for serving, curl requests, batch prediction, and Docker

In the MLflow UI at http://127.0.0.1:5000:
- Experiment "L1/M8_deployment/1_model_serving" with one run
- The run contains the logged model with signature and input example

## Key Takeaways

- `mlflow.pyfunc.PythonModel` lets you wrap any Python code -- including
  LLM API calls -- as a servable MLflow model.
- `mlflow models serve` turns any logged model into a REST API with zero
  application code.
- Models need a **signature** and **input example** for reliable serving.
- The `/invocations` endpoint accepts JSON in `dataframe_split` or
  `instances` format.
- `pip_requirements` in `log_model()` ensures the serving environment has
  all needed packages.
- `mlflow.pyfunc.load_model()` is the in-process alternative when you
  don't need HTTP serving.

## Next Steps

Continue to L1-M8.2 (AI Gateway Overview) to learn how MLflow can route
requests across multiple LLM providers with rate limiting, fallbacks, and
unified API access. In Level 2, we will explore advanced serving patterns
including custom endpoints and deployment strategies.
