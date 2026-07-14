---
globs: ["tutorial/**/*.py"]
---

# MLFlow Patterns and APIs

## Core APIs by Level

### Level 1 — Essentials

#### Experiment Tracking (L1-M1)
- `mlflow.set_experiment()` — always set before logging
- `mlflow.start_run()` — context manager for runs
- `mlflow.log_param()` / `mlflow.log_params()` — log configuration
- `mlflow.log_metric()` / `mlflow.log_metrics()` — log numeric results
- `mlflow.log_artifact()` / `mlflow.log_artifacts()` — log files
- `mlflow.set_tag()` / `mlflow.set_tags()` — add metadata
- `mlflow.search_runs()` — query and filter runs
- `mlflow.search_experiments()` — query experiments
- `mlflow.enable_system_metrics_logging()` — CPU/memory/GPU tracking

#### Models and Registry (L1-M2)
- `mlflow.<flavor>.log_model()` — save models (sklearn, pytorch, pyfunc, langchain, etc.)
- `mlflow.<flavor>.load_model()` — load for inference
- `mlflow.models.infer_signature()` — infer input/output schema
- `mlflow.register_model()` — register in model registry
- `mlflow.pyfunc.PythonModel` — custom model wrapper

#### Autologging (L1-M3)
- `mlflow.autolog()` — universal autolog
- `mlflow.sklearn.autolog()` — scikit-learn
- `mlflow.xgboost.autolog()` — XGBoost
- `mlflow.langchain.autolog()` — LangChain and LangGraph
- `mlflow.openai.autolog()` — OpenAI
- `mlflow.anthropic.autolog()` — Anthropic/Claude
- `mlflow.transformers.autolog()` — Hugging Face

#### Evaluation (L1-M4)
- `mlflow.evaluate()` — the main evaluation entry point
  - `model_type`: `"question-answering"`, `"text-summarization"`, `"text"`
  - `evaluators`: `"default"` or custom
  - Pass `data` as pandas DataFrame
- Built-in LLM metrics: `toxicity`, `flesch_kincaid_grade_level`, `token_count`
- GenAI metrics: `answer_similarity`, `answer_correctness`, `faithfulness`, `relevance`

#### Tracing (L1-M5)
- `@mlflow.trace` — decorator for function-level tracing
- `mlflow.start_span()` — manual span creation (context manager)
- Auto-tracing via `mlflow.langchain.autolog()`

#### GenAI Features (L1-M6)
- `mlflow.genai.register_prompt()` — prompt registry
- `mlflow.genai.scorers` — built-in and custom scorers
- `mlflow.genai.judges` — LLM-as-judge
- `mlflow.genai.datasets` — evaluation dataset management

#### Data (L1-M7)
- `mlflow.data` — dataset logging
- `mlflow.log_input()` — link datasets to runs

#### Deployment (L1-M8)
- `mlflow models serve` — local REST API
- `mlflow models predict` — batch prediction
- AI Gateway route configuration

### Level 2 — Practitioner

#### Advanced Tracking (L2-M1)
- Nested runs with `nested=True`
- `mlflow.config.enable_async_logging()` — async logging
- `mlflow.log_image()`, `mlflow.log_table()`, `mlflow.log_figure()` — rich artifacts
- `MlflowClient` — programmatic CRUD

#### Advanced Models (L2-M2)
- `ModelSignature` — custom input/output schemas
- `PythonModel.load_context()` — load dependencies
- Model lifecycle: aliases (`champion`, `challenger`), stage transitions

#### Deep Evaluation (L2-M3)
- `make_metric()` — custom metric functions
- RAG metrics: `faithfulness`, `relevance`, `context_recall`
- `mlflow.genai.evaluation` — full GenAI evaluation framework
- `mlflow.genai.labeling` — human-in-the-loop
- `mlflow.log_assessment()` — human feedback

#### Advanced Tracing (L2-M4)
- LangGraph state transition tracing
- Temporal.io workflow/activity tracing
- OpenTelemetry export
- Trace-based latency/cost analysis

#### Prompt Engineering (L2-M6)
- Prompt A/B testing workflows
- `mlflow.genai.optimize` — automated prompt tuning

### Level 3 — Expert

#### Agent Evaluation (L3-M1)
- `mlflow.genai.agent_tester` — automated agent test generation
- `mlflow.genai.simulators` — conversation simulation
- Custom scorer implementation for agent-specific metrics
- Architecture comparison methodology

#### Custom Integrations (L3-M2)
- Custom autolog implementations
- MLflow plugin development
- Cross-framework tracing

#### Production (L3-M3)
- Trace sampling strategies
- Prometheus/Grafana export
- CI/CD quality gates
- Drift detection

## Patterns to Follow

### Experiment Naming
Use hierarchical names matching the level/module structure:
```python
mlflow.set_experiment("L1/M1_tracking/2_tracking_basics")
```

### Run Naming
Give runs descriptive names:
```python
with mlflow.start_run(run_name="temperature_0.7_gemma4"):
```

### Nested Runs for Comparisons
Use nested runs when comparing configurations:
```python
with mlflow.start_run(run_name="temperature_comparison"):
    for temp in [0.3, 0.7, 1.0]:
        with mlflow.start_run(run_name=f"temp_{temp}", nested=True):
            mlflow.log_param("temperature", temp)
```

### Always Check the MLFlow Source Code
When implementing MLFlow features, verify the API exists and its signature by consulting:
- Source code: `~/Projects/github/mlflow/mlflow`
- Documentation: `/Users/lkellers/Projects/github/mlflow/mlflow/docs/docs`

Do NOT guess at API names or parameters. Read the source if unsure.
