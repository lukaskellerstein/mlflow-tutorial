# MLFlow Tutorial: Three-Level Course

## Philosophy

This tutorial is structured in three progressive levels:

- **Level 1 — Essentials**: Breadth-first. Touch every major MLflow feature. Short lessons (~30 min). Goal: understand what MLflow can do and when to reach for each feature.
- **Level 2 — Practitioner**: Go deeper in each area with real-world scenarios. Longer lessons (~1-2 hours). Goal: build working projects and develop muscle memory.
- **Level 3 — Expert**: Production patterns, custom integrations, advanced evaluation, enterprise features. Goal: master MLflow for production AI systems, with special focus on agent evaluation.

Each level builds on the previous. A user can stop after Level 1 and have a working mental model of the entire platform, or continue through Level 3 for full mastery.

## Target Audience

- **Level 1**: Anyone starting with MLflow — data scientists, ML engineers, AI developers
- **Level 2**: Practitioners building real ML/AI applications who need depth
- **Level 3**: Teams shipping AI agents to production who need evaluation, monitoring, and custom integrations

## Technical Stack

- **Python**: 3.10+
- **Package Manager**: `uv` (every lesson is a standalone project)
- **MLFlow**: Latest (2.x+)
- **LLM (local)**: Gemini 4 2B quantized via Ollama (`gemma4:e2b`)
- **Agent Frameworks**: LangChain v1.0+, LangGraph, Claude Agent SDK, Codex SDK, DeepAgents
- **Traditional ML**: scikit-learn, XGBoost, PyTorch
- **Vector DB**: Chroma (RAG examples)
- **Workflow Orchestration**: Temporal.io (optional)
- **Observability**: Grafana (production monitoring)

## Reference Sources

- **MLFlow**
  - Source code: `~/Projects/github/mlflow/mlflow`
  - Documentation: `/Users/lkellers/Projects/github/mlflow/mlflow/docs/docs`
- **LangChain + LangGraph**:
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai`
- **Temporal.io**:
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python`
- **Claude Agent SDK**:
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
  - Source code: `~/Projects/github/anthropics/claude-agent-sdk-python`
- **Codex SDK**:
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`
  - Source code: `~/Projects/github/openai/codex/sdk`
- **Deep agents**:
  - Source code: `~/Projects/github/lanchain-ai/deepagents`

---
---

# LEVEL 1 — ESSENTIALS (Breadth)

*Goal: Touch every major MLflow feature. Understand the landscape.*
*Estimated time: ~12-15 hours*

---

## L1-M1: Core Platform

### L1-1.1 — What is MLflow? Architecture Overview
**Duration:** 20 min
**Topics:**
- MLflow's 5 pillars: Tracking, Models, Registry, Evaluation, Deployment
- Architecture: tracking server, backend store (SQLite/Postgres), artifact store (local/S3/GCS)
- Installing MLflow with `uv`
- Starting the tracking server and UI locally
- Key concepts: experiments, runs, parameters, metrics, artifacts, tags

**Deliverables:**
- Running MLflow server with UI at http://127.0.0.1:5000
- Diagram of MLflow architecture (in README)

---

### L1-1.2 — Experiment Tracking Basics
**Duration:** 30 min
**Topics:**
- Creating experiments with `mlflow.set_experiment()`
- Starting runs with `mlflow.start_run()`
- Logging parameters (`log_param`, `log_params`)
- Logging metrics (`log_metric`, `log_metrics`) — single values and step-based
- Logging artifacts (`log_artifact`, `log_artifacts`) — files, plots, configs
- Setting tags (`set_tag`, `set_tags`)
- Viewing results in MLflow UI

**Deliverables:**
- Script that logs a simple scikit-learn experiment with params, metrics, and a plot artifact

---

### L1-1.3 — Search and Query API
**Duration:** 20 min
**Topics:**
- `mlflow.search_runs()` — filtering and sorting runs
- Search syntax: `metrics.accuracy > 0.9 AND params.model = "rf"`
- `mlflow.search_experiments()`
- `MlflowClient` for programmatic access
- Exporting results to pandas DataFrames

**Deliverables:**
- Script that creates multiple runs, then queries and compares them programmatically

---

### L1-1.4 — System Metrics Logging
**Duration:** 15 min
**Topics:**
- Enabling system metrics: `mlflow.enable_system_metrics_logging()`
- What gets logged: CPU, memory, disk, GPU utilization
- Viewing system metrics in UI alongside model metrics
- Use cases: identifying resource bottlenecks during training

**Deliverables:**
- Training run with system metrics visible in MLflow UI

---

## L1-M2: Models and Registry

### L1-2.1 — MLflow Models and Flavors
**Duration:** 30 min
**Topics:**
- What is an MLflow Model? (the `MLmodel` file, flavors, signatures)
- Built-in flavors overview: `sklearn`, `pytorch`, `transformers`, `pyfunc`, `openai`, `langchain`, etc.
- Model signatures: `ModelSignature`, `infer_signature()`
- `mlflow.<flavor>.log_model()` and `mlflow.<flavor>.load_model()`
- Input examples for documentation

**Deliverables:**
- Log a scikit-learn model with signature and input example
- Load it back and run predictions

---

### L1-2.2 — Model Registry
**Duration:** 30 min
**Topics:**
- Registering models: `mlflow.register_model()`
- Model versions and aliases (`champion`, `challenger`)
- Model descriptions and tags
- Transitioning models through stages
- Loading models by name and version/alias

**Deliverables:**
- Register a model, create versions, set aliases, load by alias

---

### L1-2.3 — PyFunc — The Universal Model Wrapper
**Duration:** 30 min
**Topics:**
- What is PyFunc and why it matters (universal interface)
- `mlflow.pyfunc.log_model()` with custom `PythonModel` class
- `predict()` interface
- Wrapping arbitrary Python code as an MLflow model
- Use case: wrapping an LLM prompt template as a model

**Deliverables:**
- Custom PyFunc model that wraps a prompt template + LLM call

---

## L1-M3: Autologging

### L1-3.1 — Traditional ML Autologging
**Duration:** 30 min
**Topics:**
- `mlflow.autolog()` — the universal autolog
- Framework-specific: `mlflow.sklearn.autolog()`, `mlflow.xgboost.autolog()`, `mlflow.pytorch.autolog()`
- What gets auto-logged per framework: params, metrics, models, artifacts
- Supported frameworks overview: sklearn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, Statsmodels, Prophet, etc.
- Disabling/configuring autolog behavior

**Deliverables:**
- Side-by-side: sklearn + XGBoost autologging on the same dataset, comparing logged outputs

---

### L1-3.2 — LLM and GenAI Autologging
**Duration:** 30 min
**Topics:**
- `mlflow.openai.autolog()` — trace OpenAI calls
- `mlflow.anthropic.autolog()` — trace Anthropic/Claude calls
- `mlflow.langchain.autolog()` — trace LangChain and LangGraph
- `mlflow.transformers.autolog()` — Hugging Face models
- Other LLM integrations: Mistral, Gemini, Bedrock, Groq, LiteLLM
- What gets captured: inputs, outputs, latencies, token counts, model info

**Deliverables:**
- Script showing autologging for Ollama via LangChain, with traces visible in UI

---

## L1-M4: Evaluation

### L1-4.1 — Traditional ML Evaluation
**Duration:** 30 min
**Topics:**
- `mlflow.evaluate()` for classification and regression
- Built-in metrics: accuracy, precision, recall, F1, AUC, MAE, RMSE
- Evaluation artifacts: confusion matrix, ROC curve, lift curve
- Custom metrics with `make_metric()`
- Comparing models via evaluation results in UI

**Deliverables:**
- Evaluate a classifier with built-in metrics, view confusion matrix in UI

---

### L1-4.2 — LLM Evaluation Basics
**Duration:** 30 min
**Topics:**
- `mlflow.evaluate()` for LLMs — `model_type="question-answering"`, `"text-summarization"`, `"text"`
- Built-in LLM metrics: `toxicity`, `flesch_kincaid_grade_level`, `token_count`
- GenAI metrics: `answer_similarity`, `answer_correctness`, `faithfulness`
- Creating evaluation datasets with pandas DataFrames
- Interpreting evaluation results

**Deliverables:**
- Evaluate a Q&A system on a small dataset with both built-in and GenAI metrics

---

### L1-4.3 — LLM-as-Judge
**Duration:** 30 min
**Topics:**
- What is LLM-as-judge and why use it?
- Using judge metrics: `answer_correctness`, `faithfulness`, `relevance`
- Viewing judge justifications in MLflow UI
- Limitations and biases of LLM judges

**Deliverables:**
- Run LLM-as-judge evaluation, examine justifications in UI

---

## L1-M5: Tracing

### L1-5.1 — Automatic Tracing
**Duration:** 30 min
**Topics:**
- What is tracing? (vs. logging — structured execution flow)
- Auto-tracing with `mlflow.langchain.autolog()` — zero-code instrumentation
- Trace structure: spans, parent-child relationships, inputs/outputs
- Viewing traces in MLflow UI (Traces tab)
- Trace search and filtering

**Deliverables:**
- Multi-step LangChain chain with auto-traced execution visible in UI

---

### L1-5.2 — Manual Tracing
**Duration:** 30 min
**Topics:**
- `@mlflow.trace` decorator — function-level tracing
- `mlflow.start_span()` — manual span creation (context manager)
- Adding metadata to spans: `span.set_inputs()`, `span.set_outputs()`, `span.set_attributes()`
- Combining auto and manual tracing

**Deliverables:**
- Application with both auto-traced LLM calls and manual spans for business logic

---

## L1-M6: GenAI Features

### L1-6.1 — Prompt Registry
**Duration:** 20 min
**Topics:**
- Registering prompts: `mlflow.genai.register_prompt()`
- Prompt versioning and loading
- Prompt templates with variables
- Searching and managing prompts
- Use case: centralizing prompt management across teams

**Deliverables:**
- Register, version, and load prompts programmatically

---

### L1-6.2 — GenAI Scorers and Judges
**Duration:** 30 min
**Topics:**
- Built-in scorers overview
- Custom scorers: `mlflow.genai.scorers`
- LLM judges: `mlflow.genai.judges` — using an LLM to evaluate another LLM
- Configuring judge models and criteria

**Deliverables:**
- Custom scorer + LLM judge evaluating a Q&A system

---

### L1-6.3 — Datasets and Labeling
**Duration:** 20 min
**Topics:**
- `mlflow.genai.datasets` — creating and managing evaluation datasets
- Dataset schemas for different task types
- `mlflow.genai.labeling` — human-in-the-loop labeling workflows
- Building ground truth datasets

**Deliverables:**
- Create a GenAI evaluation dataset, add labels, use for evaluation

---

## L1-M7: Data and Datasets

### L1-7.1 — Dataset Logging and Lineage
**Duration:** 20 min
**Topics:**
- `mlflow.data` module — logging datasets alongside runs
- Dataset sources: Pandas, Spark, Delta, HuggingFace, HTTP
- Dataset schemas and profiling
- Data lineage: connecting datasets to runs and models
- `mlflow.log_input()` for tracking which data trained which model

**Deliverables:**
- Log a pandas DataFrame as a dataset, link it to a training run

---

## L1-M8: Deployment and Serving

### L1-8.1 — Model Serving Basics
**Duration:** 30 min
**Topics:**
- `mlflow models serve` — local REST API serving
- Serving endpoints: `/invocations`, `/ping`, `/version`
- Input formats: JSON, CSV, split-orient
- `mlflow models predict` — batch prediction from CLI
- Docker containerization: `mlflow models build-docker`

**Deliverables:**
- Serve a model locally, call it via `curl`, run batch predictions

---

### L1-8.2 — AI Gateway Overview
**Duration:** 20 min
**Topics:**
- What is the AI Gateway? (unified LLM endpoint management)
- Route configuration: providers, rate limits, fallbacks
- Supported providers: OpenAI, Anthropic, Mistral, Gemini, Bedrock, etc.
- Cost management and usage tracking
- When to use Gateway vs. direct API calls

**Deliverables:**
- Configure a simple gateway route and call it

---

## L1-M9: Projects

### L1-9.1 — MLflow Projects
**Duration:** 20 min
**Topics:**
- What is an MLflow Project? (reproducible ML workflows)
- `MLproject` file format
- Running projects: `mlflow run`
- Conda and Docker environments
- Git-based projects
- Use cases: reproducibility, CI/CD, collaboration

**Deliverables:**
- Create a simple MLflow Project, run it from CLI

---

## L1-M10: Authentication and Administration

### L1-10.1 — Authentication and Permissions
**Duration:** 15 min
**Topics:**
- Enabling authentication on the tracking server
- User management: creating users, setting permissions
- Experiment and model permissions
- API key authentication
- When you need auth (multi-user, production) vs. when you don't (local dev)

**Deliverables:**
- Enable auth on local server, create a user, set experiment permissions

---

### Level 1 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Core Platform | 4 lessons | ~1.5 hours |
| M2: Models & Registry | 3 lessons | ~1.5 hours |
| M3: Autologging | 2 lessons | ~1 hour |
| M4: Evaluation | 3 lessons | ~1.5 hours |
| M5: Tracing | 2 lessons | ~1 hour |
| M6: GenAI Features | 3 lessons | ~1 hour |
| M7: Data & Datasets | 1 lesson | ~20 min |
| M8: Deployment & Serving | 2 lessons | ~50 min |
| M9: Projects | 1 lesson | ~20 min |
| M10: Auth & Admin | 1 lesson | ~15 min |
| **Total** | **22 lessons** | **~10-12 hours** |

---
---

# LEVEL 2 — PRACTITIONER (Depth)

*Goal: Go deeper in each area. Build real-world projects.*
*Prerequisite: Level 1 completed*
*Estimated time: ~25-30 hours*

---

## L2-M1: Advanced Tracking

### L2-1.1 — Nested Runs and Run Hierarchies
**Duration:** 45 min
**Topics:**
- Nested runs for hyperparameter sweeps
- Parent-child run relationships
- Using nested runs with cross-validation
- Organizing runs with tags for filtering
- Best practices for experiment organization

**Deliverables:**
- Hyperparameter grid search with nested runs, results filterable in UI

---

### L2-1.2 — Async and Batch Logging
**Duration:** 30 min
**Topics:**
- `mlflow.config.enable_async_logging()` — non-blocking logging
- Performance impact of sync vs. async
- Batch logging large numbers of metrics
- Step-based metric logging for training curves
- Logging in distributed/parallel training

**Deliverables:**
- Training loop with async logging, step-based loss curves in UI

---

### L2-1.3 — Artifact Management Deep Dive
**Duration:** 45 min
**Topics:**
- Artifact storage backends: local, S3, GCS, Azure Blob
- Logging different artifact types: images, audio, tables, dicts, figures
- `mlflow.log_image()`, `mlflow.log_table()`, `mlflow.log_figure()`
- Artifact organization and naming conventions
- Large artifact handling and storage limits

**Deliverables:**
- Run that logs multiple artifact types (plots, tables, JSON configs), organized in folders

---

### L2-1.4 — MlflowClient — Programmatic Access
**Duration:** 45 min
**Topics:**
- `MlflowClient` vs. fluent API — when to use which
- CRUD operations: create/get/update/delete experiments, runs, models
- Run lifecycle management
- Downloading artifacts programmatically
- Building custom dashboards and reports from MLflow data

**Deliverables:**
- Script that builds a comparison report across experiments using MlflowClient

---

## L2-M2: Advanced Models

### L2-2.1 — Model Signatures Deep Dive
**Duration:** 45 min
**Topics:**
- `ModelSignature` — input/output schema definition
- `infer_signature()` from training data
- Column-based vs. tensor-based signatures
- Signature enforcement during serving
- Handling complex input types (images, nested JSON)
- Params in signatures (for inference-time configuration)

**Deliverables:**
- Models with different signature types, tested with enforcement

---

### L2-2.2 — Custom PyFunc Models
**Duration:** 1 hour
**Topics:**
- Advanced `PythonModel` subclassing
- `load_context()` for loading dependencies (files, other models)
- `predict()` with params support
- Multi-model ensembles as a single PyFunc
- Wrapping REST API clients as PyFunc models
- Dependency management: `conda_env`, `pip_requirements`, `extra_pip_requirements`

**Deliverables:**
- Custom PyFunc that wraps an ensemble of sklearn + LLM, with configurable predict params

---

### L2-2.3 — Model Registry Workflows
**Duration:** 45 min
**Topics:**
- Model lifecycle: None → Staging → Production → Archived
- Alias-based deployment (`champion`, `challenger`)
- Model descriptions, tags, and annotations
- Webhooks for registry events
- Comparing model versions side-by-side
- Promoting models through CI/CD

**Deliverables:**
- Full registry workflow: train → register → test → promote → serve

---

## L2-M3: Deep Evaluation

### L2-3.1 — Custom Metrics and Evaluators
**Duration:** 1 hour
**Topics:**
- `make_metric()` — custom metric functions
- Custom evaluators: subclassing `EvaluationMetric`
- Metrics that use artifacts (confusion matrix, ROC curve)
- Combining built-in and custom metrics
- Metric validation and thresholds
- Evaluation with `extra_metrics` and `custom_artifacts`

**Deliverables:**
- Custom metric suite for a domain-specific task (e.g., code quality scoring)

---

### L2-3.2 — RAG System Evaluation
**Duration:** 1.5 hours
**Topics:**
- Building a RAG system with LangChain + Chroma
- Context-aware metrics: `faithfulness`, `relevance`, `context_recall`
- Evaluating retrieval quality vs. generation quality separately
- Comparing chunking strategies (size, overlap, method)
- Comparing embedding models
- End-to-end RAG evaluation pipeline

**Deliverables:**
- RAG system with evaluation comparing 3 chunking strategies
- Faithfulness and relevance metrics tracked per configuration

---

### L2-3.3 — GenAI Evaluation Framework
**Duration:** 1 hour
**Topics:**
- `mlflow.genai.evaluation` — the full framework
- Creating evaluation datasets with `mlflow.genai.datasets`
- Built-in scorers vs. custom scorers
- LLM judges with custom criteria and rubrics
- Evaluation runs and comparison in UI
- Batch evaluation across multiple models

**Deliverables:**
- Evaluation framework comparing 3 different LLM configurations on a shared dataset

---

### L2-3.4 — Human-in-the-Loop Evaluation
**Duration:** 45 min
**Topics:**
- `mlflow.genai.labeling` — active labeling workflows
- Assessments: `mlflow.log_assessment()` for human feedback
- Expectation setting and issue tracking
- Combining automated + human evaluation
- Building feedback loops for model improvement

**Deliverables:**
- Labeling workflow that combines LLM-judge pre-screening with human review

---

## L2-M4: Advanced Tracing

### L2-4.1 — Tracing LangGraph State Machines
**Duration:** 1.5 hours
**Topics:**
- LangGraph `StateGraph` with MLflow auto-tracing
- Tracing state transitions between nodes
- Conditional edge tracing
- Parallel node execution tracing
- State visibility and debugging
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph`

**Deliverables:**
- LangGraph workflow with conditional branches, fully traced in MLflow
- Performance analysis identifying slow nodes

---

### L2-4.2 — Tracing Temporal.io Workflows
**Duration:** 1.5 hours
**Topics:**
- Temporal.io workflow and activity basics
- Integrating MLflow tracing with Temporal activities
- Long-running process observability
- Retry and failure tracking with traces
- Durable execution + ML observability
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python/MY/5_AI`

**Deliverables:**
- Temporal workflow with AI activities, fully traced in MLflow
- Workflow execution timeline with failure/retry visibility

---

### L2-4.3 — OpenTelemetry Integration
**Duration:** 45 min
**Topics:**
- MLflow's OpenTelemetry (OTel) foundation
- Exporting traces to OTel-compatible backends (Jaeger, Zipkin)
- Custom span processors and exporters
- Combining MLflow traces with infrastructure traces
- Distributed tracing across services

**Deliverables:**
- MLflow traces exported to an OTel-compatible backend

---

### L2-4.4 — Trace-based Debugging and Analysis
**Duration:** 45 min
**Topics:**
- Using traces to find latency bottlenecks
- Token usage analysis from traces
- Cost estimation from traced LLM calls
- Trace search and filtering at scale
- Building custom trace analysis pipelines
- Trace-based alerts and anomaly detection

**Deliverables:**
- Analysis pipeline that reads traces and produces a latency/cost report

---

## L2-M5: Agent Observability

### L2-5.1 — LangChain Agent Tracking
**Duration:** 1.5 hours
**Topics:**
- Creating agents with LangChain v1.0+ (`create_react_agent`)
- Auto-logging agents with `mlflow.langchain.autolog()`
- Tracking tool calls and reasoning steps
- Agent iteration and decision tracking
- Comparing agent configurations
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent`

**Deliverables:**
- ReAct agent with custom tools, fully traced
- Tool usage metrics and decision visualization

---

### L2-5.2 — LangGraph Agent Observability
**Duration:** 2 hours
**Topics:**
- Building agents with LangGraph (`StateGraph`, nodes, edges)
- Auto-tracing state transitions
- Visualizing agent execution graphs
- Conditional branching and routing in traces
- Debugging agent behavior with traces
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/5_agent`

**Deliverables:**
- LangGraph agent with conditional logic, state transition traces, and execution graph visualization

---

### L2-5.3 — Multi-Agent Systems
**Duration:** 2.5 hours
**Topics:**
- Multi-agent patterns: collaboration, supervision, swarm
- Building multi-agent graphs with agent handoffs
- Tracing inter-agent communication and state sharing
- `create_react_agent` for individual agents in a graph
- Aggregating metrics across agents
- Debugging collaboration failures
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/6_agents`

**Deliverables:**
- Multi-agent system (researcher + summarizer + coder) with full tracing
- Per-agent performance metrics and handoff analysis

---

## L2-M6: Prompt Engineering and Optimization

### L2-6.1 — Prompt Management at Scale
**Duration:** 45 min
**Topics:**
- Prompt Registry deep dive
- Prompt versioning strategies
- A/B testing prompts with MLflow
- Prompt templates with complex variables
- Team collaboration on prompts
- Prompt performance tracking over time

**Deliverables:**
- Prompt A/B test comparing 3 prompt variants with tracked metrics

---

### L2-6.2 — Prompt Optimization
**Duration:** 1 hour
**Topics:**
- `mlflow.genai.optimize` — automated prompt tuning
- In-context learning optimization
- Few-shot example selection
- Systematic prompt improvement workflow
- Tracking optimization history

**Deliverables:**
- Optimized prompt with tracked improvement trajectory

---

## L2-M7: AI Gateway Deep Dive

### L2-7.1 — Gateway Configuration and Routing
**Duration:** 1 hour
**Topics:**
- Route configuration: models, rate limits, API keys
- Provider routing: primary/fallback chains
- Load balancing across providers
- Cost management and budget limits
- Usage tracking and analytics
- Supported providers: OpenAI, Anthropic, Mistral, Gemini, Bedrock, Groq, etc.

**Deliverables:**
- Gateway with multi-provider routing, fallbacks, and rate limits

---

## L2-M8: Deployment Patterns

### L2-8.1 — Model Serving Deep Dive
**Duration:** 1 hour
**Topics:**
- Serving configurations and customization
- Custom request/response handling
- Serving multiple models
- Health checks and monitoring
- Docker-based deployment: `mlflow models build-docker`
- Cloud deployment patterns: AWS, GCP, Azure

**Deliverables:**
- Dockerized model server with custom configuration

---

### L2-8.2 — Batch Prediction Pipelines
**Duration:** 45 min
**Topics:**
- `mlflow models predict` for batch inference
- Building batch prediction scripts
- Scheduling predictions (cron, Airflow, Temporal)
- Result logging and tracking
- Error handling and retry strategies

**Deliverables:**
- Batch prediction pipeline with result tracking in MLflow

---

## L2-M9: Framework Integrations Deep Dive

### L2-9.1 — PyTorch + MLflow
**Duration:** 1 hour
**Topics:**
- `mlflow.pytorch.autolog()` — full capabilities
- Training loop integration
- Logging checkpoints and model artifacts
- Distributed training with MLflow
- PyTorch Lightning integration

**Deliverables:**
- PyTorch training pipeline with full MLflow integration

---

### L2-9.2 — Hugging Face Transformers + MLflow
**Duration:** 1.5 hours
**Topics:**
- `mlflow.transformers.autolog()` — auto-logging
- Fine-tuning tracking with training metrics
- Model logging and loading HF models
- Pipeline serving via MLflow
- Comparing base vs. fine-tuned models

**Deliverables:**
- Fine-tuning experiment with full tracking and model comparison

---

### L2-9.3 — Sentence Transformers + MLflow
**Duration:** 45 min
**Topics:**
- Embedding model tracking
- Evaluating embedding quality
- Logging embedding models to registry
- Use case: tracking embeddings for RAG systems

**Deliverables:**
- Embedding model evaluation with quality metrics tracked

---

### Level 2 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Advanced Tracking | 4 lessons | ~2.5 hours |
| M2: Advanced Models | 3 lessons | ~2.5 hours |
| M3: Deep Evaluation | 4 lessons | ~4 hours |
| M4: Advanced Tracing | 4 lessons | ~4.5 hours |
| M5: Agent Observability | 3 lessons | ~6 hours |
| M6: Prompt Engineering | 2 lessons | ~1.75 hours |
| M7: AI Gateway | 1 lesson | ~1 hour |
| M8: Deployment | 2 lessons | ~1.75 hours |
| M9: Framework Integrations | 3 lessons | ~3.25 hours |
| **Total** | **26 lessons** | **~25-30 hours** |

---
---

# LEVEL 3 — EXPERT (Mastery)

*Goal: Production patterns, custom integrations, advanced agent evaluation. Full mastery.*
*Prerequisite: Levels 1 and 2 completed*
*Estimated time: ~25-35 hours*

---

## L3-M1: Advanced Agent Evaluation (Core Focus)

### L3-1.1 — Agent Testing Framework
**Duration:** 2 hours
**Topics:**
- `mlflow.genai.agent_tester` — automated agent test generation
- `mlflow.genai.simulators` — conversation simulation for testing
- Simulating user interactions with varying complexity
- Success criteria definition and validation
- Generating edge cases and adversarial inputs
- Regression testing for agents

**Deliverables:**
- Automated test suite for a LangGraph agent
- Simulated conversations with failure analysis
- Regression test baseline

---

### L3-1.2 — Agent Quality Metrics Design
**Duration:** 2 hours
**Topics:**
- Designing metrics for agent-specific behaviors:
  - Task completion rate (binary + partial credit)
  - Tool selection accuracy (precision/recall of tool choices)
  - Reasoning quality (coherence, relevance, completeness)
  - Plan quality (for plan-and-execute agents)
  - Collaboration quality (for multi-agent systems)
- Custom scorer implementation for each metric
- Aggregation strategies across test cases
- Statistical significance testing for agent comparisons

**Deliverables:**
- Custom metric suite covering all agent quality dimensions
- Statistical comparison of two agent architectures

---

### L3-1.3 — Agent Architecture Comparison
**Duration:** 2.5 hours
**Topics:**
- Systematic comparison of agent architectures:
  - ReAct vs. Plan-and-Execute
  - Single-agent vs. multi-agent
  - LangChain agents vs. LangGraph agents
- Controlled evaluation methodology
- Ablation studies: which component matters most?
- Cost-quality tradeoff analysis
- Prompt sensitivity analysis

**Deliverables:**
- Comparison study with 3+ agent architectures on a shared benchmark
- Cost-quality Pareto frontier visualization

---

### L3-1.4 — Agent Optimization
**Duration:** 2 hours
**Topics:**
- `mlflow.genai.optimize` for agent instruction tuning
- Systematic prompt optimization for agent system prompts
- Tool description optimization
- Few-shot example selection for agents
- Hyperparameter tuning: temperature, max_tokens, top_p
- Iterative optimization loop with evaluation feedback

**Deliverables:**
- Optimized agent with tracked improvement trajectory across iterations

---

### L3-1.5 — End-to-End Agent Evaluation Pipeline
**Duration:** 2.5 hours
**Topics:**
- Designing a complete evaluation pipeline:
  1. Dataset creation and management
  2. Automated test generation
  3. Multi-dimensional scoring (functional, quality, performance, cost)
  4. Human review for borderline cases
  5. Regression detection and alerting
- Pipeline automation with CI/CD integration
- Versioning evaluation datasets alongside model versions
- Evaluation-driven development workflow

**Deliverables:**
- Complete automated evaluation pipeline for agents
- CI/CD integration (GitHub Actions or similar)
- Dashboard with quality trends over time

---

## L3-M2: Custom Framework Integrations

### L3-2.1 — Claude Agent SDK + MLflow
**Duration:** 2.5 hours
**Topics:**
- Claude Agent SDK architecture and lifecycle
- Building custom MLflow tracing for Claude agents
- Wrapping agent execution with `@mlflow.trace` and manual spans
- Logging agent decisions, tool calls, and outputs
- Custom autolog implementation for Claude Agent SDK
- Evaluation of Claude-based agents with MLflow
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- Source: `~/Projects/github/anthropics/claude-agent-sdk-python`

**Deliverables:**
- Claude Agent SDK agent with full MLflow tracing
- Custom autolog wrapper
- Evaluation results comparing Claude agent configurations

---

### L3-2.2 — Codex SDK + MLflow
**Duration:** 2.5 hours
**Topics:**
- Codex SDK architecture (TypeScript-based)
- Cross-language integration: calling TypeScript from Python (subprocess, REST, or Node bridge)
- Building MLflow tracing for Codex operations
- Logging code generation metrics: correctness, compilation success, test pass rate
- Evaluating code generation quality with custom scorers
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`
- Source: `~/Projects/github/openai/codex/sdk`

**Deliverables:**
- Codex SDK integration with MLflow tracing
- Code generation evaluation pipeline with quality metrics

---

### L3-2.3 — DeepAgents + MLflow
**Duration:** 2 hours
**Topics:**
- DeepAgents architecture and multi-agent patterns
- Existing MLflow integration (if any) vs. custom
- Tracing multi-agent orchestration flows
- Evaluating multi-agent collaboration quality
- Comparing DeepAgents vs. LangGraph multi-agent patterns
- Source: `~/Projects/github/lanchain-ai/deepagents`

**Deliverables:**
- DeepAgents system with MLflow tracing
- Comparison with LangGraph multi-agent approach

---

### L3-2.4 — Building Custom Autolog Integrations
**Duration:** 2 hours
**Topics:**
- MLflow autolog architecture: how it works internally
- Building a custom autolog for any framework
- Monkey-patching vs. decorator-based approaches
- Trace integration for custom frameworks
- Publishing custom integrations as MLflow plugins
- Testing and validating autolog implementations

**Deliverables:**
- Reusable autolog template for arbitrary Python frameworks
- Published as a local MLflow plugin

---

## L3-M3: Production Monitoring and Operations

### L3-3.1 — Production Tracing at Scale
**Duration:** 1.5 hours
**Topics:**
- High-volume trace collection strategies
- Sampling strategies: head-based, tail-based, probabilistic
- Trace storage and retention policies
- Trace-based SLO (Service Level Objective) monitoring
- Anomaly detection on trace data (latency spikes, error rate changes)
- Cost per trace and budget management

**Deliverables:**
- Production tracing configuration with sampling and retention policies

---

### L3-3.2 — Grafana Dashboards for MLflow
**Duration:** 2 hours
**Topics:**
- Exporting MLflow metrics to Prometheus
- Building Grafana dashboards for:
  - Model performance over time
  - Agent quality metrics trends
  - Latency and cost tracking
  - Error rates and failure patterns
- Setting up alerts on quality degradation
- Dashboard templates for common patterns

**Deliverables:**
- Grafana dashboard showing agent performance metrics from MLflow
- Alert rules for quality regression and latency spikes

---

### L3-3.3 — Feedback Loops and Continuous Improvement
**Duration:** 1.5 hours
**Topics:**
- Collecting user feedback on agent responses
- `mlflow.log_assessment()` for production feedback
- Feeding production data back into evaluation datasets
- Identifying drift: prompt drift, data drift, quality drift
- Active learning: selecting the most informative examples for labeling
- Closing the loop: feedback → retrain/re-prompt → evaluate → deploy

**Deliverables:**
- Feedback collection pipeline with drift detection
- Active learning selection strategy

---

### L3-3.4 — CI/CD for AI Applications
**Duration:** 1.5 hours
**Topics:**
- Automated evaluation in CI pipelines (GitHub Actions)
- Quality gates: minimum metric thresholds for deployment
- Model validation before promotion
- Canary deployments with A/B evaluation
- Rollback strategies based on production metrics
- Environment promotion: dev → staging → production

**Deliverables:**
- GitHub Actions workflow with evaluation gates
- Canary deployment configuration with automated rollback

---

## L3-M4: Advanced MLflow Features

### L3-4.1 — MLflow Plugins and Extensibility
**Duration:** 1.5 hours
**Topics:**
- MLflow plugin system architecture
- Custom model flavors
- Custom artifact stores
- Custom tracking backends
- Plugin development workflow
- Publishing and distributing plugins

**Deliverables:**
- Custom MLflow plugin (model flavor or artifact store)

---

### L3-4.2 — Multi-tenant and Enterprise Patterns
**Duration:** 1 hour
**Topics:**
- Workspace isolation and multi-tenancy
- Authentication and authorization at scale
- Experiment and model permissions by team
- Audit logging
- Secrets management for API keys
- High-availability MLflow server deployment

**Deliverables:**
- Multi-tenant MLflow configuration with team-based permissions

---

### L3-4.3 — MLflow + MCP (Model Context Protocol)
**Duration:** 1 hour
**Topics:**
- What is MCP? (standardized tool/resource protocol for AI)
- MLflow's MCP integration
- Using MCP tools in traced agent workflows
- Logging MCP interactions in MLflow

**Deliverables:**
- Agent using MCP tools with full MLflow tracing

---

### L3-4.4 — Advanced Data Management
**Duration:** 1 hour
**Topics:**
- Dataset versioning strategies at scale
- Data lineage across the full ML lifecycle
- Feature store integration patterns
- Large-scale dataset handling (Spark, Delta Lake)
- Data quality monitoring with MLflow

**Deliverables:**
- Data lineage pipeline connecting datasets → training runs → models → predictions

---

## L3-M5: Capstone Projects

### L3-5.1 — Capstone: Production AI Agent Platform
**Duration:** 4-6 hours
**Topics:**
- Build a complete AI agent platform with:
  - Multi-agent system (LangGraph) for a real task (e.g., research assistant)
  - Full MLflow tracing and observability
  - Automated evaluation pipeline with custom metrics
  - Prompt registry for managed prompts
  - Model registry with versioned agent configurations
  - Grafana monitoring dashboard
  - CI/CD with quality gates
  - Feedback collection and continuous improvement loop

**Deliverables:**
- Production-ready AI agent platform
- Complete observability stack
- Documentation for team onboarding

---

### L3-5.2 — Capstone: Agent Framework Benchmark
**Duration:** 4-6 hours
**Topics:**
- Build a standardized benchmark comparing agent frameworks:
  - LangChain/LangGraph agents
  - Claude Agent SDK agents
  - DeepAgents multi-agent systems
  - Custom PyFunc-wrapped agents
- Shared evaluation dataset and metrics
- Statistical analysis of results
- Cost-quality-latency comparison
- Recommendations for framework selection

**Deliverables:**
- Benchmark suite with reproducible results
- Framework comparison report
- Decision matrix for framework selection

---

### Level 3 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Advanced Agent Evaluation | 5 lessons | ~11 hours |
| M2: Custom Framework Integrations | 4 lessons | ~9 hours |
| M3: Production Operations | 4 lessons | ~6.5 hours |
| M4: Advanced MLflow Features | 4 lessons | ~4.5 hours |
| M5: Capstone Projects | 2 projects | ~8-12 hours |
| **Total** | **19 lessons + 2 capstones** | **~25-35 hours** |

---
---

# Complete Course Summary

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 — Essentials** | Breadth: every feature area | 22 lessons | ~10-12 hours |
| **Level 2 — Practitioner** | Depth: real-world projects | 26 lessons | ~25-30 hours |
| **Level 3 — Expert** | Mastery: production + agents | 19 lessons + 2 capstones | ~25-35 hours |
| **Total** | | **67 lessons + 2 capstones** | **~60-77 hours** |

---

## Project Structure

```
tutorial/
├── tutorial_new_syllabus.md          # This file — the master syllabus
├── level_1/
│   ├── M1_core_platform/
│   │   ├── 1_architecture_overview/
│   │   ├── 2_tracking_basics/
│   │   ├── 3_search_query_api/
│   │   └── 4_system_metrics/
│   ├── M2_models_registry/
│   │   ├── 1_models_flavors/
│   │   ├── 2_model_registry/
│   │   └── 3_pyfunc/
│   ├── M3_autologging/
│   │   ├── 1_traditional_ml/
│   │   └── 2_llm_genai/
│   ├── M4_evaluation/
│   │   ├── 1_traditional_ml_eval/
│   │   ├── 2_llm_eval_basics/
│   │   └── 3_llm_as_judge/
│   ├── M5_tracing/
│   │   ├── 1_auto_tracing/
│   │   └── 2_manual_tracing/
│   ├── M6_genai_features/
│   │   ├── 1_prompt_registry/
│   │   ├── 2_scorers_judges/
│   │   └── 3_datasets_labeling/
│   ├── M7_data_datasets/
│   │   └── 1_dataset_logging/
│   ├── M8_deployment/
│   │   ├── 1_model_serving/
│   │   └── 2_ai_gateway/
│   ├── M9_projects/
│   │   └── 1_mlflow_projects/
│   └── M10_auth/
│       └── 1_auth_permissions/
├── level_2/
│   ├── M1_advanced_tracking/
│   │   ├── 1_nested_runs/
│   │   ├── 2_async_batch_logging/
│   │   ├── 3_artifact_management/
│   │   └── 4_mlflow_client/
│   ├── M2_advanced_models/
│   │   ├── 1_signatures_deep_dive/
│   │   ├── 2_custom_pyfunc/
│   │   └── 3_registry_workflows/
│   ├── M3_deep_evaluation/
│   │   ├── 1_custom_metrics/
│   │   ├── 2_rag_evaluation/
│   │   ├── 3_genai_evaluation/
│   │   └── 4_human_in_loop/
│   ├── M4_advanced_tracing/
│   │   ├── 1_langgraph_tracing/
│   │   ├── 2_temporal_tracing/
│   │   ├── 3_opentelemetry/
│   │   └── 4_trace_analysis/
│   ├── M5_agent_observability/
│   │   ├── 1_langchain_agents/
│   │   ├── 2_langgraph_agents/
│   │   └── 3_multiagent_systems/
│   ├── M6_prompt_engineering/
│   │   ├── 1_prompt_management/
│   │   └── 2_prompt_optimization/
│   ├── M7_ai_gateway/
│   │   └── 1_gateway_routing/
│   ├── M8_deployment/
│   │   ├── 1_serving_deep_dive/
│   │   └── 2_batch_prediction/
│   └── M9_framework_integrations/
│       ├── 1_pytorch/
│       ├── 2_huggingface/
│       └── 3_sentence_transformers/
├── level_3/
│   ├── M1_agent_evaluation/
│   │   ├── 1_agent_testing/
│   │   ├── 2_quality_metrics/
│   │   ├── 3_architecture_comparison/
│   │   ├── 4_agent_optimization/
│   │   └── 5_evaluation_pipeline/
│   ├── M2_custom_integrations/
│   │   ├── 1_claude_agent_sdk/
│   │   ├── 2_codex_sdk/
│   │   ├── 3_deepagents/
│   │   └── 4_custom_autolog/
│   ├── M3_production/
│   │   ├── 1_production_tracing/
│   │   ├── 2_grafana_dashboards/
│   │   ├── 3_feedback_loops/
│   │   └── 4_cicd/
│   ├── M4_advanced_features/
│   │   ├── 1_plugins/
│   │   ├── 2_enterprise/
│   │   ├── 3_mcp/
│   │   └── 4_data_management/
│   └── M5_capstones/
│       ├── 1_agent_platform/
│       └── 2_framework_benchmark/
```

## MLflow Feature Coverage Matrix

This matrix shows which MLflow features are covered at each level:

| Feature Area | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Experiment Tracking | Basics | Nested runs, async, MlflowClient | — |
| System Metrics | Overview | — | — |
| Search/Query API | Basics | Advanced filtering | — |
| Models & Flavors | Overview, PyFunc | Signatures, custom PyFunc | Plugins |
| Model Registry | Basics | Lifecycle workflows | Enterprise |
| Autologging (Traditional ML) | Overview | — | — |
| Autologging (LLM/GenAI) | Overview | — | Custom autolog |
| Traditional ML Evaluation | Basics | Custom metrics | — |
| LLM Evaluation | Basics, LLM-as-judge | RAG eval, GenAI framework | Agent eval pipeline |
| Human Evaluation | — | Labeling, assessments | Feedback loops |
| Tracing (Auto) | Basics | LangGraph, Temporal | Production scale |
| Tracing (Manual) | Basics | OTel, analysis | Custom frameworks |
| Prompt Registry | Basics | A/B testing, optimization | — |
| GenAI Scorers/Judges | Basics | Custom scorers | Agent-specific metrics |
| Data/Datasets | Logging, lineage | — | Advanced management |
| AI Gateway | Overview | Routing, fallbacks | — |
| Model Serving | CLI basics | Docker, cloud | — |
| Batch Prediction | — | Pipelines | — |
| MLflow Projects | Overview | — | — |
| Authentication | Overview | — | Multi-tenant |
| Agent Tracking | — | LangChain, LangGraph, Multi-agent | Claude SDK, Codex, DeepAgents |
| Agent Evaluation | — | — | Testing, metrics, optimization, pipeline |
| CI/CD | — | — | Quality gates, canary |
| Grafana Monitoring | — | — | Dashboards, alerts |
| Plugins/Extensibility | — | — | Custom flavors, plugins |
| MCP Integration | — | — | MCP + tracing |
| PyTorch Deep | — | Training integration | — |
| Hugging Face Deep | — | Fine-tuning tracking | — |
| Capstone Projects | — | — | 2 full projects |
