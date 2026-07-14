# MLFlow Tutorial: Three-Level Course

## Philosophy

This tutorial is structured in three progressive levels:

- **Level 1 — Essentials**: Breadth-first. Touch every major MLflow feature. Short lessons (~30 min). Goal: understand what MLflow can do and when to reach for each feature.
- **Level 2 — Practitioner**: Go deeper in each area with real-world scenarios. Longer lessons (~1-2 hours). Goal: build working projects and develop muscle memory.
- **Level 3 — Expert**: Production patterns, custom integrations, advanced evaluation. Goal: master MLflow for production AI systems, with special focus on agent evaluation.

Each level builds on the previous. A user can stop after Level 1 and have a working mental model of the entire platform, or continue through Level 3 for full mastery.

## Target Audience

- **Level 1**: Anyone starting with MLflow — AI developers, ML engineers, data scientists working with LLMs
- **Level 2**: Practitioners building real AI applications who need depth
- **Level 3**: Teams shipping AI agents to production who need evaluation, monitoring, and custom integrations

## Technical Stack

- **Python**: 3.10+
- **Package Manager**: `uv` (every lesson is a standalone project)
- **MLFlow**: Latest (2.x+)
- **LLM provider**: LMStudio (local, OpenAI-compatible API on `localhost:1234`)
- **LLM models**:
  - `google/gemma-4-e4b` — 4B model for simple/fast tasks (Level 1, basic examples)
  - `google/gemma-4-26b-a4b` — 26B MoE model for complex tasks (Level 2/3, evaluation judges, agents)
  - `text-embedding-nomic-embed-text-v1.5` — embedding model for RAG/vector DB
- **Agent Frameworks**: LangChain v1.0+, LangGraph, DeepAgents, Claude Agent SDK
- **Vector DB**: Qdrant (via Podman Compose)
- **Evaluation Benchmark**: SWE-Bench
- **Workflow Orchestration**: Temporal.io (via Podman Compose)
- **Observability**: Grafana + Prometheus (via Podman Compose)
- **Container runtime**: Podman (not Docker)

## Reference Sources

- **MLFlow**
  - Source code: `~/Projects/github/mlflow/mlflow`
  - Documentation: `/Users/lkellers/Projects/github/mlflow/mlflow/docs/docs`
- **LangChain**:
  - Source code: `/Users/lkellers/Projects/github/langchain-ai/langchain`
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain`
- **LangGraph**:
  - Source code: `/Users/lkellers/Projects/github/langchain-ai/langgraph`
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph`
- **DeepAgents**:
  - Source code: `/Users/lkellers/Projects/github/langchain-ai/deepagents`
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/3_deepagents`
- **Claude Agent SDK**:
  - Source code: `/Users/lkellers/Projects/github/anthropics/claude-agent-sdk-python`
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- **Temporal.io**:
  - Code samples: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python`

---
---

# LEVEL 1 — ESSENTIALS (Breadth)

*Goal: Touch every major MLflow feature. Understand the landscape.*
*Estimated time: ~9-11 hours*

---

## L1-M1: Tracking

### L1-M1.1 — Your First MLflow Run
**Duration:** 15 min
**Topics:**
- MLflow's 5 pillars: Tracking, Models, Registry, Evaluation, Deployment
- Architecture: tracking server, backend store (PostgreSQL), artifact store
- Key concepts: experiments, runs, parameters, metrics, artifacts, tags
- Calling a local LLM via LMStudio (OpenAI-compatible API)
- Logging LLM configuration as parameters and results as metrics

**Deliverables:**
- Script that calls an LLM, logs params/metrics/tags, and verifies in the MLflow UI

---

### L1-M1.2 — Tracking LLM Experiments
**Duration:** 30 min
**Topics:**
- Comparing LLM configurations (temperature sweeps) as separate runs
- Bulk logging with `log_params()` and `log_metrics()`
- Step-based metric logging (`log_metric(..., step=N)`) across multiple prompts
- Logging LLM responses as text artifacts (`log_artifact()`)
- Setting tags (`set_tag`, `set_tags`)
- Viewing and comparing results in MLflow UI

**Deliverables:**
- Script that runs temperature comparisons, logs step-based token metrics, and saves response artifacts

---

### L1-M1.3 — Search and Query API
**Duration:** 20 min
**Topics:**
- `mlflow.search_runs()` — filtering and sorting runs
- Search syntax: `params.temperature = '0.3' AND params.prompt_topic = 'transformers'`
- `mlflow.search_experiments()`
- `MlflowClient` for programmatic access
- Exporting results to pandas DataFrames for aggregation

**Deliverables:**
- Script that creates multiple LLM runs, then queries and compares them programmatically

---

### L1-M1.4 — System Metrics Logging
**Duration:** 15 min
**Topics:**
- Enabling system metrics: `mlflow.enable_system_metrics_logging()`
- What gets logged: CPU, memory, disk, network, GPU utilization
- Viewing system metrics in UI alongside LLM metrics
- Use cases: identifying resource bottlenecks during LLM inference

**Deliverables:**
- LLM inference run with system metrics visible in MLflow UI

---

## L1-M2: Models and Registry

### L1-M2.1 — MLflow Models and Flavors
**Duration:** 30 min
**Topics:**
- What is an MLflow Model? (the `MLmodel` file, flavors, signatures)
- Key flavors for LLM work: `pyfunc`, `langchain`, `openai`, `transformers`
- Model signatures: `ModelSignature`, `infer_signature()`
- `mlflow.<flavor>.log_model()` and `mlflow.<flavor>.load_model()`
- Input examples for documentation
- Logging a LangChain agent (compiled `StateGraph`) as an MLflow model

**Deliverables:**
- Log a LangChain agent with signature and input example
- Load it back and run inference

---

### L1-M2.2 — Model Registry
**Duration:** 30 min
**Topics:**
- Registering models: `mlflow.register_model()`
- Model versions and aliases (`champion`, `challenger`)
- Model descriptions and tags
- Transitioning models through stages
- Loading models by name and version/alias

**Deliverables:**
- Register an LLM model, create versions, set aliases, load by alias

---

## L1-M3: Tracing

### L1-M3.1 — Autologging and Auto-Tracing
**Duration:** 30 min
**Topics:**
- `mlflow.openai.autolog()` — trace OpenAI-compatible calls (LMStudio)
- `mlflow.langchain.autolog()` — trace LangChain agents (`create_agent`)
- `mlflow.autolog()` — the universal autolog (enables all 16+ GenAI integrations)
- Other LLM integrations: `mlflow.anthropic.autolog()`, Mistral, Gemini, Bedrock, Groq, LiteLLM, CrewAI, DSPy, and more
- What gets captured: inputs, outputs, latencies, token counts, model info
- Searching and inspecting traces programmatically with `mlflow.search_traces()`
- Trace structure: spans, parent-child relationships

**Deliverables:**
- Script showing autologging for LMStudio via OpenAI SDK, via LangChain agent, and via universal `mlflow.autolog()`, with traces visible in UI

---

### L1-M3.2 — Manual Tracing
**Duration:** 30 min
**Topics:**
- `@mlflow.trace` decorator — function-level tracing
- `mlflow.start_span()` — manual span creation (context manager)
- Adding metadata to spans: `span.set_inputs()`, `span.set_outputs()`, `span.set_attributes()`
- Combining auto and manual tracing in a single trace tree

**Deliverables:**
- Application with both auto-traced LLM calls and manual spans for business logic

---

## L1-M4: Evaluation

### L1-M4.1 — LLM Evaluation Basics
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

### L1-M4.2 — LLM-as-Judge
**Duration:** 30 min
**Topics:**
- What is LLM-as-judge and why use it?
- Using judge metrics: `answer_correctness`, `faithfulness`, `relevance`
- Viewing judge justifications in MLflow UI
- Limitations and biases of LLM judges

**Deliverables:**
- Run LLM-as-judge evaluation, examine justifications in UI

---

## L1-M5: GenAI Features

### L1-M5.1 — Prompt Registry
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

### L1-M5.2 — GenAI Scorers and Judges
**Duration:** 30 min
**Topics:**
- Built-in scorers overview
- Custom scorers: `mlflow.genai.scorers`
- LLM judges: `mlflow.genai.judges` — using an LLM to evaluate another LLM
- Configuring judge models and criteria

**Deliverables:**
- Custom scorer + LLM judge evaluating a Q&A system

---

### L1-M5.3 — Datasets and Labeling
**Duration:** 20 min
**Topics:**
- `mlflow.genai.datasets` — creating and managing evaluation datasets
- Dataset schemas for different task types
- `mlflow.genai.labeling` — human-in-the-loop labeling workflows
- Building ground truth datasets

**Deliverables:**
- Create a GenAI evaluation dataset, add labels, use for evaluation

---

## L1-M6: Data and Datasets

### L1-M6.1 — Dataset Logging and Lineage
**Duration:** 20 min
**Topics:**
- `mlflow.data` module — logging datasets alongside runs
- Dataset sources: Pandas, HuggingFace, HTTP
- Dataset schemas and profiling
- Data lineage: connecting evaluation datasets to runs and models
- `mlflow.log_input()` for tracking which data was used in evaluations

**Deliverables:**
- Log an LLM evaluation dataset, link it to an evaluation run

---

## L1-M7: Deployment and Serving

### L1-M7.1 — Model Serving Basics
**Duration:** 30 min
**Topics:**
- `mlflow models serve` — local REST API serving
- Serving endpoints: `/invocations`, `/ping`, `/version`
- Input formats: JSON, split-orient
- `mlflow models predict` — batch prediction from CLI
- Serving a PyFunc-wrapped LLM model

**Deliverables:**
- Serve an LLM model locally, call it via `curl`, run batch predictions

---

### L1-M7.2 — AI Gateway Overview
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

## L1-M8: Authentication and Administration

### L1-M8.1 — Authentication and Permissions
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
| M1: Tracking | 4 lessons | ~1.5 hours |
| M2: Models & Registry | 3 lessons | ~1.5 hours |
| M3: Tracing | 2 lessons | ~1 hour |
| M4: Evaluation | 2 lessons | ~1 hour |
| M5: GenAI Features | 3 lessons | ~1 hour |
| M6: Data & Datasets | 1 lesson | ~20 min |
| M7: Deployment & Serving | 2 lessons | ~50 min |
| M8: Auth & Admin | 1 lesson | ~15 min |
| **Total** | **18 lessons** | **~8-10 hours** |

---
---

# LEVEL 2 — PRACTITIONER (Depth)

*Goal: Go deeper in each area. Build real-world projects.*
*Prerequisite: Level 1 completed*
*Estimated time: ~20-25 hours*

---

## L2-M1: Advanced Tracking

### L2-M1.1 — Nested Runs and Run Hierarchies
**Duration:** 45 min
**Topics:**
- Nested runs for LLM configuration sweeps (temperature, model variants, prompt variants)
- Parent-child run relationships
- Organizing prompt/model comparisons with nested runs
- Organizing runs with tags for filtering
- Best practices for experiment organization

**Deliverables:**
- LLM configuration sweep (temperature x prompt variant) with nested runs, results filterable in UI

---

### L2-M1.2 — Async and Batch Logging
**Duration:** 30 min
**Topics:**
- `mlflow.config.enable_async_logging()` — non-blocking logging
- Performance impact of sync vs. async
- Batch logging large numbers of metrics during LLM evaluation runs
- Step-based metric logging for iterative LLM workflows
- Logging in parallel/concurrent LLM inference

**Deliverables:**
- Batch LLM evaluation loop with async logging, step-based quality metrics in UI

---

### L2-M1.3 — Artifact Management Deep Dive
**Duration:** 45 min
**Topics:**
- Artifact storage backends: local, S3, GCS, Azure Blob
- Logging LLM-specific artifacts: generated texts, evaluation reports, trace exports
- `mlflow.log_image()`, `mlflow.log_table()`, `mlflow.log_figure()`
- Artifact organization and naming conventions
- Large artifact handling and storage limits

**Deliverables:**
- Run that logs LLM evaluation artifacts (response tables, metric plots, JSON reports), organized in folders

---

### L2-M1.4 — MlflowClient — Programmatic Access
**Duration:** 45 min
**Topics:**
- `MlflowClient` vs. fluent API — when to use which
- CRUD operations: create/get/update/delete experiments, runs, models
- Run lifecycle management
- Downloading artifacts programmatically
- Building custom dashboards and reports from MLflow data

**Deliverables:**
- Script that builds a comparison report across LLM experiments using MlflowClient

---

## L2-M2: Advanced Models

### L2-M2.1 — Model Signatures Deep Dive
**Duration:** 45 min
**Topics:**
- `ModelSignature` — input/output schema definition
- `infer_signature()` from LLM input/output pairs
- Signatures for chat messages, completions, tool call interfaces
- Signature enforcement during serving
- Handling complex input types (nested JSON, chat history)
- Params in signatures (for inference-time configuration: temperature, max_tokens)

**Deliverables:**
- LLM models with different signature types (chat, completion, tool-call), tested with enforcement

---

### L2-M2.2 — Custom PyFunc Models
**Duration:** 1 hour
**Topics:**
- Advanced `PythonModel` subclassing
- `load_context()` for loading dependencies (config files, prompt templates)
- `predict()` with params support
- Wrapping a RAG pipeline as a single PyFunc model
- Wrapping an agent as a PyFunc model
- Dependency management: `conda_env`, `pip_requirements`, `extra_pip_requirements`

**Deliverables:**
- Custom PyFunc that wraps a RAG pipeline (LangChain + Qdrant), with configurable predict params

---

### L2-M2.3 — Model Registry Workflows
**Duration:** 45 min
**Topics:**
- Model lifecycle: None → Staging → Production → Archived
- Alias-based deployment (`champion`, `challenger`)
- Model descriptions, tags, and annotations
- Comparing model versions side-by-side
- Promoting LLM models through CI/CD

**Deliverables:**
- Full registry workflow: build LLM model → register → evaluate → promote → serve

---

## L2-M3: Deep Evaluation

### L2-M3.1 — Custom Metrics and Evaluators
**Duration:** 1 hour
**Topics:**
- `make_metric()` — custom metric functions for LLM output quality
- Custom evaluators: subclassing `EvaluationMetric`
- Domain-specific LLM metrics (e.g., code quality scoring, instruction following, safety)
- Combining built-in and custom metrics
- Metric validation and thresholds
- Evaluation with `extra_metrics` and `custom_artifacts`

**Deliverables:**
- Custom metric suite for an LLM task (e.g., code generation quality scoring)

---

### L2-M3.2 — RAG System Evaluation
**Duration:** 1.5 hours
**Topics:**
- Building a RAG system with LangChain + Qdrant
- Context-aware metrics: `faithfulness`, `relevance`, `context_recall`
- Evaluating retrieval quality vs. generation quality separately
- Comparing chunking strategies (size, overlap, method)
- Comparing embedding models
- End-to-end RAG evaluation pipeline

**Deliverables:**
- RAG system with evaluation comparing 3 chunking strategies
- Faithfulness and relevance metrics tracked per configuration

---

### L2-M3.3 — GenAI Evaluation Framework
**Duration:** 1 hour
**Topics:**
- `mlflow.genai.evaluation` — the full framework
- Creating evaluation datasets with `mlflow.genai.datasets`
- Built-in scorers vs. custom scorers
- LLM judges with custom criteria and rubrics
- Evaluation runs and comparison in UI
- Batch evaluation across multiple models/configurations

**Deliverables:**
- Evaluation framework comparing 3 different LLM configurations on a shared dataset

---

### L2-M3.4 — Human-in-the-Loop Evaluation
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

### L2-M4.1 — Tracing LangGraph State Machines
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

### L2-M4.2 — Tracing Temporal.io Workflows
**Duration:** 1.5 hours
**Topics:**
- Temporal.io workflow and activity basics
- Integrating MLflow tracing with Temporal activities
- Long-running process observability
- Retry and failure tracking with traces
- Durable execution + AI observability
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python/MY/5_AI`

**Deliverables:**
- Temporal workflow with AI activities, fully traced in MLflow
- Workflow execution timeline with failure/retry visibility

---

### L2-M4.3 — OpenTelemetry Integration
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

### L2-M4.4 — Trace-based Debugging and Analysis
**Duration:** 45 min
**Topics:**
- Using traces to find latency bottlenecks in LLM pipelines
- Token usage analysis from traces
- Cost estimation from traced LLM calls
- Trace search and filtering at scale
- Building custom trace analysis pipelines
- Trace-based alerts and anomaly detection

**Deliverables:**
- Analysis pipeline that reads traces and produces a latency/cost report

---

## L2-M5: Agent Observability

### L2-M5.1 — LangChain Agent Tracking
**Duration:** 1.5 hours
**Topics:**
- Creating agents with LangChain v1+ (`create_agent` from `langchain.agents`)
- Tools with the `@tool` decorator (`langchain_core.tools`)
- Auto-logging agents with `mlflow.langchain.autolog()`
- Tracking tool calls and reasoning steps
- Agent middleware: `HumanInTheLoopMiddleware`, `TodoListMiddleware`
- Comparing agent configurations
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent`

**Deliverables:**
- Agent (`create_agent`) with custom tools, fully traced
- Tool usage metrics and decision visualization

---

### L2-M5.2 — LangGraph Agent Observability
**Duration:** 2 hours
**Topics:**
- Building agents with LangGraph (`StateGraph`, nodes, edges, `ToolNode`)
- `create_agent` returns a compiled `StateGraph` — understanding the relationship
- Auto-tracing state transitions
- Visualizing agent execution graphs
- Conditional edges (`add_conditional_edges`) and routing in traces
- Debugging agent behavior with traces
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/5_agent`

**Deliverables:**
- LangGraph agent with conditional logic, state transition traces, and execution graph visualization

---

### L2-M5.3 — Multi-Agent Systems
**Duration:** 2.5 hours
**Topics:**
- Multi-agent patterns: collaboration, supervision, swarm
- Building multi-agent graphs with agent handoffs (`Command(goto=..., graph=Command.PARENT)`)
- Tracing inter-agent communication and state sharing
- `create_agent` for individual agents composed as subgraphs
- Swarm pattern: transfer tools for agent-to-agent handoff
- Aggregating metrics across agents
- Debugging collaboration failures
- Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/6_agents`

**Deliverables:**
- Multi-agent system (researcher + summarizer + coder) with full tracing
- Per-agent performance metrics and handoff analysis

---

## L2-M6: Prompt Engineering and Optimization

### L2-M6.1 — Prompt Management at Scale
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

### L2-M6.2 — Prompt Optimization
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

### L2-M7.1 — Gateway Configuration and Routing
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

### L2-M8.1 — LLM Serving Deep Dive
**Duration:** 1 hour
**Topics:**
- Serving LLM-backed models (PyFunc-wrapped pipelines, agents)
- Custom request/response handling for chat interfaces
- Serving multiple model versions
- Health checks and monitoring
- Docker-based deployment: `mlflow models build-docker`
- Cloud deployment patterns

**Deliverables:**
- Dockerized LLM model server with custom configuration

---

### L2-M8.2 — Batch Prediction Pipelines
**Duration:** 45 min
**Topics:**
- `mlflow models predict` for batch LLM inference
- Building batch prediction scripts for LLM evaluation
- Scheduling predictions (cron, Temporal)
- Result logging and tracking
- Error handling and retry strategies

**Deliverables:**
- Batch LLM inference pipeline with result tracking in MLflow

---

## L2-M9: LLM Fine-Tuning

### L2-M9.1 — Hugging Face Transformers + MLflow
**Duration:** 1.5 hours
**Topics:**
- `mlflow.transformers.autolog()` — auto-logging for fine-tuning
- Fine-tuning a small LLM with training metrics tracking
- Logging checkpoints and model artifacts
- Model logging and loading HF models via MLflow
- Comparing base vs. fine-tuned models with evaluation metrics
- Pipeline serving via MLflow

**Deliverables:**
- Fine-tuning experiment with full tracking and base vs. fine-tuned model comparison

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
| M9: LLM Fine-Tuning | 1 lesson | ~1.5 hours |
| **Total** | **24 lessons** | **~22-27 hours** |

---
---

# LEVEL 3 — EXPERT (Mastery)

*Goal: Production patterns, custom integrations, advanced agent evaluation. Full mastery.*
*Prerequisite: Levels 1 and 2 completed*
*Estimated time: ~25-35 hours*

---

## L3-M1: Advanced Agent Evaluation (Core Focus)

### L3-M1.1 — Agent Testing Framework
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

### L3-M1.2 — Agent Quality Metrics Design
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

### L3-M1.3 — Agent Architecture Comparison
**Duration:** 2.5 hours
**Topics:**
- Systematic comparison of agent architectures:
  - Single-agent (`create_agent`) vs. custom `StateGraph` agents
  - Single-agent vs. multi-agent (swarm, supervision, collaboration)
  - LangChain/LangGraph agents vs. DeepAgents (`create_deep_agent`)
- Controlled evaluation methodology
- Ablation studies: which component matters most?
- Cost-quality tradeoff analysis
- Prompt sensitivity analysis

**Deliverables:**
- Comparison study with 3+ agent architectures on a shared benchmark
- Cost-quality Pareto frontier visualization

---

### L3-M1.4 — Agent Optimization
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

### L3-M1.5 — End-to-End Agent Evaluation Pipeline
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

### L3-M2.1 — Claude Agent SDK + MLflow
**Duration:** 2.5 hours
**Topics:**
- Claude Agent SDK architecture and lifecycle
- Building custom MLflow tracing for Claude agents
- Wrapping agent execution with `@mlflow.trace` and manual spans
- Logging agent decisions, tool calls, and outputs
- Custom autolog implementation for Claude Agent SDK
- Evaluation of Claude-based agents with MLflow
- Reference code: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- Source: `/Users/lkellers/Projects/github/anthropics/claude-agent-sdk-python`

**Deliverables:**
- Claude Agent SDK agent with full MLflow tracing
- Custom autolog wrapper
- Evaluation results comparing Claude agent configurations

---

### L3-M2.2 — DeepAgents + MLflow
**Duration:** 2 hours
**Topics:**
- DeepAgents architecture: `create_deep_agent()` built on top of `create_agent()`
- Built-in tools (filesystem, planning, sub-agent delegation via `task` tool)
- Sub-agents with isolated context windows
- Backends: `StateBackend`, `FilesystemBackend`, `CompositeBackend`
- Tracing multi-agent orchestration flows with MLflow
- Evaluating multi-agent collaboration quality
- Comparing DeepAgents vs. LangGraph multi-agent patterns
- Reference: `/Users/lkellers/Projects/github/langchain-ai/deepagents`

**Deliverables:**
- DeepAgents system with MLflow tracing
- Comparison with LangGraph multi-agent approach

---

### L3-M2.3 — SWE-Bench Evaluation Pipeline
**Duration:** 2.5 hours
**Topics:**
- SWE-Bench: the standardized benchmark for coding agents
- Setting up SWE-Bench Verified dataset from HuggingFace
- Building an agent that attempts SWE-Bench tasks
- Integrating SWE-Bench evaluation with MLflow tracking
- Logging per-instance results, pass rates, and error analysis
- Comparing agent configurations on SWE-Bench
- Reference: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified

**Deliverables:**
- SWE-Bench evaluation pipeline integrated with MLflow
- Agent performance comparison across configurations
- Per-instance failure analysis logged as artifacts

---

### L3-M2.4 — Building Custom Autolog Integrations
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

### L3-M3.1 — Production Tracing at Scale
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

### L3-M3.2 — Grafana Dashboards for MLflow
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

### L3-M3.3 — Feedback Loops and Continuous Improvement
**Duration:** 1.5 hours
**Topics:**
- Collecting user feedback on agent responses
- `mlflow.log_assessment()` for production feedback
- Feeding production data back into evaluation datasets
- Identifying drift: prompt drift, data drift, quality drift
- Active learning: selecting the most informative examples for labeling
- Closing the loop: feedback → re-prompt → evaluate → deploy

**Deliverables:**
- Feedback collection pipeline with drift detection
- Active learning selection strategy

---

### L3-M3.4 — CI/CD for AI Applications
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

### L3-M4.1 — MLflow Plugins and Extensibility
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

### L3-M4.2 — Multi-tenant and Enterprise Patterns
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

### L3-M4.3 — MLflow + MCP (Model Context Protocol)
**Duration:** 1 hour
**Topics:**
- What is MCP? (standardized tool/resource protocol for AI)
- MLflow's MCP integration
- Using MCP tools in traced agent workflows
- Logging MCP interactions in MLflow

**Deliverables:**
- Agent using MCP tools with full MLflow tracing

---

### L3-M4.4 — Advanced Data Management
**Duration:** 1 hour
**Topics:**
- Dataset versioning strategies at scale
- Data lineage across the full AI lifecycle
- Large-scale evaluation dataset management
- Data quality monitoring with MLflow
- Connecting evaluation datasets → runs → models → production

**Deliverables:**
- Data lineage pipeline connecting evaluation datasets → agent runs → models → production metrics

---

## L3-M5: Capstone Projects

### L3-M5.1 — Capstone: Production AI Agent Platform
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

### L3-M5.2 — Capstone: Agent Framework Benchmark
**Duration:** 4-6 hours
**Topics:**
- Build a standardized benchmark comparing agent frameworks:
  - LangChain agents
  - LangGraph agents
  - DeepAgents multi-agent systems
  - Claude Agent SDK agents
  - Custom PyFunc-wrapped agents
- Shared evaluation dataset and metrics
- SWE-Bench subset as standardized coding benchmark
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
| **Total** | **19 lessons + 2 capstones** | **~28-38 hours** |

---
---

# Complete Course Summary

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 — Essentials** | Breadth: every feature area | 19 lessons | ~9-11 hours |
| **Level 2 — Practitioner** | Depth: real-world projects | 24 lessons | ~22-27 hours |
| **Level 3 — Expert** | Mastery: production + agents | 19 lessons + 2 capstones | ~28-38 hours |
| **Total** | | **62 lessons + 2 capstones** | **~59-76 hours** |

---

## Changes from Previous Syllabus

### Removed (pure ML, no LLM/agent relevance)
- **L1-M3.1** Traditional ML Autologging (sklearn, XGBoost, PyTorch autolog)
- **L1-M4.1** Traditional ML Evaluation (classifier metrics, confusion matrix, ROC)
- **L1-M9** MLflow Projects (legacy feature, not relevant to LLM/agent workflows)
- **L2-M9.1** PyTorch + MLflow (pure ML training)
- **L2-M9.3** Sentence Transformers + MLflow (embedding model tracking — covered in RAG eval)
- **L3-M2.2** Codex SDK (not in preferred tech stack)

### Added
- **L3-M2.3** SWE-Bench Evaluation Pipeline (standardized coding agent benchmark)

### Consolidated
- **L1-M3 (Autologging) + L1-M5 (Tracing)** merged into **L1-M3 Tracing** (autologging IS auto-tracing; keeping them separate was redundant). Modules M6–M9 renumbered to M5–M8.

### Kept (reframed for LLM focus)
- **L2-M9.1** HuggingFace Transformers — reframed for LLM fine-tuning (not generic ML training)
- **L3-M2.1** Claude Agent SDK — custom MLflow tracing integration

### Reframed (from ML to LLM/agent focus)
- **L1-M2.1** Models & Flavors: sklearn → LangChain agent logging (`langchain` flavor)
- **L2-M1.1** Nested Runs: hyperparameter grid search → LLM configuration sweeps
- **L2-M1.2** Async Logging: training loop → batch LLM evaluation
- **L2-M2.1** Signatures: tensor-based → chat/completion/tool-call signatures
- **L2-M2.2** Custom PyFunc: sklearn ensemble → RAG pipeline wrapping
- **L2-M3.1** Custom Metrics: generic → LLM-specific (code quality, instruction following)
- **L2-M8.1** Serving: generic model server → LLM model server
- **L2-M9.1** HuggingFace: generic training → LLM fine-tuning with base vs. fine-tuned comparison
- **L3-M4.4** Data Management: feature stores → evaluation dataset management
- **L3-M5.2** Capstone Benchmark: LangChain/LangGraph/DeepAgents/Claude SDK (removed Codex), added SWE-Bench subset

### Tech Stack Updates
- **LMStudio** replaces Ollama (OpenAI-compatible API on localhost:1234)
- **Gemma4-E4B** (4B) as primary small model, **Gemma4-26B** for complex tasks
- **Claude Agent SDK** kept as agent framework
- **DeepAgents** added as a primary agent framework (alongside LangChain/LangGraph)
- **SWE-Bench** added for standardized evaluation benchmarking
- Removed: scikit-learn, XGBoost, Codex SDK

---

## Project Structure

```
tutorial/
├── syllabus.md                     # This file — the master syllabus
├── level_1/
│   ├── M1_tracking/
│   │   ├── 1_first_run/
│   │   ├── 2_tracking_basics/
│   │   ├── 3_search_query_api/
│   │   └── 4_system_metrics/
│   ├── M2_models_registry/
│   │   ├── 1_models_flavors/
│   │   └── 2_model_registry/
│   ├── M3_tracing/
│   │   ├── 1_autologging/
│   │   └── 2_manual_tracing/
│   ├── M4_evaluation/
│   │   ├── 1_llm_eval_basics/
│   │   └── 2_llm_as_judge/
│   ├── M5_genai_features/
│   │   ├── 1_prompt_registry/
│   │   ├── 2_scorers_judges/
│   │   └── 3_datasets_labeling/
│   ├── M6_data_datasets/
│   │   └── 1_dataset_logging/
│   ├── M7_deployment/
│   │   ├── 1_model_serving/
│   │   └── 2_ai_gateway/
│   └── M8_auth/
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
│   └── M9_llm_finetuning/
│       └── 1_huggingface/
├── level_3/
│   ├── M1_agent_evaluation/
│   │   ├── 1_agent_testing/
│   │   ├── 2_quality_metrics/
│   │   ├── 3_architecture_comparison/
│   │   ├── 4_agent_optimization/
│   │   └── 5_evaluation_pipeline/
│   ├── M2_custom_integrations/
│   │   ├── 1_claude_agent_sdk/
│   │   ├── 2_deepagents/
│   │   ├── 3_swe_bench/
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

| Feature Area | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Experiment Tracking | Basics | Nested runs, async, MlflowClient | — |
| System Metrics | Overview | — | — |
| Search/Query API | Basics | Advanced filtering | — |
| Models & Flavors | LLM flavors, PyFunc | Signatures, custom PyFunc | Plugins |
| Model Registry | Basics | Lifecycle workflows | Enterprise |
| Tracing (Auto + Autolog) | OpenAI, LangChain, universal | LangGraph, Temporal | Production scale, custom autolog |
| Tracing (Manual) | Decorator, start_span | OTel, analysis | Custom frameworks |
| LLM Evaluation | Basics, LLM-as-judge | RAG eval, GenAI framework, custom metrics | Agent eval pipeline |
| Human Evaluation | — | Labeling, assessments | Feedback loops |
| Prompt Registry | Basics | A/B testing, optimization | — |
| GenAI Scorers/Judges | Basics | Custom scorers | Agent-specific metrics |
| Data/Datasets | Logging, lineage | — | Advanced management |
| AI Gateway | Overview | Routing, fallbacks | — |
| Model Serving | CLI basics | Docker, cloud | — |
| Batch Prediction | — | Pipelines | — |
| Authentication | Overview | — | Multi-tenant |
| LLM Fine-Tuning | — | HuggingFace Transformers | — |
| Agent Tracking | — | LangChain, LangGraph, Multi-agent | Claude Agent SDK, DeepAgents |
| Agent Evaluation | — | — | Testing, metrics, optimization, pipeline, SWE-Bench |
| CI/CD | — | — | Quality gates, canary |
| Grafana Monitoring | — | — | Dashboards, alerts |
| Plugins/Extensibility | — | — | Custom flavors, plugins |
| MCP Integration | — | — | MCP + tracing |
| Capstone Projects | — | — | 2 full projects |
