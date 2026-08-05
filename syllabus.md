# MLFlow Tutorial: Three-Level Course

## Philosophy

This tutorial is structured in three domain-based levels:

- **Level 1 — Models**: Everything about models and LLMs in MLflow. Tracking, tracing, evaluation (offline and online), prompt registry, deployment, AI gateway, and optimization. Each topic is covered end-to-end so that a user finishing Level 1 has full command of MLflow for single-model workflows. Uses `google/gemma-4-e4b` (fast, lightweight) for all lessons.
- **Level 2 — AI Agents**: Everything about AI agents. Assumes Level 1 knowledge. Covers agent frameworks (LangChain, LangGraph, multi-agent), custom integrations (Claude Agent SDK, DeepAgents), agent evaluation (instruments, offline including standardized benchmarks, online), and agent optimization. Uses `google/gemma-4-26b-a4b` (stronger reasoning) for all lessons.
- **Level 3 — Advanced**: Production patterns, infrastructure, extensibility, and capstone projects. Ties together everything from Levels 1 and 2 into production-grade systems.

Each level builds on the previous. A user can stop after Level 1 and have complete mastery of MLflow for model/LLM workflows, continue through Level 2 for agent expertise, or go through Level 3 for production readiness.

## Target Audience

- **Level 1**: Anyone starting with MLflow for LLM work -- AI developers, ML engineers, data scientists working with language models
- **Level 2**: Practitioners building AI agent systems who need observability, evaluation, and optimization
- **Level 3**: Teams shipping AI agents to production who need monitoring, CI/CD, custom integrations, and enterprise patterns

## Technical Stack

- **Python**: 3.10+
- **Package Manager**: `uv` (every lesson is a standalone project)
- **MLFlow**: Latest (2.x+)
- **LLM provider**: LMStudio (local, OpenAI-compatible API on `localhost:1234`)
- **LLM models**:
  - `google/gemma-4-e4b` -- 4B model for simple/fast tasks (Level 1)
  - `google/gemma-4-26b-a4b` -- 26B MoE model for complex tasks (Level 2/3, evaluation judges, agents)
  - `text-embedding-nomic-embed-text-v1.5` -- embedding model for RAG/vector DB
- **Agent Frameworks**: LangChain v1.0+, LangGraph, DeepAgents, Claude Agent SDK
- **Vector DB**: Qdrant (via Podman Compose)
- **Evaluation Benchmarks**: SWE-Bench, GAIA
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

## LEVEL 1 -- MODELS

*Goal: Complete mastery of MLflow for single-model and LLM workflows. Tracking, tracing, evaluation (offline and online), prompt registry, deployment, gateway, and optimization -- each topic covered end-to-end.*
*LLM model: `google/gemma-4-e4b`*
*Estimated time: ~17.25 hours (19 lessons)*

---

### L1-M1: Tracking

#### L1-M1.1 -- Tracking Fundamentals and Logging

**Duration:** 45 min
**Topics:**
- MLflow's pillars: Tracking, Models, Registry, Evaluation, Deployment
- Architecture: tracking server, backend store (PostgreSQL), artifact store
- Key concepts: experiments, runs, parameters, metrics, artifacts, tags
- Calling a local LLM via LMStudio (OpenAI-compatible API)
- Logging LLM configuration as parameters and results as metrics
- Bulk logging with `log_params()` and `log_metrics()`
- Step-based metric logging (`log_metric(..., step=N)`) across multiple prompts
- Logging LLM responses as text artifacts (`log_artifact()`)
- Setting tags (`set_tag`, `set_tags`)
- Enabling system metrics: `mlflow.enable_system_metrics_logging()`
- What gets logged: CPU, memory, disk, network, GPU utilization
- Viewing and comparing results in MLflow UI

**Deliverables:**
- Script that calls an LLM, logs params/metrics/tags/artifacts, enables system metrics, and verifies everything in the MLflow UI

---

#### L1-M1.2 -- Search, Query API, and MlflowClient

**Duration:** 45 min
**Topics:**
- `mlflow.search_runs()` -- filtering and sorting runs
- Search syntax: `params.temperature = '0.3' AND params.prompt_topic = 'transformers'`
- `mlflow.search_experiments()`
- `MlflowClient` for programmatic access
- `MlflowClient` vs. fluent API -- when to use which
- CRUD operations: create/get/update/delete experiments, runs
- Downloading artifacts programmatically
- Exporting results to pandas DataFrames for aggregation

**Deliverables:**
- Script that creates multiple LLM runs, then queries and compares them using both fluent API and MlflowClient

---

#### L1-M1.3 -- Advanced Tracking Patterns

**Duration:** 60 min
**Topics:**
- Nested runs for LLM configuration sweeps (temperature, model variants, prompt variants)
- Parent-child run relationships
- Organizing prompt/model comparisons with nested runs
- Organizing runs with tags for filtering
- `mlflow.config.enable_async_logging()` -- non-blocking logging
- Performance impact of sync vs. async logging
- Batch logging large numbers of metrics during LLM evaluation runs
- Logging in parallel/concurrent LLM inference
- Artifact storage and organization
- Logging LLM-specific artifacts: generated texts, evaluation reports
- `mlflow.log_image()`, `mlflow.log_table()`, `mlflow.log_figure()`
- Best practices for experiment organization

**Deliverables:**
- LLM configuration sweep with nested runs, async logging, and organized artifacts visible in the MLflow UI

---

### L1-M2: Tracing

#### L1-M2.1 -- Auto-Tracing and Manual Tracing

**Duration:** 45 min
**Topics:**
- `mlflow.openai.autolog()` -- trace OpenAI-compatible calls (LMStudio)
- `mlflow.langchain.autolog()` -- trace LangChain agents
- `mlflow.autolog()` -- the universal autolog (enables all 16+ GenAI integrations)
- Other LLM integrations: `mlflow.anthropic.autolog()`, Mistral, Gemini, Bedrock, Groq, LiteLLM, CrewAI, DSPy, and more
- What gets captured: inputs, outputs, latencies, token counts, model info
- Searching and inspecting traces programmatically with `mlflow.search_traces()`
- Trace structure: spans, parent-child relationships
- `@mlflow.trace` decorator -- function-level tracing
- `mlflow.start_span()` -- manual span creation (context manager)
- Adding metadata to spans: `span.set_inputs()`, `span.set_outputs()`, `span.set_attributes()`
- Combining auto and manual tracing in a single trace tree

**Deliverables:**
- Application with both auto-traced LLM calls and manual spans for business logic, traces visible in UI

---

#### L1-M2.2 -- Trace Analysis and Debugging

**Duration:** 45 min
**Topics:**
- Using traces to find latency bottlenecks in LLM pipelines
- Token usage analysis from traces
- Cost estimation from traced LLM calls
- Trace search and filtering at scale
- Building custom trace analysis pipelines
- Debugging LLM pipeline failures using trace data

**Deliverables:**
- Analysis pipeline that reads traces and produces a latency/cost report

---

### L1-M3: Models and Registry

#### L1-M3.1 -- Models, Flavors, and Signatures

**Duration:** 60 min
**Topics:**
- What is an MLflow Model? (the `MLmodel` file, flavors, signatures)
- Key flavors for LLM work: `pyfunc`, `langchain`, `openai`, `transformers`
- Model signatures: `ModelSignature`, `infer_signature()`
- `mlflow.<flavor>.log_model()` and `mlflow.<flavor>.load_model()`
- Input examples for documentation
- Signatures for chat messages, completions, tool call interfaces
- Signature enforcement during serving
- Handling complex input types (nested JSON, chat history)
- Params in signatures (for inference-time configuration: temperature, max_tokens)

**Deliverables:**
- LLM models with different signature types (chat, completion), logged and loaded back for inference

---

#### L1-M3.2 -- Custom PyFunc Models

**Duration:** 60 min
**Topics:**
- `PythonModel` subclassing
- `load_context()` for loading dependencies (config files, prompt templates)
- `predict()` with params support
- Wrapping a RAG pipeline as a single PyFunc model
- Wrapping an LLM workflow as a PyFunc model
- Dependency management: `conda_env`, `pip_requirements`, `extra_pip_requirements`

**Deliverables:**
- Custom PyFunc that wraps an LLM pipeline, with configurable predict params

---

#### L1-M3.3 -- Model Registry Workflows

**Duration:** 45 min
**Topics:**
- Registering models: `mlflow.register_model()`
- Model versions and aliases (`champion`, `challenger`)
- Model descriptions and tags
- Model lifecycle: None to Staging to Production to Archived
- Alias-based deployment
- Comparing model versions side-by-side
- Loading models by name and version/alias

**Deliverables:**
- Full registry workflow: log LLM model, register, create versions, set aliases, compare, load by alias

---

### L1-M4: Evaluation

*Fundamentals first, then the split that organises everything after it.
**M4.2 Offline** works from a curated dataset with known expectations, scores
every case, and runs when you say so -- it answers "is this version good enough
to ship?" **M4.3 Online** scores sampled production traces that have no expected
answers, on a schedule the server owns -- it answers "is what shipped still
good?" Neither replaces the other.*

*The same split reappears at L2-M2 for agents. Benchmarking, which belongs under
offline, is deliberately not covered at this level: benchmarking a model you did
not train is mostly reading a published number, so it earns its place only once
you are evaluating an agent you built (L2-M2.2).*

#### L1-M4.1: Fundamentals

##### L1-M4.1.1 -- Evaluation Fundamentals

**Duration:** 60 min
**Topics:**
- `mlflow.evaluate()` for LLMs -- `model_type="question-answering"`, `"text-summarization"`, `"text"`
- Built-in LLM metrics: `toxicity`, `flesch_kincaid_grade_level`, `token_count`
- GenAI metrics: `answer_similarity`, `answer_correctness`, `faithfulness`
- Creating evaluation datasets with pandas DataFrames
- Interpreting evaluation results
- What is LLM-as-judge and why use it?
- Using judge metrics: `answer_correctness`, `faithfulness`, `relevance`
- Viewing judge justifications in MLflow UI
- Limitations and biases of LLM judges
- Built-in scorers overview: `mlflow.genai.scorers`
- LLM judges: `mlflow.genai.judges` -- using an LLM to evaluate another LLM
- Configuring judge models and criteria

**Deliverables:**
- Evaluate a Q&A system on a small dataset with built-in metrics, GenAI metrics, LLM-as-judge, and custom scorers

---

#### L1-M4.2: Offline

*Curated dataset, known expectations, full coverage, you pull the trigger.*

##### L1-M4.2.1 -- GenAI Framework and Custom Metrics

**Duration:** 60 min
**Topics:**
- `mlflow.genai.evaluation` -- the full framework
- Creating evaluation datasets with `mlflow.genai.datasets`
- Built-in scorers vs. custom scorers
- `make_metric()` -- custom metric functions for LLM output quality
- Domain-specific LLM metrics (e.g., code quality scoring, instruction following, safety)
- Combining built-in and custom metrics
- Metric validation and thresholds
- Evaluation with `extra_metrics` and `custom_artifacts`
- LLM judges with custom criteria and rubrics
- Batch evaluation across multiple models/configurations

**Deliverables:**
- Custom metric suite for an LLM task, with evaluation framework comparing 3 different LLM configurations on a shared dataset

---

##### L1-M4.2.2 -- RAG System Evaluation

**Duration:** 60 min
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

##### L1-M4.2.3 -- Datasets and Human-in-the-Loop

**Duration:** 60 min
**Topics:**
- `mlflow.data` module -- logging datasets alongside runs
- Dataset constructors: `from_pandas()`, `from_numpy()`, `from_huggingface()`
- Dataset schemas, digests, and profiling
- Data lineage: `mlflow.log_input()` with context tags
- Running LLM inference on evaluation datasets
- `mlflow.log_table()` / `mlflow.load_table()` for results and labels
- `mlflow.genai.labeling` -- active labeling workflows
- Assessments: `mlflow.log_assessment()` for human feedback
- Combining automated + human evaluation
- Building ground truth datasets and feedback loops

**Deliverables:**
- Create datasets with schema inspection, run LLM inference, add human labels, combine automated and human evaluation, query lineage

---

#### L1-M4.3: Online

*Production traces, no ground truth, sampled coverage, the server pulls the
trigger.*

##### L1-M4.3.1 -- Online Scoring for LLM Applications

**Duration:** 60 min
**Topics:**
- Why offline evaluation is not enough: real users ask things your dataset never imagined
- `make_judge(...)` then `judge.register()` -- registration is what makes online scoring possible
- `scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=, filter_string=))`
- Nothing here is agent-specific: `scorer.start()` samples **traces**, so any traced
  LLM call qualifies -- a single `chat.completions.create()` is enough
- `scorer.update()`, `scorer.stop()`, and reading `ScorerStatus`
- Why sampling exists: judge cost scales with traffic, not with dataset size
- Reading assessments back off live traces and plotting a quality trend
- The four axes that separate online from offline: input, ground truth, coverage, trigger

**Deliverables:**
- A traced LLM app with a registered judge scoring a sampled share of its live traffic
- Quality trend assembled from online assessments, next to the offline score for the same app

---

### L1-M5: Prompt Registry and Management

#### L1-M5.1 -- Prompt Registry and Management

**Duration:** 45 min
**Topics:**
- Registering prompts: `mlflow.genai.register_prompt()`
- Prompt versioning and loading
- Prompt templates with variables
- Searching and managing prompts
- Prompt versioning strategies
- A/B testing prompts with MLflow
- Prompt templates with complex variables
- Team collaboration on prompts
- Prompt performance tracking over time

**Deliverables:**
- Prompt A/B test comparing 3 prompt variants with tracked metrics, using the prompt registry

---

### L1-M6: Deployment and Gateway

#### L1-M6.1 -- Model Serving

**Duration:** 60 min
**Topics:**
- `mlflow models serve` -- local REST API serving
- Serving endpoints: `/invocations`, `/ping`, `/version`
- Input formats: JSON, split-orient
- `mlflow models predict` -- batch prediction from CLI
- Serving a PyFunc-wrapped LLM model
- Custom request/response handling for chat interfaces
- Serving multiple model versions
- Health checks and monitoring
- Docker-based deployment: `mlflow models build-docker`

**Deliverables:**
- Serve an LLM model locally, call it via curl, demonstrate multi-version serving

---

#### L1-M6.2 -- Batch Prediction

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

#### L1-M6.3 -- AI Gateway

**Duration:** 60 min
**Topics:**
- What is the AI Gateway? (unified LLM endpoint management)
- Route configuration: providers, rate limits, fallbacks
- Supported providers: OpenAI, Anthropic, Mistral, Gemini, Bedrock, etc.
- Cost management and usage tracking
- When to use Gateway vs. direct API calls
- Provider routing: primary/fallback chains
- Load balancing across providers
- Budget limits and analytics

**Deliverables:**
- Gateway with multi-provider routing, fallbacks, and rate limits

---

### L1-M7: Optimization

*Evaluation measures; optimization changes the model and re-measures. It comes
last because it cannot exist without M4 -- every technique here is steered by a
scorer defined there. Two ways to change a model's behaviour: change its context
(M7.1) or change its weights (M7.2). The agent counterpart is L2-M3.*

#### L1-M7.1 -- Prompt Optimization

**Duration:** 60 min
**Topics:**
- Systematic prompt improvement workflow
- In-context learning optimization
- Few-shot example selection
- Tracking optimization history
- `mlflow.genai.optimize_prompts()` for automated prompt tuning, steered by an M4 scorer
- Why the prompt must be registered (M5) before it can be optimized

**Deliverables:**
- Optimized prompt with tracked improvement trajectory

---

#### L1-M7.2 -- Fine-Tuning: HuggingFace Transformers + MLflow

**Duration:** 60 min
**Topics:**
- `mlflow.transformers.autolog()` -- auto-logging for fine-tuning
- Fine-tuning a small LLM with training metrics tracking
- Logging checkpoints and model artifacts
- Model logging and loading HF models via MLflow
- Comparing base vs. fine-tuned models with evaluation metrics
- Pipeline serving via MLflow

**Deliverables:**
- Fine-tuning experiment with full tracking and base vs. fine-tuned model comparison

---

#### Level 1 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Tracking | 3 lessons | ~2.5 hours |
| M2: Tracing | 2 lessons | ~1.5 hours |
| M3: Models and Registry | 3 lessons | ~2.75 hours |
| M4: Evaluation (1 fundamentals, 3 offline, 1 online) | 5 lessons | ~5 hours |
| M5: Prompt Registry and Management | 1 lesson | ~0.75 hours |
| M6: Deployment and Gateway | 3 lessons | ~2.75 hours |
| M7: Optimization | 2 lessons | ~2 hours |
| **Total** | **19 lessons** | **~17.25 hours** |

---
---

## LEVEL 2 -- AI AGENTS

*Goal: Complete mastery of AI agent building, observability, evaluation and optimization with MLflow. Covers agent frameworks, custom integrations, agent-specific evaluation (offline, online, and standardized benchmarks), and optimization.*
*Prerequisite: Level 1 completed*
*LLM model: `google/gemma-4-26b-a4b`*
*Estimated time: ~22.5 hours (15 lessons)*

---

### L2-M1: Agent Frameworks

#### L2-M1.1 -- LangChain + LangGraph Agents

**Duration:** 90 min
**Topics:**
- Creating agents with LangChain v1+ (`create_agent` from `langchain.agents`)
- Tools with the `@tool` decorator (`langchain_core.tools`)
- Building the same agent by hand with LangGraph (`StateGraph`, nodes, edges, `ToolNode`)
- `create_agent` returns a compiled `StateGraph` -- one `mlflow.langchain.autolog()` call instruments both
- ReAct agent pattern and how it maps to MLflow traces
- Tracking tool calls, reasoning steps, and state transitions between nodes
- Conditional edge tracing (`add_conditional_edges`) and parallel node execution
- Agent middleware: `HumanInTheLoopMiddleware`, `TodoListMiddleware`
- Multi-agent patterns on LangGraph: collaboration, supervision, swarm
- Agent handoffs (`Command(goto=..., graph=Command.PARENT)`) and inter-agent trace analysis
- Comparing agent configurations (model, temperature, tools)
- Reference: `~/Projects/Github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent` and `.../2_langgraph/5_agent`

**Deliverables:**
- The same ReAct agent built twice -- `create_agent` and a hand-rolled `StateGraph` -- with both traces compared side by side
- Tool usage and state transition metrics, execution graph visualization

---

#### L2-M1.2 -- DeepAgents + MLflow

**Duration:** 90 min
**Topics:**
- DeepAgents architecture: `create_deep_agent()` built on top of `create_agent()`
- Built-in tools (filesystem, planning, sub-agent delegation via `task` tool)
- Sub-agents with isolated context windows
- Backends: `StateBackend`, `FilesystemBackend`, `CompositeBackend`
- Tracing multi-agent orchestration flows with MLflow
- Evaluating multi-agent collaboration quality
- Comparing DeepAgents sub-agent delegation vs. LangGraph shared-state multi-agent patterns
- Reference: `~/Projects/Github/langchain-ai/deepagents`

**Deliverables:**
- DeepAgents system with MLflow tracing
- Comparison with the LangGraph multi-agent approach from L2-M1.1

---

#### L2-M1.3 -- Claude Agent SDK + MLflow

**Duration:** 90 min
**Topics:**
- Claude Agent SDK architecture and lifecycle
- Building custom MLflow tracing for a framework with no native autolog
- Wrapping agent execution with `@mlflow.trace` and manual spans
- Logging agent decisions, tool calls, and outputs
- Custom autolog implementation for Claude Agent SDK
- External tools over an MCP server (STDIO transport)
- Reference code: `~/Projects/Github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- Source: `~/Projects/Github/anthropics/claude-agent-sdk-python`

**Deliverables:**
- Claude Agent SDK agent with full MLflow tracing and custom autolog wrapper
- Cost and duration metrics captured from a framework MLflow does not instrument

---

### L2-M2: Agent Evaluation

*Three groups, in the order you use them. **M2.1 Instruments** builds the
materials: a dataset, a set of judges, a metric suite. **M2.2 Offline** answers
"is this version good enough to ship?" -- against curated data you own, and
against public benchmarks you do not. **M2.3 Online** answers "is what shipped
still good?" -- against sampled production traces.*

*Benchmarking lives under Offline deliberately: a benchmark is an offline
evaluation whose dataset and metric are frozen and externally owned, so the
number means something to someone outside your team. Nothing else separates it.*

*Every lesson is a standalone leaf. Where two lessons need the same judge or
dataset, each carries its own copy -- no lesson imports from another.*

---

#### L2-M2.1: Instruments

##### L2-M2.1.1 -- Agent Test Generation and Simulation

**Duration:** 90 min
**Topics:**
- Hand-written test suites as the baseline, and where they stop scaling
- `mlflow.genai.test_agent()` -- self-description, test generation, simulation, issue discovery
- `guidance` and `num_test_cases` for steering what gets tested
- `ConversationSimulator` -- multi-turn simulation with `goal`, `persona`, `simulation_guidelines`
- `max_turns` and why single-shot test lists miss multi-turn failures
- `mlflow.genai.simulators.generate_test_cases()` -- distilling goal and persona from existing traces
- Promoting discovered issues into a versioned `mlflow.genai.create_dataset()`
- Regression baselines that survive across lessons

**Deliverables:**
- Auto-discovered issue list for a LangGraph agent, with failure analysis
- Versioned evaluation dataset reused by every later lesson in the module

---

##### L2-M2.1.2 -- Judges for Agents: Inline, Registered, Aligned

**Duration:** 90 min
**Topics:**
- Three ways to express the same rubric, and what each one costs you:
  - **Inline** -- `@scorer` + hand-built prompt + direct LLM call. Full control, no governance, dies with the script
  - **Registered** -- `make_judge(name, instructions, model=, base_url=)` then `judge.register(name=)`. Named, versioned, reusable, and the only form that can run online
  - **Built-in** -- `Correctness`, `Guidelines`, `RelevanceToQuery`, `Safety`, `ToolCallCorrectness`, `ToolCallEfficiency`
- Judge discovery and versioning: `list_scorers()`, `get_scorer(name, version=)`, `delete_scorer()`
- `ScorerKind` and the registration rule that follows from it: `@scorer` functions are `DECORATOR` kind and **cannot** be registered against a non-Databricks tracking URI (they deserialize via `exec()`); `make_judge` produces `INSTRUCTIONS` kind and registers fine against a local server
- Judge alignment: `judge.align(traces, optimizer)` -- correcting a judge against human labels instead of hand-tuning its prompt
- Alignment optimizers (DSPy / SIMBA / GEPA) and when alignment beats prompt editing
- Choosing a judge model through the LiteLLM gateway; judge cost as a first-class concern

**Deliverables:**
- One rubric implemented three ways (inline, registered, aligned), scored on the M2.1.1 dataset
- Disagreement table showing where the inline and aligned judges diverge
- A registered, versioned judge -- the pattern later lessons re-implement for themselves, since every lesson is a standalone leaf

---

##### L2-M2.1.3 -- Agent Quality Metrics and Session Scorers

**Duration:** 90 min
**Topics:**
- Designing metrics for agent-specific behaviors:
  - Task completion rate (binary + partial credit)
  - Tool selection accuracy (precision/recall/F1 of tool choices)
  - Reasoning quality (coherence, relevance, completeness)
  - Plan quality (for plan-and-execute agents)
- Composite scorers: combining sub-dimensions with explicit, tunable weights
- **Session-level scorers** -- the multi-turn dimension single-turn metrics cannot reach:
  `ConversationCompleteness`, `UserFrustration`, `ConversationalToolCallEfficiency`,
  `ConversationalRoleAdherence`, `KnowledgeRetention`
- `is_session_level_scorer` and why session scoring takes a different execution path
- Aggregation strategies across test cases; statistical significance for agent comparisons

**Deliverables:**
- Metric suite covering both single-turn and session dimensions
- Scores computed over M2.1.1's simulated conversations, not just single-shot cases

---

#### L2-M2.2: Offline

*Curated input, known ground truth, full coverage, and you pull the trigger.
Answers "is this version good enough to ship?" The first two lessons measure
against your own bar; the last three measure against everyone else's.*

##### L2-M2.2.1 -- Agent Architecture Comparison

**Duration:** 90 min
**Topics:**
- Systematic comparison of agent architectures:
  - Single-agent (`create_agent`) vs. custom `StateGraph` agents
  - Single-agent vs. multi-agent (swarm, supervision, collaboration)
  - LangChain/LangGraph agents vs. DeepAgents (`create_deep_agent`)
- Controlled evaluation methodology -- one dataset, one scorer set, one judge version
- Scoring through `mlflow.genai.evaluate()` with a registered judge at a pinned version, so
  results stay comparable outside the script that produced them
- Ablation studies: which component matters most?
- Cost-quality tradeoff analysis and the Pareto frontier
- Prompt sensitivity analysis

**Deliverables:**
- Comparison study with 3+ agent architectures on a shared dataset
- Cost-quality Pareto frontier visualization

---

##### L2-M2.2.2 -- Offline Gates and Regression Detection

**Duration:** 90 min
**Topics:**
- The offline pipeline end to end: dataset -> agent -> score -> gates -> report
- Dataset creation and versioning
- Multi-dimensional scoring (functional, quality, performance, cost)
- Quality gates and thresholds -- the build fails when a gate fails
- Regression detection against a stored baseline
- CI/CD integration (GitHub Actions or similar)
- The four axes that will separate this from M2.3: input, ground truth, coverage, trigger

**Deliverables:**
- Reproducible offline pipeline with quality gates wired into CI
- Regression report comparing a candidate agent against a stored baseline

---

##### L2-M2.2.3 -- SWE-Bench Evaluation

**Duration:** 90 min
**Topics:**
- SWE-Bench: the standardized benchmark for coding agents
- Setting up SWE-Bench Verified dataset from HuggingFace
- Building an agent that attempts SWE-Bench tasks
- Integrating SWE-Bench evaluation with MLflow tracking
- Logging per-instance results, pass rates, and error analysis
- Comparing agent configurations on SWE-Bench
- **No held-out split**: SWE-Bench Verified ships gold patches and the
  `FAIL_TO_PASS` / `PASS_TO_PASS` lists publicly in `split="test"`, so there is no
  clean half to optimize against. Why that makes published numbers hard to trust
- Reference: <https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified>

**Deliverables:**
- SWE-Bench evaluation pipeline integrated with MLflow
- Agent performance comparison across configurations
- Per-instance failure analysis logged as artifacts

---

##### L2-M2.2.4 -- GAIA Benchmark

**Duration:** 90 min
**Topics:**
- GAIA: General AI Assistants benchmark
- Setting up the GAIA dataset and evaluation harness
- `split="validation"` (answers public) vs. `split="test"` (answers withheld,
  scored by leaderboard submission) -- the contrast with SWE-Bench, and why a
  benchmark with a held-out half is the only kind you can safely optimize against
- Building an agent that handles GAIA tasks (web search, file manipulation, reasoning)
- Multi-step reasoning evaluation with MLflow tracking
- Comparing agent architectures on GAIA
- Analyzing failure modes by task category

**Deliverables:**
- GAIA evaluation pipeline integrated with MLflow
- Agent performance breakdown by task category
- Failure analysis and improvement recommendations

---

##### L2-M2.2.5 -- Custom Domain-Specific Benchmark

**Duration:** 90 min
**Topics:**
- Designing domain-specific evaluation benchmarks
- Dataset curation and quality assurance
- Metric design for domain-specific tasks
- **Designing in a held-out split from the start** -- a dev half you tune against
  and a test half you only ever report on, so the benchmark survives being
  optimized against (the lesson M2.2.3 and M2.2.4 teach the hard way)
- Baseline establishment and difficulty calibration
- Benchmark versioning and reproducibility
- Publishing and sharing benchmarks

**Deliverables:**
- Custom benchmark for a chosen domain (e.g., customer support, code review, data analysis)
- Benchmark suite with reproducible evaluation pipeline
- Documentation and baseline results

---

#### L2-M2.3: Online

*Production traces, no ground truth, sampled coverage, and the server pulls the
trigger. Answers "is what shipped still good?" Benchmarking has no counterpart
here -- live traffic has no frozen dataset and no expected answers.*

##### L2-M2.3.1 -- Online Scoring on Production Traces

**Duration:** 90 min
**Topics:**
- `scorer.register()` then `scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=, filter_string=))`
- Why only a registered judge can run online: `@scorer` functions are `DECORATOR`
  kind and cannot be registered against a non-Databricks tracking URI
- Assessments attaching to live traces; the server scheduler picking up active scorers
- `scorer.update()`, `scorer.stop()`, and reading `ScorerStatus`
- Why sampling exists: judge cost scales with traffic, not with dataset size
- Choosing `filter_string` to score the traffic that matters instead of all of it
- Reading quality trends back out of accumulated assessments
- Where the seam to Level 3 falls: this lesson produces the assessments, L3-M1
  consumes them in dashboards and alerts

**Deliverables:**
- A registered judge scoring a sampled live trace stream on a schedule
- Quality trend over time, assembled from online assessments
- The same agent seen both ways: gated offline in M2.2.2, monitored online here

---

### L2-M3: Agent Optimization

*Evaluation measures; optimization changes the system and re-measures. It comes
after evaluation because it cannot exist without it --
`optimize_prompts(..., scorers=[...])` takes a scorer as an input.*

*Only the first lesson has a real MLflow optimizer. For everything else MLflow's
role is to **track the search, not run it**: nested runs, one child per
configuration, scored by the same judge. That pattern is the transferable part,
and it works for any knob invented later.*

#### L2-M3.1 -- Prompt and Instruction Optimization

**Duration:** 90 min
**Topics:**
- The manual baseline: a hand-built grid over system prompts, tool descriptions and temperature
- `mlflow.genai.optimize_prompts()` -- automated instruction tuning against a scorer
- `predict_fn`, `prompt_uris`, `train_data`, `optimizer`, `scorers` -- what each argument controls
- Why the target prompt must live in the prompt registry and be applied with `PromptVersion.format`
- Tool description optimization and few-shot example selection
- Hyperparameter tuning: temperature, max_tokens, top_p
- The honest comparison: did automated optimization beat the hand-tuned grid, and at what token cost?

**Deliverables:**
- Optimized agent with a tracked improvement trajectory across iterations
- Manual grid vs. `optimize_prompts` compared on quality and spend

---

#### L2-M3.2 -- Agent Configuration Optimization

**Duration:** 90 min
**Topics:**
- The knobs MLflow has no optimizer for, and the one pattern that covers all of them:
  a tracked search over configurations, scored by a registered judge
- **Model selection** -- the highest-leverage knob in practice, swept through the gateway
- **Tool and MCP server budget** -- which servers and tools to expose at all;
  why fewer tools frequently beats more
- **Skills and subagents** -- delegation topology as a search space
- Nested runs as the search log: one parent per sweep, one child per configuration
- Reading a Pareto frontier over quality, latency and cost rather than a single winner
- Knowing when to stop: variance across repeats vs. the size of the improvement

**Deliverables:**
- A configuration sweep over models, tool budgets and delegation topology, fully tracked
- Pareto frontier identifying which configurations are worth their cost

---

#### L2-M3.3 -- Optimizing Against Benchmarks Without Destroying Them

**Duration:** 90 min
**Topics:**
- The trap: a benchmark you optimize against stops being a measurement and
  becomes training data
- Dev/test discipline -- optimize on the split you will never report
- GAIA as the clean case (`validation` public, `test` withheld) vs. SWE-Bench
  Verified as the contaminated one (everything public in `test`)
- Building your own held-out split when the benchmark does not provide one
- Detecting overfitting: the gap between dev score and held-out score
- Why leaderboard numbers routinely fail to reproduce in deployment
- Tracking which split every run was scored on, so the distinction survives review

**Deliverables:**
- An agent optimized on a dev split and reported on a held-out split
- The dev/held-out gap tracked across optimization iterations as an overfitting signal

---

#### Level 2 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Agent Frameworks | 3 lessons | ~4.5 hours |
| M2: Agent Evaluation (3 instruments, 5 offline, 1 online) | 9 lessons | ~13.5 hours |
| M3: Agent Optimization | 3 lessons | ~4.5 hours |
| **Total** | **15 lessons** | **~22.5 hours** |

---
---

## LEVEL 3 -- ADVANCED

*Goal: Production patterns, infrastructure, extensibility, and capstone projects. Full mastery of MLflow for production AI systems.*
*Prerequisite: Levels 1 and 2 completed*
*Estimated time: ~19 hours (11 lessons)*

---

### L3-M1: Production Operations

#### L3-M1.1 -- Production Tracing at Scale

**Duration:** 90 min
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

#### L3-M1.2 -- Grafana Dashboards for MLflow

**Duration:** 120 min
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

#### L3-M1.3 -- Feedback Loops and Continuous Improvement

**Duration:** 90 min
**Topics:**
- Collecting user feedback on agent responses
- `mlflow.log_assessment()` for production feedback
- Feeding production data back into evaluation datasets
- Identifying drift: prompt drift, data drift, quality drift
- Active learning: selecting the most informative examples for labeling
- Closing the loop: feedback to re-prompt to evaluate to deploy

**Deliverables:**
- Feedback collection pipeline with drift detection
- Active learning selection strategy

---

#### L3-M1.4 -- CI/CD for AI Applications

**Duration:** 90 min
**Topics:**
- Automated evaluation in CI pipelines (GitHub Actions)
- Quality gates: minimum metric thresholds for deployment
- Model validation before promotion
- Canary deployments with A/B evaluation
- Rollback strategies based on production metrics
- Environment promotion: dev to staging to production

**Deliverables:**
- GitHub Actions workflow with evaluation gates
- Canary deployment configuration with automated rollback

---

### L3-M2: Advanced Tracing

#### L3-M2.1 -- OpenTelemetry Integration

**Duration:** 60 min
**Topics:**
- MLflow's OpenTelemetry (OTel) foundation
- Exporting traces to OTel-compatible backends (Jaeger, Zipkin)
- Custom span processors and exporters
- Combining MLflow traces with infrastructure traces
- Distributed tracing across services

**Deliverables:**
- MLflow traces exported to an OTel-compatible backend

---

#### L3-M2.2 -- Temporal.io Workflow Tracing

**Duration:** 90 min
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

### L3-M3: Extensibility

#### L3-M3.1 -- Custom Autolog Integrations

**Duration:** 120 min
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

#### L3-M3.2 -- MLflow Plugins

**Duration:** 90 min
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

#### L3-M3.3 -- Enterprise Patterns and Data Management

**Duration:** 90 min
**Topics:**
- Workspace isolation and multi-tenancy
- Authentication and authorization at scale
- Experiment and model permissions by team
- Audit logging and secrets management
- High-availability MLflow server deployment
- Dataset versioning strategies at scale
- Data lineage across the full AI lifecycle
- Large-scale evaluation dataset management
- Data quality monitoring with MLflow
- Connecting evaluation datasets to runs to models to production

**Deliverables:**
- Multi-tenant MLflow configuration with team-based permissions
- Data lineage pipeline connecting evaluation datasets to agent runs to models to production metrics

---

### L3-M4: Capstones

#### L3-M4.1 -- Capstone: Production AI Agent Platform

**Duration:** 2.5 hours
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

#### L3-M4.2 -- Capstone: Agent Framework Benchmark

**Duration:** 2.5 hours
**Topics:**
- Build a standardized benchmark comparing agent frameworks:
  - LangChain agents
  - LangGraph agents
  - DeepAgents multi-agent systems
  - Claude Agent SDK agents
  - Custom PyFunc-wrapped agents
- Shared evaluation dataset and metrics
- SWE-Bench subset as standardized coding benchmark
- GAIA subset as general assistant benchmark
- Statistical analysis of results
- Cost-quality-latency comparison
- Recommendations for framework selection

**Deliverables:**
- Benchmark suite with reproducible results
- Framework comparison report
- Decision matrix for framework selection

---

#### Level 3 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Production Operations | 4 lessons | ~6.5 hours |
| M2: Advanced Tracing | 2 lessons | ~2.5 hours |
| M3: Extensibility | 3 lessons | ~5 hours |
| M4: Capstones | 2 projects | ~5 hours |
| **Total** | **11 lessons** | **~19 hours** |

---
---

## Complete Course Summary

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 -- Models** | Models/LLMs end-to-end | 19 lessons | ~17.25 hours |
| **Level 2 -- AI Agents** | Agent frameworks, evaluation, optimization | 15 lessons | ~22.5 hours |
| **Level 3 -- Advanced** | Production, extensibility, capstones | 11 lessons | ~19 hours |
| **Total** | | **45 lessons** | **~58.75 hours** |

---

### Project Structure

```text
tutorial/
├── syllabus.md                         # This file -- the master syllabus
├── level_1_models/
│   ├── M1_tracking/
│   │   ├── 1_tracking_fundamentals/
│   │   ├── 2_search_query_api/
│   │   └── 3_advanced_tracking/
│   ├── M2_tracing/
│   │   ├── 1_auto_manual_tracing/
│   │   └── 2_trace_analysis/
│   ├── M3_models_registry/
│   │   ├── 1_models_flavors_signatures/
│   │   ├── 2_custom_pyfunc/
│   │   └── 3_registry_workflows/
│   ├── M4_evaluation/
│   │   ├── 1_fundamentals/
│   │   │   └── 1_evaluation_fundamentals/
│   │   ├── 2_offline/
│   │   │   ├── 1_genai_custom_metrics/
│   │   │   ├── 2_rag_evaluation/
│   │   │   └── 3_datasets_human_in_loop/
│   │   └── 3_online/
│   │       └── 1_online_scoring/
│   ├── M5_prompt_registry/
│   │   └── 1_prompt_registry_management/
│   ├── M6_deployment_gateway/
│   │   ├── 1_model_serving/
│   │   ├── 2_batch_prediction/
│   │   └── 3_ai_gateway/
│   └── M7_optimization/
│       ├── 1_prompt_optimization/
│       └── 2_finetuning_huggingface/
├── level_2_agents/
│   ├── M1_agent_frameworks/
│   │   ├── 1_langchain_langgraph/
│   │   ├── 2_deepagents/
│   │   └── 3_claude_agent_sdk/
│   ├── M2_agent_evaluation/
│   │   ├── 1_instruments/
│   │   │   ├── 1_agent_testing/
│   │   │   ├── 2_judges/
│   │   │   └── 3_quality_metrics/
│   │   ├── 2_offline/
│   │   │   ├── 1_architecture_comparison/
│   │   │   ├── 2_offline_gates/
│   │   │   ├── 3_swe_bench/
│   │   │   ├── 4_gaia/
│   │   │   └── 5_custom_benchmark/
│   │   └── 3_online/
│   │       └── 1_online_scoring/
│   └── M3_agent_optimization/
│       ├── 1_prompt_instruction_optimization/
│       ├── 2_configuration_optimization/
│       └── 3_benchmark_optimization/
├── level_3_advanced/
│   ├── M1_production_operations/
│   │   ├── 1_production_tracing/
│   │   ├── 2_grafana_dashboards/
│   │   ├── 3_feedback_loops/
│   │   └── 4_cicd/
│   ├── M2_advanced_tracing/
│   │   ├── 1_opentelemetry/
│   │   └── 2_temporal_tracing/
│   ├── M3_extensibility/
│   │   ├── 1_custom_autolog/
│   │   ├── 2_plugins/
│   │   └── 3_enterprise_data/
│   └── M4_capstones/
│       ├── 1_agent_platform/
│       └── 2_framework_benchmark/
```

### MLflow Feature Coverage Matrix

| Feature Area | Level 1 (Models) | Level 2 (Agents) | Level 3 (Advanced) |
|---|---|---|---|
| Experiment Tracking | Fundamentals, search, nested runs, async | Agent run tracking | -- |
| System Metrics | Overview | -- | -- |
| Search/Query API | Fluent API + MlflowClient | -- | -- |
| Models & Flavors | LLM flavors, PyFunc, signatures | -- | Plugins |
| Model Registry | Full lifecycle, aliases, comparison | -- | Enterprise |
| Tracing (Auto) | OpenAI, LangChain, universal autolog | LangGraph, multi-agent | Production scale, custom autolog |
| Tracing (Manual) | Decorator, start_span, analysis | Custom framework tracing | OTel, Temporal |
| Evaluation -- offline | Fundamentals, GenAI framework, RAG eval, datasets | Comparison, offline gates, benchmarks | CI/CD gates |
| Evaluation -- online | Registered judge on sampled live traces | Same, on agent traces | Consumes the assessments |
| Human Evaluation | Labeling, assessments, ground truth | -- | Feedback loops |
| Prompt Engineering | Registry, versioning, A/B testing | -- | -- |
| GenAI Scorers/Judges | Built-in + custom scorers, LLM judges | Agent metrics, inline vs. registered judges, alignment, session scorers | -- |
| Data/Datasets | Logging, lineage, schema | Benchmarks (SWE-Bench, GAIA, custom) | Enterprise data management |
| AI Gateway | Multi-provider routing, fallbacks | -- | -- |
| Model Serving | CLI, Docker, multi-version | -- | -- |
| Batch Prediction | Pipelines | -- | -- |
| Optimization | Prompt optimization, fine-tuning | Instructions, tool/MCP budget, skills, subagents, model choice | -- |
| Benchmarking (offline eval, frozen external data) | -- | SWE-Bench, GAIA, custom domain; held-out splits | Framework comparison |
| Agent Tracking | -- | LangChain, LangGraph, multi-agent, Claude SDK, DeepAgents | -- |
| Agent Evaluation | -- | Instruments (simulation, judges, metrics), offline, online | Consumes online assessments |
| CI/CD | -- | Evaluation pipeline | Quality gates, canary |
| Grafana Monitoring | -- | -- | Dashboards, alerts |
| Plugins/Extensibility | -- | -- | Custom flavors, autolog, plugins |
| Enterprise Patterns | -- | -- | Multi-tenant, data lineage |
| Capstone Projects | -- | -- | 2 full projects |
