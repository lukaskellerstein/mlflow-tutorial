# MLFlow Tutorial: Three-Level Course

## Philosophy

This tutorial is structured in three domain-based levels:

- **Level 1 — Models**: Everything about models and LLMs in MLflow. Tracking, tracing, evaluation (offline and online), prompt registry, deployment, AI gateway, and optimization. Each topic is covered end-to-end so that a user finishing Level 1 has full command of MLflow for single-model workflows. Uses the `gemma-chat` alias for all lessons.
- **Level 2 — AI Agents**: Everything about AI agents. Assumes Level 1 knowledge. Covers agent frameworks (LangChain, LangGraph, multi-agent), custom integrations (Claude Agent SDK, DeepAgents), agent evaluation (instruments, offline including standardized benchmarks, online), and agent optimization. Uses the `gemma-agent` and `gemma-judge` aliases -- agents and the judges that grade them are named separately.
- **Level 3 — Advanced**: Production patterns, infrastructure, extensibility, and capstone projects. Ties together everything from Levels 1 and 2 into production-grade systems.

Each level builds on the previous. A user can stop after Level 1 and have complete mastery of MLflow for model/LLM workflows, continue through Level 2 for agent expertise, or go through Level 3 for production readiness.

## Target Audience

- **Level 1**: Anyone starting with MLflow for LLM work -- AI developers, ML engineers, data scientists working with language models
- **Level 2**: Practitioners building AI agent systems who need observability, evaluation, and optimization
- **Level 3**: Teams shipping AI agents to production who need monitoring, CI/CD, custom integrations, and enterprise patterns

## Technical Stack

- **Python**: 3.10+
- **Package Manager**: `uv` (every lesson is a standalone project)
- **MLFlow**: 3.x
- **LLM entry point**: the MLflow AI Gateway
  (`localhost:5555/gateway/mlflow/v1`, OpenAI-compatible). There is no separate
  proxy: the tracking server IS the gateway. Every lesson calls it and nothing
  else -- see "The gateway convention" below.
- **LLM provider behind it**: Unsloth Studio (local, GPU) — the only one
- **LLM aliases**:
  - `gemma-chat` -- the lesson's own LLM call, the thing under observation
  - `gemma-judge` -- LLM-as-judge, scorers, simulators
  - `gemma-agent` -- agent loops and tool calling
  - `gemma-tight` -- context-overflow demos. LiteLLM enforced a 7168-token guard
    here; the MLflow gateway has no equivalent, so overflow fails at the model
  - `gemma-31b-local` -- the denser local model
    The first five resolve to `unsloth/gemma-4-26B-A4B-it-qat-GGUF` today, and
    `gemma-31b-local` to the 31B -- see "The gateway convention".
  - `nomic-embed` / `text-embedding-3-small` -- the local embedding model
  - `gpt-4.1-mini` -- MLflow's aligner chat model, hardcoded by that name

  Eight aliases, three models, one provider. Every one is local and none has a
  fallback.
- **Agent Frameworks**: LangChain v1.0+, LangGraph, DeepAgents, Claude Agent SDK
- **Vector DB**: Qdrant (via Podman Compose)
- **Evaluation Benchmarks**: SWE-Bench, GAIA
- **Workflow Orchestration**: Temporal.io (via Podman Compose)
- **Observability**: Grafana + Prometheus (via Podman Compose)
- **Container runtime**: Podman (not Docker)

## The gateway convention

**Every lesson in all three levels talks to the MLflow AI Gateway, and nothing
else.** No lesson names a provider URL, a provider API key, or a raw model id.
It names an alias — `gemma-chat`, `gemma-judge`, `gemma-agent` — and the gateway
decides what that means.

The gateway is not a separate service. It is the same MLflow server the lesson
already logs its runs and traces to, which is why a lesson needs no second URL
and no key at all.

```python
GATEWAY_URL = "http://127.0.0.1:5555/gateway/mlflow/v1"
GATEWAY_KEY = "not-needed"  # this gateway has no keys
client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
client.chat.completions.create(model="gemma-chat", ...)
```

Two things live in `infra/mlflow/gateway/seed_gateway.py` rather than in lesson
code, and each is a decision the course would otherwise have to repeat in 50
places:

| Concern | Mechanism |
|:--|:--|
| Which model an alias resolves to | the `ENDPOINTS` entry and its `model` |
| What happens when it errors | `fallbacks` — an ordered chain, left to right |

MLflow has no config file of its own: its gateway lives in the tracking database
and arrives over an API. So compose runs a one-shot `mlflow-seed` on every
`up -d` that writes the aliases in. That script holds the alias list itself —
there is no separate YAML, because nothing but the script ever read one. Adding
an alias needs only `up -d`; **changing** one needs
`podman compose run --rm mlflow-seed --reset --prune`, because the seeder is
idempotent and reuses an endpoint it already has.

Aliases are named for the **job**, not the model size. `gemma-chat`,
`gemma-judge` and `gemma-agent` all resolve to one model today; the split
exists so that giving judges a stronger model later is one config line rather
than a sweep through forty lessons. A lesson that both runs an agent and judges
it names both, so the two can diverge without re-reading the lesson.

The teaching point is not the proxy. It is that **provider choice is
configuration, not code** — swapping a local model for a hosted one, or adding a
fallback, changes one file and every lesson follows.

Two consequences worth stating, because both surprise people:

- **An alias can never silently become a different model.** There is no hosted
  provider and no fallback chain, so a lesson that cannot reach Unsloth fails
  and says so. Aliases used to fall back to OpenRouter, which was right for a
  demo and wrong for a comparison — the optimization sweeps (L2-M3.2, L2-M3.3)
  had to name fixed cloud aliases to get a trustworthy result. They run on
  `gemma-agent` and `gemma-31b-local` now.
- **Server-side judges cannot use the constants above.** A scorer started with
  `scorer.start()` runs inside the MLflow server, on its own schedule, long after
  your script has exited — so it has no base URL to borrow. It names
  `gateway:/gemma-judge`, and the server already holds that endpoint because the
  seeder built it. This is the main simplification the single gateway bought:
  the lesson used to build a secret, a model definition and an endpoint by hand.
  L1-M4.3.1 and L2-M2.3.1.1 are the worked examples.
- **Embeddings are the one thing this gateway does not serve OpenAI-style.**
  `POST /gateway/mlflow/v1/embeddings` answers 404; the only alias-addressed
  route is `POST /gateway/<alias>/mlflow/invocations`. L1-M3.2 wraps it in
  fifteen lines. L2-M2.1.1.2 is the single place in the course that has to go
  around the gateway entirely, because MLflow's judge aligner insists on
  `{OPENAI_BASE_URL}/embeddings`.

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
*LLM alias: `gemma-chat`*
*Estimated time: ~17.25 hours (19 lessons)*

---

### L1-M1: Tracking

#### L1-M1.1 -- Tracking Fundamentals and Logging

**Duration:** 45 min
**Topics:**
- MLflow's pillars: Tracking, Models, Registry, Evaluation, Deployment
- Architecture: tracking server, backend store (PostgreSQL), artifact store
- Key concepts: experiments, runs, parameters, metrics, artifacts, tags
- Calling a local LLM through the MLflow AI Gateway (OpenAI-compatible API)
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
- `mlflow.openai.autolog()` -- trace OpenAI-compatible calls (the MLflow AI Gateway)
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

### L1-M6: Deployment

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
| M6: Deployment | 2 lessons | ~1.75 hours |
| M7: Optimization | 2 lessons | ~2 hours |
| **Total** | **18 lessons** | **~16.25 hours** |

---
---

## LEVEL 2 -- AI AGENTS

*Goal: Complete mastery of AI agent building, observability, evaluation and optimization with MLflow. Covers agent frameworks, custom integrations, agent-specific evaluation (offline, online, and standardized benchmarks), and optimization.*
*Prerequisite: Level 1 completed*
*LLM aliases: `gemma-agent`, `gemma-judge`*
*Estimated time: ~37.75 hours (33 lessons)*

---

### L2-M1: Agent Frameworks

*Six lessons in two groups, split by **scope** -- how much of an interaction one
run covers. `1_turn` runs one task from an empty message list, and meets each
framework. `2_conversation` runs four dependent turns in one session, and teaches
only what changes: where the state lives, and how MLflow groups the traces.*

*The same three frameworks appear in both groups, deliberately. Comparing one
framework across the two groups shows what memory costs; comparing the three
frameworks inside `2_conversation` shows that they disagree about who owns the
session key -- LangGraph and DeepAgents make you invent one, the Claude Agent SDK
hands you one after the turn is over.*

#### L2-M1.1: Turn -- one task, one answer

*Nothing is carried between tasks. Every run starts from an empty message list.*

##### L2-M1.1.1 -- LangChain + LangGraph Agents

**Duration:** 90 min
**Topics:**
- Creating agents with LangChain v1+ (`create_agent` from `langchain.agents`)
- Tools with the `@tool` decorator (`langchain_core.tools`)
- `create_agent` returns a compiled `StateGraph` -- one `mlflow.langchain.autolog()` call instruments the whole agent, node spans included
- Drawing the compiled graph the agent is made of (`agent.get_graph().draw_mermaid()`)
- ReAct agent pattern and how it maps to MLflow traces
- Tracking tool calls, reasoning steps, and state transitions between nodes
- Conditional edge tracing (`add_conditional_edges`) and parallel node execution
- Agent middleware: `HumanInTheLoopMiddleware`, `TodoListMiddleware`
- Multi-agent patterns on LangGraph: collaboration, supervision, swarm
- Agent handoffs (`Command(goto=..., graph=Command.PARENT)`) and inter-agent trace analysis
- Comparing agent configurations (model, temperature, tools)
- Reference: `~/Projects/Github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent` and `.../2_langgraph/5_agent`

**Deliverables:**
- One ReAct agent from `create_agent`, built by a single `build_agent()` and used by every part of the lesson
- Tool usage and state transition metrics, execution graph visualization from the compiled graph itself

---

##### L2-M1.1.2 -- DeepAgents + MLflow

**Duration:** 90 min
**Topics:**
- DeepAgents architecture: `create_deep_agent()` built on top of `create_agent()`
- Built-in tools (filesystem, planning, sub-agent delegation via `task` tool)
- Sub-agents with isolated context windows
- Backends: `StateBackend` and `FilesystemBackend` -- where a file lives once the turn ends, and how MLflow logs it
- Tracing multi-agent orchestration flows with MLflow
- Evaluating multi-agent collaboration quality
- Comparing DeepAgents sub-agent delegation vs. LangGraph shared-state multi-agent patterns
- Reference: `~/Projects/Github/langchain-ai/deepagents`

**Deliverables:**
- DeepAgents system with MLflow tracing
- Comparison with the LangGraph multi-agent approach from L2-M1.1.1

---

##### L2-M1.1.3 -- Claude Agent SDK + MLflow

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

#### L2-M1.2: Conversation -- many turns, one session

*The same three frameworks, now carrying state across a turn boundary. Every
lesson runs the same four-turn conversation, where turns 2 and 4 are unanswerable
without memory, and every lesson has the same four parts: run it, run a control
that removes the memory and fails, show the framework-specific twist, then read
the whole session back.*

*Two keys have to line up, and they belong to different systems. The framework's
thread key decides what the AGENT remembers; MLflow's `session_id` decides which
traces belong to ONE conversation. Setting both to the same value is what lets
one id be followed from the caller through the agent and into the trace store.*

*Requires MLflow 3.11+: `mlflow.search_sessions()`, and the `session_id` / `user`
arguments on `mlflow.update_current_trace()` and `mlflow.tracing.context()`.*

##### L2-M1.2.1 -- Multi-Turn Conversations with LangChain + LangGraph

**Duration:** 45 min
**Topics:**
- `thread_id` versus `session_id` -- two keys, two owners, one value
- LangGraph checkpointers: `InMemorySaver`, and what changes for `SqliteSaver` / `PostgresSaver`
- `create_agent(checkpointer=...)` -- one argument is the whole difference between a stateless agent and one with memory
- Sending only the new message; the checkpointer restores the rest
- Two ways to stamp a session: `mlflow.update_current_trace(session_id=...)` inside a `@mlflow.trace` function, versus `mlflow.tracing.context(...)` around a block
- `@mlflow.trace` serializes every argument -- why the agent is bound in a closure
- Reading a conversation back with `mlflow.search_sessions()`; `Session.id`, `len()`, iteration in time order

**Deliverables:**
- A four-turn conversation logged as one MLflow session, plus a control run on a fresh thread that provably loses the memory
- Two runnable scripts, `main_decorator.py` and `main_context.py`, that differ only in the stamping call -- everything they hold fixed lives in a shared `conversation.py`
- A side-by-side comparison of the two stamping APIs, including what each does to the trace previews

---

##### L2-M1.2.2 -- Multi-Turn Conversations with DeepAgents

**Duration:** 45 min
**Topics:**
- What a deep agent has to remember: messages, todos, and the virtual filesystem
- One `checkpointer=` argument carries all three, because all three are graph state
- Verifying persistence from `state["files"]` rather than by asking the model
- The backend decides the scope: `StateBackend` is per-thread, `FilesystemBackend` is per-directory
- A checkpointer can scope what it stores, and cannot scope a disk -- one thread reads another thread's file
- Isolating properly: `StateBackend`, or a per-session `root_dir`
- Sharing on purpose: `CompositeBackend` routes `/memories/` to a `StoreBackend`, scoped by a namespace you choose -- the same crossing as the leak, decided per path
- `write_todos` is offered, not forced -- why the todo channel is often empty on a local model

**Deliverables:**
- A four-turn conversation whose file survives every turn, with the file contents read from the graph state
- A demonstrated cross-thread file leak under `FilesystemBackend`, logged as the `cross_thread_read` metric
- A demonstrated per-path share under `CompositeBackend` and `StoreBackend`, logged as `shared_memory_read` = 1 next to `cross_thread_read` = 0

---

##### L2-M1.2.3 -- Multi-Turn Conversations with the Claude Agent SDK

**Duration:** 45 min
**Topics:**
- The inversion: the SDK assigns the session id, and reports it in `ResultMessage.session_id`
- Stamping a trace on the way out -- `update_current_trace` only has to run before the trace closes
- An open `ClaudeSDKClient` IS the conversation; a new client is a new session
- Why L2-M1.1.3 had no memory: one fresh client per query
- `ClaudeAgentOptions(resume=...)` -- continuing a session from a new client with no checkpoint store to run
- A resumed turn keeps the same session id, so MLflow files it into the original conversation
- Asserting on it: `distinct_session_ids` and `session_id_preserved` as metrics
- Cost control with `max_budget_usd`; no gateway and no local fallback

**Deliverables:**
- A four-turn conversation grouped under the id the SDK chose, with the control run proving a new client starts over
- A resumed fifth turn that lands in the same MLflow session

---

### L2-M2: Agent Evaluation

*Three groups, in the order you use them. **M2.1 Instruments** builds the
materials, and splits again by scope: three lessons on judging one turn, four on
judging a whole conversation, two on storing what they produce. **M2.2 Offline**
answers
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

*Nine lessons in three groups, split by **scope** -- what a scorer is allowed to
see, which is what decides what it can ask. `1_turn` judges one request and its
answer. `2_conversation` judges a whole discussion. `3_dataset_store` keeps what
both produce. Inside each group the lessons still climb the same ladder: how
much a human has to write, falling toward none.*

*Scope is not the same as pass/fail. **Ground truth** decides that: a case with
a right answer yields pass or fail, a case with only a rubric yields a score.
Both appear at both scopes.*

##### L2-M2.1.1: Turn -- one request, one answer

###### L2-M2.1.1.1 -- Hand-Written Agent Test Suites

**Duration:** 45 min
**Topics:**
- A test case with two halves: expected answer AND expected tool calls
- A hand-rolled runner, with one nested MLflow run per case
- Pass/fail reporting by difficulty; `mlflow.log_table()` for the results frame
- Regression baselines stored as a run artifact, not a file on disk
- Catching a real regression: the same agent shipped with a tool removed
- The three structural limits -- single-turn, only-what-you-imagined, linear cost

**Deliverables:**
- A working hand-rolled test harness, with a caught regression and a delta report
- A measured statement of the ceiling that motivates the rest of the group

---

###### L2-M2.1.1.2 -- Judges for Agents: Inline, Registered, Aligned

**Duration:** 90 min
**Topics:**
- Three ways to express one rubric, and what each costs you:
  - **Inline** -- `@scorer` + hand-built prompt. Full control, no governance
  - **Registered** -- `make_judge(...)` then `.register()`. The only form that can run online
  - **Built-in** -- `Correctness`, `Guidelines`, `RelevanceToQuery`, `Safety`
- `ScorerKind`: `@scorer` is `DECORATOR` kind and cannot be registered against a
  non-Databricks server; `make_judge` produces `INSTRUCTIONS` kind and registers fine
- Judge alignment: `judge.align(traces, optimizer)` against human labels
- **Alignment is turn-only.** `align()` raises `NotImplementedError` on a
  session-level scorer -- which is why judges live in the turn group
- `class Judge(Scorer)`: every judge is a scorer, and the test is `model=`

**Deliverables:**
- One rubric implemented three ways, with a disagreement table
- A registered, versioned judge

---

###### L2-M2.1.1.3 -- Turn-Level Quality Metrics

**Duration:** 90 min
**Topics:**
- Scope A, from `(inputs, outputs)`: task completion with partial credit,
  reasoning quality via an inline judge, a composite with visible weights
- Scope B, from a flattened dict: tool selection precision / recall / F1
- Scope B, from the TRACE: `ToolCallCorrectness` and `ToolCallEfficiency`,
  which see the arguments and the repeated call that a flattened list discards
- `ToolCallCorrectness` runs ground-truth-free by default
- Aggregation across cases; comparing two configurations side by side

**Deliverables:**
- A metric suite covering both turn scopes, over two agent configurations
- A named statement of what a turn scorer can never ask

---

##### L2-M2.1.2: Conversation -- many turns, one session

*One engine, three sources of goal, two kinds of verdict. The simulator only
ever runs forward -- goal to conversation. What changes between lessons is
where the goal came from.*

###### L2-M2.1.2.1 -- Conversation Simulation: the Engine

**Duration:** 45 min
**Topics:**
- `ConversationSimulator` -- `goal`, `persona`, `simulation_guidelines`, `max_turns`
- **`persona` is ONE user.** The simulator has exactly two sides, that user and
  your agent. There is no multi-party mode
- `goal` is the only required key; `context` and `expectations` are optional
- The `predict_fn` contract: `input` xor `messages`, readable return shapes,
  and `mlflow_session_id` for stateful agents
- One conversation = one session = several traces; `mlflow.search_sessions()`
- **A simulator test case is a scenario, not an assertion.** It cannot pass or
  fail. The internal "goal achieved?" check only decides when to stop

**Deliverables:**
- Multi-turn conversations traced per turn and grouped into sessions
- A stated boundary: this lesson produces traces and grades nothing

---

###### L2-M2.1.2.2 -- Goals and Personas Distilled from Real Sessions

**Duration:** 45 min
**Topics:**
- `mlflow.genai.simulators.generate_test_cases()` -- goal and persona inferred
  out of existing sessions, so production traffic writes the suite
- Why this is not a second execution direction: it produces a GOAL, which then
  runs forward through the same simulator
- Closing the loop: a distilled case fed straight back in
- The silent-failure trap: a session whose answer will not parse is DROPPED, so
  the function returns a shorter list rather than raising

**Deliverables:**
- Goals and personas distilled with no human input, then replayed
- The count-first reading pattern that makes a silent drop visible

---

###### L2-M2.1.2.3 -- Judging a Whole Conversation

**Duration:** 90 min
**Topics:**
- Session-level scorers -- the multi-turn dimension no turn scorer can reach
- All **seven** built-ins: `ConversationCompleteness`, `UserFrustration`,
  `KnowledgeRetention`, `ConversationalToolCallEfficiency`,
  `ConversationalRoleAdherence`, `ConversationalSafety`, `ConversationalGuidelines`
- Writing your own: a `@scorer` whose parameter is named `session` becomes
  session-level automatically, and needs no LLM at all
- `discover_issues()` -- judging in WORDS, with severity and root causes
- Three traps: every trace needs a `session_id`; traces export asynchronously;
  these judges answer with strings, and `bool("no")` is `True`
- **Polarity is not uniform.** `user_frustration` is good at 0.0, the rest at 1.0
- A judge that failed to parse its own answer is not a score of zero

**Deliverables:**
- One conversation scored by eight session scorers, logged under `session/<name>`
- A caught guideline violation, and a caught judge parse failure

---

###### L2-M2.1.2.4 -- test_agent(): the Whole Pipeline in One Call

**Duration:** 45 min
**Topics:**
- `mlflow.genai.test_agent()` -- describe, generate, simulate, discover
- It is a WRAPPER: it builds a `ConversationSimulator` and calls
  `discover_issues()`. It reuses M2.1.2.1 and M2.1.2.3, and never touches M2.1.2.2
- The order is description -> goals -> conversations, never the reverse
- `traces=` / `experiment_id=`: the description step falls back to reading
  existing traces when the agent cannot describe itself
- It returns **issues, never scores**. For numbers, run session scorers over
  `result.simulation_traces` yourself
- Why "0 issues" from an LLM judge is not proof

**Deliverables:**
- Auto-discovered issue list for an agent, from nothing but the agent
- A side-by-side of generated cases against hand-written ones

---

##### L2-M2.1.3: Dataset Store

*Where the cases live after the script that made them ends. Who wrote a record
decides what you can assert about it, and decides nothing about how it is stored.*

###### L2-M2.1.3.1 -- The Dataset Store: Hand-Written Records

**Duration:** 45 min
**Topics:**
- Why an agent dataset differs from L1-M4.2.3's model dataset: `inputs` is a
  message list, `expectations` name tools
- `mlflow.genai.create_dataset()`, `merge_records()`, `to_df()`, `delete_records()`
- `merge_records` as an upsert, and why that makes it CI-safe
- Versioning by tag: `set_dataset_tags()` merges, `delete_dataset_tag()` removes
- `search_datasets()` / `get_dataset()`, and why a bare search is dangerous
- The richest record shape -- answer AND route -- and why it is the one that
  yields pass or fail
- No LLM calls -- the lesson is about the store, not the agent

**Deliverables:**
- A versioned `support_agent_regression` dataset
- The upsert demonstrated by merging the same six records twice

---

###### L2-M2.1.3.2 -- The Dataset Store: Generated and Multi-Turn Records

**Duration:** 45 min
**Topics:**
- The thinner expectation shapes: route-only from a distilled goal, and empty
  from `test_agent` -- judged rather than compared
- The **multi-turn record**: `inputs` carrying a conversation already in
  progress, so a stored case can start at turn 3
- Merging four producers into one dataset with one call
- **Two kinds of test case, one store.** An evaluation record feeds
  `mlflow.genai.evaluate()`; a simulator scenario (`goal`, `persona`) feeds
  `ConversationSimulator`. They are not interchangeable, and the simulator
  raises when handed the wrong one

**Deliverables:**
- One dataset holding four record shapes, including multi-turn
- A runtime demonstration that the two kinds of `test_cases` cannot be swapped

---

#### L2-M2.2: Offline

*Curated input, known ground truth, full coverage, and you pull the trigger.
Answers "is this version good enough to ship?" The first two lessons measure
against your own bar; the last three measure against everyone else's.*

##### L2-M2.2.1: Turn -- one request, one answer

###### L2-M2.2.1.1 -- Agent Architecture Comparison

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

###### L2-M2.2.1.2 -- Offline Gates and Regression Detection

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

###### L2-M2.2.1.3 -- Comparing Two Versions of One Agent

**Duration:** 60 min
**Topics:**
- PAIRED comparison: one agent, two versions, the SAME cases in both
- Why that is a different question from M2.2.1.1's three architectures --
  independent means versus matched pairs
- WIN / LOSS / TIE per case, which a mean cannot give you: +0.05 is equally
  consistent with "six cases better" and "one much better, two worse"
- The SIGN TEST, and reading it honestly on the 6-20 cases a real suite has
- Deterministic scoring, because a judge adds variance to BOTH arms and on a
  small suite that noise swamps the effect
- Not to be confused with L2-M3.2: that SEARCHES a space, this DECIDES between
  two candidates you already have

**Deliverables:**
- A paired v1-vs-v2 report with per-case verdicts and a p-value
- A worked case where the mean improves and the verdict is still "not proven"

---

###### L2-M2.2.1.4 -- Reading an Evaluation: Slices and Failure Taxonomy

**Duration:** 60 min
**Topics:**
- No new metric -- two ways of reading results you already have
- SLICES: the aggregate lies. 0.75 overall can be 1.00 on three segments and
  0.00 on a fourth, and users experience their slice, not your mean
- Why slice SIZE decides whether a low slice is a lead or a finding
- A slice you did not LABEL cannot be analysed afterwards -- label before you
  need it
- TAXONOMY: cluster failures by cause, rank by frequency, read the cumulative
  share. "12 failed" is not actionable; "9 of 12 share one cause" is
- Ordering the classifier most-specific-first, so you count root causes rather
  than symptoms
- Simpson's paradox: an aggregate can move opposite to every slice when two
  runs have different slice mixes

**Deliverables:**
- A slice table exposing a segment the aggregate hid
- A ranked failure taxonomy that names the next piece of work

---

###### L2-M2.2.1.5 -- SWE-Bench Evaluation

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

###### L2-M2.2.1.6 -- GAIA Benchmark

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

###### L2-M2.2.1.7 -- Custom Domain-Specific Benchmark

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

##### L2-M2.2.2: Conversation -- many turns, one session

*Same pipeline as the turn branch, different unit. A support agent can answer
every individual turn correctly and still leave the customer unhelped -- and
"was the task ever finished?" is a property of the session, so a turn gate is
structurally unable to see it.*

###### L2-M2.2.2.1 -- Offline Gates on Whole Conversations

**Duration:** 60 min
**Topics:**
- The same gate pipeline with the UNIT changed: scenario -> agent -> session
  scorers -> threshold
- Gating on `ConversationCompleteness`, `KnowledgeRetention`, `UserFrustration`
- **Polarity in a gate**: `user_frustration` is good at 0.0 and the other two at
  1.0, so a gate that compares everything with `>=` passes the worst agent
- A judge that failed to parse contributes NOTHING -- scoring it 0.0 would fail
  the gate on a judge bug rather than on agent quality
- Baseline stored as a run tag, and regression read against it
- **Why one run is not a gate**: the user side is generated, so the same
  candidate scores differently twice. Mean over many, pass^k, or a tolerance band

**Deliverables:**
- A ship/block verdict computed from session scorers, with a frozen baseline
- A measured statement of run-to-run variance and what to do about it

---

###### L2-M2.2.2.2 -- Comparing Two Agent Versions Across Conversations

**Duration:** 60 min
**Topics:**
- The method depends entirely on WHO PLAYS THE USER:
  - **scripted** turns are fixed strings, both versions face the identical
    conversation, and pairing works exactly as at turn scope
  - **simulated** turns are generated and REACT, so v2 answers turn 1
    differently, the user asks a different turn 2, and there is nothing to pair
- Comparing distributions over k runs per version when pairing is unavailable
- Why 2 runs can never separate two versions: the spread is the size of the effect
- Reading a tie honestly -- session scorers answer yes/no, so they are COARSE,
  and a tie means the suite lacks RESOLUTION rather than the versions matching
- Scripted for the per-commit gate, simulated for the periodic sweep. Not rivals

**Deliverables:**
- The same question answered both ways, with the divergence made visible
- A stated rule for which method a given suite is entitled to use

---

###### L2-M2.2.2.3 -- Where Conversations Break Down: Funnel and Slices

**Duration:** 60 min
**Topics:**
- A session scorer returns ONE verdict for N turns -- enough to fail a build,
  useless for fixing it
- The FUNNEL: how many conversations are still healthy after turn 1, 2, 3 --
  and which turn is the cliff
- Why a funnel is not a per-turn pass rate: once a conversation breaks it stays
  broken, because the customer already had the bad experience
- Conversation-specific failure causes, `lost_earlier_context` above all
- Scoring EVERY TURN, not just the session -- score only the session and this
  analysis is impossible afterwards
- The cause here is a real production trade-off: a history window caps token
  cost and latency, and the funnel is how you measure what it cost

**Deliverables:**
- A funnel locating the turn where conversations die
- A ranked cause table, and a named trade-off behind the top cause

---

###### L2-M2.2.2.4 -- A Multi-Turn Agent Benchmark

**Duration:** 60 min
**Topics:**
- Why SWE-Bench and GAIA are single-turn, and what breaks when they are not
- The three fixes, after the tau-bench pattern:
  - **pin the user model in the benchmark spec**, not in the caller's config
  - **score the final STATE** of a database the agent had to mutate, not the text
  - **report pass^k**, not just pass@1
- A task that passes by NOT acting, so refusing correctly scores
- No LLM judge anywhere: the metric is a dict comparison, so re-scoring an old
  transcript gives the same answer forever
- What the benchmark still does not buy: comparability with another team, unless
  they run the same user model, simulator version and k

**Deliverables:**
- A working multi-turn benchmark with pass@1 and pass^k reported side by side
- The gap between those two numbers on one agent, and what it means

---

#### L2-M2.3: Online

*Production traces, no ground truth, sampled coverage, and the server pulls the
trigger. Answers "is what shipped still good?" Benchmarking has no counterpart
here -- live traffic has no frozen dataset and no expected answers.*

##### L2-M2.3.1: Turn -- one production trace

###### L2-M2.3.1.1 -- Online Scoring on Production Traces

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

###### L2-M2.3.1.2 -- A/B Testing Two Versions on Live Traffic

**Duration:** 45 min
**Topics:**
- Both versions serving AT THE SAME TIME against traffic nobody chose
- Why you cannot pair: request 7 is served by exactly one arm. Compare
  distributions, and rely on assignment being random with respect to the question
- The mechanism in three moves: assign per user, TAG the trace with the arm,
  let ONE registered judge score a sample of both and split by tag
- Two judges, or a judge retuned between arms, measures the judges not the agents
- **`hash()` is salted per process.** Bucketing on it silently reassigns users
  after every restart; `hashlib.md5` here is a STABILITY choice, not a security one
- What online buys that offline cannot: the traffic MIX is real, including the
  questions you would never have written

**Deliverables:**
- A live A/B with both arms scored by one judge and split by trace tag
- A stated reason why the result is or is not readable at this sample size

---

##### L2-M2.3.2: Conversation -- one production session

###### L2-M2.3.2.1 -- Online Session Scoring

**Duration:** 45 min
**Topics:**
- Sampling production SESSIONS rather than traces: "did this customer get
  helped?" instead of "was this reply good?"
- `Scorer.is_session_level_scorer` is the only thing that changes the server's
  behaviour -- it decides whether the judge is handed one trace or a list
- Built-in session scorers are `BUILTIN` kind, so they DO register against an
  open-source tracking server -- verified, not assumed
- **`.start()` demands a `gateway:/` model.** A scorer built with `openai:/`
  registers happily and then refuses to start
- **`delete_scorer(name=...)` is not enough** -- pass `version="all"`
- Cost grows with conversation length; signal arrives later, because a session
  cannot be judged until it looks finished

**Deliverables:**
- A registered, STARTED session-level scorer sampling live sessions
- A stated rule for sampling lower here than at turn level

---

###### L2-M2.3.2.2 -- A/B Testing on Live Sessions: Sticky Assignment

**Duration:** 45 min
**Topics:**
- The variant is chosen ONCE, when the session opens, and every turn of that
  session is served and tagged with it
- Why that is not a nicety: assign per request and turn 1 goes to v1 while turn
  2 goes to v2, so the conversation you score was produced by NEITHER version
- Noise averages out; a fabricated conversation does not. It is a measurement
  of a system you never shipped, counted against whichever arm you tagged it with
- Turn-scope A/B has no such failure mode, which is why this is a separate lesson
- Grouping sampled traces back into sessions by tag, then splitting by arm
- The two costs: prompt size grows with conversation length, and a session
  cannot be judged until the customer has stopped talking

**Deliverables:**
- A session-level A/B with sticky assignment demonstrated per session
- A named contrast with the turn-scope A/B on both cost and time-to-signal

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
| M1: Agent Frameworks (3 turn, 3 conversation) | 6 lessons | ~6.75 hours |
| M2: Agent Evaluation (9 instruments, 11 offline, 4 online) | 24 lessons | ~26.5 hours |
| M3: Agent Optimization | 3 lessons | ~4.5 hours |
| **Total** | **33 lessons** | **~37.75 hours** |

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
| **Level 1 -- Models** | Models/LLMs end-to-end | 18 lessons | ~16.25 hours |
| **Level 2 -- AI Agents** | Agent frameworks, evaluation, optimization | 30 lessons | ~35.5 hours |
| **Level 3 -- Advanced** | Production, extensibility, capstones | 11 lessons | ~19 hours |
| **Total** | | **59 lessons** | **~70.75 hours** |

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
│   ├── M6_deployment/
│   │   ├── 1_model_serving/
│   │   └── 2_batch_prediction/
│   └── M7_optimization/
│       ├── 1_prompt_optimization/
│       └── 2_finetuning_huggingface/
├── level_2_agents/
│   ├── M1_agent_frameworks/
│   │   ├── 1_turn/                            # one task, one answer
│   │   │   ├── 1_langchain_langgraph/
│   │   │   ├── 2_deepagents/
│   │   │   └── 3_claude_agent_sdk/
│   │   └── 2_conversation/                    # many turns, one session
│   │       ├── 1_langchain_langgraph/
│   │       ├── 2_deepagents/
│   │       └── 3_claude_agent_sdk/
│   ├── M2_agent_evaluation/
│   │   ├── 1_instruments/
│   │   │   ├── 1_turn/                        # one request, one answer
│   │   │   │   ├── 1_hand_written_tests/
│   │   │   │   ├── 2_judges/
│   │   │   │   └── 3_quality_metrics/
│   │   │   ├── 2_conversation/                # many turns, one session
│   │   │   │   ├── 1_conversation_simulation/
│   │   │   │   ├── 2_goals_from_real_sessions/
│   │   │   │   ├── 3_judging_conversations/
│   │   │   │   └── 4_test_agent/
│   │   │   └── 3_dataset_store/               # storage for both
│   │   │       ├── 1_hand_written_records/
│   │   │       └── 2_generated_records/
│   │   ├── 2_offline/
│   │   │   ├── 1_turn/                        # one request, one answer
│   │   │   │   ├── 1_architecture_comparison/     # your bar
│   │   │   │   ├── 2_offline_gates/
│   │   │   │   ├── 3_version_comparison/
│   │   │   │   ├── 4_failure_analysis/
│   │   │   │   ├── 5_swe_bench/                   # everyone's bar
│   │   │   │   ├── 6_gaia/
│   │   │   │   └── 7_custom_benchmark/
│   │   │   └── 2_conversation/                # many turns, one session
│   │   │       ├── 1_conversation_gates/          # your bar
│   │   │       ├── 2_version_comparison/
│   │   │       ├── 3_failure_analysis/
│   │   │       └── 4_multi_turn_benchmark/        # everyone's bar
│   │   └── 3_online/
│   │       ├── 1_turn/
│   │       │   ├── 1_online_scoring/
│   │       │   └── 2_live_ab_testing/
│   │       └── 2_conversation/
│   │           ├── 1_online_session_scoring/
│   │           └── 2_live_ab_testing/
│   └── M3_agent_optimization/
│       ├── 1_prompt_instruction_optimization/
│       ├── 2_configuration_optimization/
│       └── 3_benchmark_optimization/
└── level_3_advanced/
    ├── M1_production_operations/
    │   ├── 1_production_tracing/
    │   ├── 2_grafana_dashboards/
    │   ├── 3_feedback_loops/
    │   └── 4_cicd/
    ├── M2_advanced_tracing/
    │   ├── 1_opentelemetry/
    │   └── 2_temporal_tracing/
    ├── M3_extensibility/
    │   ├── 1_custom_autolog/
    │   ├── 2_plugins/
    │   └── 3_enterprise_data/
    └── M4_capstones/
        ├── 1_agent_platform/
        └── 2_framework_benchmark/
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
| AI Gateway | It IS the infra — every lesson's LLM entry point (`infra/mlflow/gateway/seed_gateway.py`) | Server-side judges name `gateway:/<alias>` | -- |
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
