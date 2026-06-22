# MLFlow Tutorial: From LLMs to AI Agents

## Overview
This comprehensive tutorial covers MLFlow's capabilities for tracking, evaluating, and monitoring Large Language Models (LLMs) and AI Agents. Each lesson is a hands-on project using Python with `uv` package manager, Gemini 4 (2B quantized) (via Ollama), and LangChain/LangGraph for agent frameworks.

## Target Audience
- Data Scientists and ML Engineers interested in LLM/Agent observability
- Developers building AI applications who need experiment tracking
- Teams looking to evaluate and optimize LLM-based systems

## Prerequisites
- Python 3.10+
- Basic understanding of LLMs and prompt engineering
- Familiarity with Python async/await patterns (for agents)
- Ollama installed with Gemini 4 (2B quantized) model (`ollama pull gemma4:e2b`)
- Basic knowledge of graph-based architectures (for LangGraph)

## Reference Code Examples
This tutorial builds upon and references the following code examples:
- **LangChain + LangGraph**: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai`
- **Temporal.io**: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python`
- **Claude Agent SDK**: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`
- **Codex SDK**: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`

---

## Module 1: MLFlow Fundamentals for LLMs

### 1.1 - Setup and First Steps
**Duration:** 30 minutes  
**Topics:**
- Installing MLFlow with `uv`
- Understanding MLFlow's architecture (tracking server, backend store, artifact store)
- Starting MLFlow UI locally
- Basic experiment tracking concepts

**Deliverables:**
- Working MLFlow server setup
- First experiment with basic logging

---

### 1.2 - Tracking LLM Calls
**Duration:** 45 minutes  
**Topics:**
- Auto-logging with `mlflow.langchain.autolog()` for LangChain v1.0+
- Manual tracking with `mlflow.start_run()`
- Logging parameters (temperature, max_tokens, model name)
- Logging metrics (latency, token count, cost estimation)
- Tracking prompts and completions with Ollama (Gemini 4 (2B quantized))

**Deliverables:**
- Simple chat application with Gemini 4 (2B quantized) (gemma4:e2b) tracked in MLFlow
- Comparison of different temperature settings

---

### 1.3 - Model Logging and Loading
**Duration:** 45 minutes  
**Topics:**
- Saving LLM configurations with `mlflow.pyfunc.log_model()`
- Using MLFlow Model Registry
- Loading and serving models
- Versioning prompts and model configurations

**Deliverables:**
- Logged prompt template as MLFlow model
- Loaded model inference script

---

## Module 2: LLM Evaluation with MLFlow

### 2.1 - Built-in Evaluation Metrics
**Duration:** 1 hour  
**Topics:**
- Understanding `mlflow.evaluate()` for LLMs
- Built-in metrics: perplexity, token_count, toxicity, flesch_kincaid_grade_level
- Creating evaluation datasets with pandas
- Model types: "question-answering", "text-summarization", "text"

**Deliverables:**
- Question-answering system evaluated on custom dataset
- Metrics comparison dashboard in MLFlow UI

---

### 2.2 - LLM-as-Judge Evaluation
**Duration:** 1 hour  
**Topics:**
- GenAI metrics: `answer_similarity`, `answer_correctness`, `faithfulness`
- Creating custom evaluation criteria with `EvaluationExample`
- Using ground truth for evaluation
- Interpreting judge scores and justifications

**Deliverables:**
- LLM judge evaluation pipeline
- Custom criteria for domain-specific evaluation

---

### 2.3 - RAG System Evaluation
**Duration:** 1.5 hours  
**Topics:**
- Building a RAG system with LangChain + Chroma
- Context-aware metrics: `faithfulness`, `relevance`, `context_recall`
- Evaluating retrieval quality
- Comparing different chunking strategies

**Deliverables:**
- RAG system with document embeddings
- Evaluation comparing retrieval configurations
- Faithfulness and relevance metrics tracked

---

## Module 3: MLFlow Tracing for LLM Applications

### 3.1 - Introduction to Tracing
**Duration:** 45 minutes  
**Topics:**
- What is tracing vs logging?
- Automatic tracing with `mlflow.langchain.autolog()`
- Understanding trace spans and parent-child relationships
- Viewing traces in MLFlow UI

**Deliverables:**
- Multi-step LLM chain with traced execution
- Trace visualization showing latency breakdown

---

### 3.2 - Manual Tracing and Custom Spans
**Duration:** 1 hour  
**Topics:**
- Using `@mlflow.trace` decorator
- Creating custom spans for business logic
- Tracing external API calls (Ollama)
- Adding metadata and tags to traces

**Deliverables:**
- Complex application with manual trace instrumentation
- Custom spans tracking preprocessing and postprocessing

---

### 3.3 - Tracing Multi-step Workflows
**Duration:** 1.5 hours  
**Topics:**
- **Option A - LangGraph**: Tracing LangGraph state machines with conditional flows
  - Sequential vs parallel node execution tracing
  - State transition visibility
  - Conditional edge tracing
  - _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph`_
- **Option B - Temporal.io**: Tracing durable workflows with MLFlow
  - Workflow and activity tracing
  - Long-running process observability
  - Retry and failure tracking
  - _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/my-workflows/temporal-io/my-python/MY/5_AI`_
- Debugging with traces (finding bottlenecks)
- Trace-based metrics aggregation

**Deliverables:**
- Multi-step workflow (LangGraph OR Temporal.io) with full MLFlow tracing
- Performance analysis identifying slow components
- State transition visualization (LangGraph) OR workflow execution timeline (Temporal.io)

---

## Module 4: AI Agent Observability

### 4.1 - Tracking Simple Agents
**Duration:** 1.5 hours  
**Topics:**
- What makes agents different from simple LLM calls?
- Creating agents with LangChain v1.0+ (`create_agent`)
- Auto-logging agents with `mlflow.langchain.autolog()`
- Tracking tool calls and reasoning steps
- Agent iteration and decision tracking
- _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/1_langchain/10_agent`_

**Deliverables:**
- ReAct agent with custom tools (calculator, mock search)
- Complete trace of agent reasoning loop
- Tool usage metrics and decision visualization

---

### 4.2 - LangGraph Agent Observability
**Duration:** 2 hours  
**Topics:**
- Introduction to LangGraph state machines and graphs
- Building agents with LangGraph (`StateGraph`, nodes, edges)
- Auto-tracing LangGraph with `mlflow.langchain.autolog()`
- Tracking state transitions between nodes
- Visualizing agent execution graphs (`visualize()`)
- Conditional branching and routing in traces
- _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/5_agent`_

**Deliverables:**
- LangGraph agent with conditional logic and multiple tools
- State transition traces with node execution details
- Graph visualization showing agent decision flow

---

### 4.3 - Multi-Agent System Tracing with LangGraph
**Duration:** 2.5 hours  
**Topics:**
- Multi-agent patterns in LangGraph (collaboration, supervision, swarm)
- Building multi-agent graphs with agent handoffs
- Tracing inter-agent communication and state sharing
- Using `create_react_agent` for individual agents in a graph
- Aggregating metrics across multiple agents
- Debugging agent collaboration failures
- _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/ai-agents-course/Version_2/6_langchain-ai/2_langgraph/6_agents`_

**Deliverables:**
- Multi-agent system with 2-3 collaborating agents (e.g., researcher + summarizer + coder)
- Complete trace showing agent handoffs and collaboration
- Per-agent performance metrics and bottleneck identification
- Comparison of supervision vs swarm patterns

---

## Module 5: Advanced Agent Evaluation

### 5.1 - Agent Testing with MLFlow GenAI
**Duration:** 1.5 hours  
**Topics:**
- Introduction to `mlflow.genai.agent_tester`
- Automated test case generation for agents
- Simulating user interactions
- Success criteria definition

**Deliverables:**
- Agent test suite with automated scenarios
- Test results dashboard
- Failure analysis reports

---

### 5.2 - Agent Quality Metrics
**Duration:** 1.5 hours  
**Topics:**
- Task completion rate
- Tool selection accuracy
- Reasoning quality evaluation
- Response time and efficiency metrics
- Custom agent-specific metrics

**Deliverables:**
- Comprehensive agent evaluation framework
- Metrics comparing different agent configurations
- Quality regression testing pipeline

---

### 5.3 - Agent Optimization with MLFlow
**Duration:** 2 hours  
**Topics:**
- Using `mlflow.genai.optimize` for agent tuning
- Comparing agent architectures (ReAct vs Plan-and-Execute)
- Prompt optimization for agent instructions
- Tool selection optimization
- A/B testing agents in MLFlow

**Deliverables:**
- Optimized agent configuration
- Comparison study of agent variants
- Best practices documentation

---

## Module 6: Production Patterns

### 6.1 - Deployment and Serving
**Duration:** 1.5 hours  
**Topics:**
- Model serving with `mlflow models serve`
- REST API endpoints for LLM/agent inference
- Batch prediction workflows
- Model versioning and rollback strategies

**Deliverables:**
- Deployed LLM service with REST API
- Batch inference script
- Version management workflow

---

### 6.2 - Monitoring Production Agents
**Duration:** 1.5 hours  
**Topics:**
- Real-time trace collection
- Setting up alerts on metrics
- Performance degradation detection
- Cost tracking and optimization
- User feedback integration

**Deliverables:**
- Production monitoring dashboard
- Alert configuration for agent failures
- Cost analysis report

---

### 6.3 - CI/CD for LLM Applications
**Duration:** 1 hour  
**Topics:**
- Automated evaluation in CI pipelines
- Regression testing with MLFlow
- Model validation gates
- Promoting models through environments (dev → staging → prod)

**Deliverables:**
- GitHub Actions workflow (or similar) for LLM testing
- Automated evaluation reports
- Promotion criteria checklist

---

## Bonus Module: Advanced Topics

### B.1 - Fine-tuning Tracking
**Duration:** 1.5 hours  
**Topics:**
- Tracking fine-tuning experiments with MLFlow
- Using Hugging Face Transformers with `mlflow.transformers.autolog()`
- PyTorch training loop integration with MLFlow
- Logging training metrics, loss curves, and checkpoints
- Comparing base vs fine-tuned Gemini models (if fine-tuning 2B)
- Dataset versioning and artifact logging for fine-tuning
- Model versioning and A/B comparison

**Deliverables:**
- Fine-tuning experiment with Hugging Face Transformers (or PyTorch)
- Complete tracking of training metrics and model checkpoints
- Comparison dashboard of fine-tuned variants vs base model

---

### B.2 - Custom Integration Patterns
**Duration:** 2 hours  
**Topics:**
- Integrating MLFlow with modern agent frameworks:
  - **Claude Agent SDK** (Anthropic's agent framework)
    - _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/5_Claude_Agent_SDK/python`_
  - **Codex SDK** (code generation agents)
    - _Reference: `/Users/lkellers/Projects/github/lukaskellerstein/vibe-coding-course/3_Codex_SDK/typescript`_
  - **DeepAgents** (multi-agent orchestration)
- Custom autolog implementations for unsupported frameworks
- Building MLFlow plugins for custom metrics
- Integration with Grafana for MLFlow metrics dashboards
  - Exporting MLFlow metrics to Prometheus
  - Creating Grafana dashboards for agent performance
  - Setting up alerts on agent quality metrics

**Deliverables:**
- Integration example with Claude Agent SDK OR Codex SDK
- Grafana dashboard showing MLFlow agent metrics
- Custom autolog wrapper for a framework
- Team documentation for adopting MLFlow with these tools

---

## Project Structure

Each lesson follows this structure:
```
tutorial/
├── 1_fundamentals/
│   ├── 1_setup/
│   │   ├── pyproject.toml (uv project)
│   │   ├── README.md (lesson guide)
│   │   ├── main.py (working code)
│   │   └── .gitignore
│   ├── 2_tracking/
│   │   ├── ...
│   └── 3_model_logging/
│       └── ...
├── 2_evaluation/
│   ├── 1_builtin_metrics/
│   ├── 2_llm_judge/
│   └── 3_rag_evaluation/
├── 3_tracing/
├── 4_agents/
├── 5_advanced_evaluation/
├── 6_production/
└── bonus/
```

## Expected Learning Outcomes

After completing this tutorial, you will be able to:

1. **Track and version** LLM experiments systematically with MLFlow
2. **Evaluate** LLM outputs using both automated metrics and LLM-as-judge
3. **Trace** complex multi-step LLM applications (LangChain, LangGraph, Temporal.io) and identify bottlenecks
4. **Monitor** AI agents including reasoning steps, tool usage, and state transitions
5. **Build and trace** multi-agent systems with LangGraph (collaboration, supervision, swarm patterns)
6. **Test and optimize** agent behavior systematically with MLFlow GenAI tools
7. **Deploy** LLM applications with proper observability and production monitoring
8. **Integrate** MLFlow with modern agent frameworks (Claude SDK, Codex, DeepAgents)
9. **Build** production-ready AI systems with quality gates and Grafana dashboards

## Estimated Total Time
- Core modules (1-5): ~21 hours
- Production patterns (6): ~4 hours
- Bonus modules: ~3.5 hours
- **Total: ~28-30 hours** (self-paced)

## Technical Stack

- **MLFlow**: Latest version (2.x+)
- **Python**: 3.10+
- **Package Manager**: `uv`
- **LLM**: Gemini 4 (2B quantized) via Ollama (`gemma4:e2b` - local, no API costs)
- **Agent Framework**: LangChain v1.0+ + LangGraph (latest)
- **Vector DB**: Chroma (for RAG examples)
- **Multi-agent**: LangGraph multi-agent patterns (Module 4.3)
- **Workflow Orchestration** (optional): Temporal.io (Module 3.3)
- **Fine-tuning**: Hugging Face Transformers + PyTorch (Bonus Module)
- **Integration Examples**: Claude Agent SDK, Codex SDK, DeepAgents (Bonus Module)
- **Observability**: Grafana (for production monitoring)

## Next Steps

Once you approve this syllabus, I will:
1. Create the complete folder structure
2. Implement each lesson with working code
3. Add detailed README.md for each lesson
4. Include solution code and expected outputs
5. Add troubleshooting guides

---

## Key Updates Based on Your Requirements

✅ **LangChain v1.0+**: All examples use latest LangChain (no deprecated LCEL chains)  
✅ **LangGraph Latest**: Multi-agent patterns using modern LangGraph (supervision, swarm, collaboration)  
✅ **Gemini 4 (2B quantized)**: Using `gemma4:e2b` via Ollama throughout  
✅ **Temporal.io Option**: Module 3.3 includes Temporal.io workflow tracing (based on your code examples)  
✅ **Multi-agent with LangGraph**: Module 4.3 uses LangGraph multi-agent (no CrewAI)  
✅ **Modern Frameworks**: Claude Agent SDK, Codex SDK, DeepAgents integration (Bonus B.2)  
✅ **Grafana Only**: Production monitoring focuses on Grafana dashboards (no Datadog)  
✅ **Fine-tuning**: Hugging Face Transformers + PyTorch (Bonus B.1)  
✅ **Vector DB**: Chroma for all RAG examples  

---

**Questions for Review:**

1. Is the progression from LLMs → Agents → Multi-agents appropriate for your learning goals?
2. For Module 3.3, do you prefer **LangGraph** or **Temporal.io** for multi-step workflow tracing? (Or implement both as options?)
3. Should I prioritize more multi-agent patterns (hierarchical, sequential, etc.) in Module 4.3?
4. Are the time estimates realistic for your schedule?
5. Any specific agent use cases to focus examples on (e.g., research agents, code review, data analysis)?
