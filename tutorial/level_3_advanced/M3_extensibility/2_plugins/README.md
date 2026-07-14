# L3-4.1 — MLflow Plugins and Extensibility

**Level:** Expert
**Duration:** 1.5 hours

## Overview

MLflow's plugin system allows you to extend nearly every aspect of the platform -- from where data is stored to how models are evaluated. This lesson explores the plugin architecture, implements custom providers and evaluators, and demonstrates how plugins enhance experiment tracking in production environments.

## Prerequisites

- Completed: L1-M1 (Tracking), L2-M1 (Advanced Tracking), L2-M3 (Deep Evaluation)
- MLflow server running at http://127.0.0.1:5000
- LMStudio running with `google/gemma-4-26b-a4b` model loaded

## Concepts

### Why Plugins?

MLflow is designed to be extensible. The core platform ships with sensible defaults (local file store, standard metrics, built-in evaluators), but production deployments often need:

- Custom metadata injection (environment info, CI/CD context, team tags)
- Custom storage backends (proprietary object stores, specialized databases)
- Custom evaluation logic (domain-specific quality checks, compliance rules)
- Custom authentication (corporate SSO, token-based auth)

Rather than forking MLflow, the plugin system lets you inject this behavior through Python entry points.

### Plugin Discovery

MLflow discovers plugins via Python's standard `importlib.metadata.entry_points()` mechanism. When a package is installed that declares an entry point in a recognized group (e.g., `mlflow.run_context_provider`), MLflow automatically loads and registers it.

Two registration patterns exist:

1. **Scheme-based** (tracking store, artifact repository, model registry): the entry point name is the URI scheme. When you set `MLFLOW_TRACKING_URI=custom-scheme://...`, MLflow routes to the plugin's store implementation.
2. **List-based** (run context, request headers, authentication): all registered providers are iterated. The entry point name is ignored; only the class matters.

### Extension Points

| Entry Point Group | Purpose |
|---|---|
| `mlflow.tracking_store` | Custom tracking backends |
| `mlflow.artifact_repository` | Custom artifact storage |
| `mlflow.model_registry_store` | Custom model registry backends |
| `mlflow.run_context_provider` | Auto-tag runs at creation time |
| `mlflow.request_header_provider` | Custom HTTP headers on requests |
| `mlflow.request_auth_provider` | Custom authentication |
| `mlflow.model_evaluator` | Custom evaluation plugins |
| `mlflow.deployments` | Custom deployment targets |
| `mlflow.project_backend` | Custom execution backends |
| `mlflow.dataset_source` | Custom dataset sources |
| `mlflow.app` | Custom server applications |

## Step-by-Step

### Step 1: Plugin Architecture Overview (Part 1)

The lesson begins by listing all 13 MLflow extension point groups and explaining the two registration patterns. This establishes the conceptual foundation.

### Step 2: Custom RunContextProvider (Part 2)

We implement `EnvironmentContextProvider`, a subclass of `mlflow.tracking.context.RunContextProvider`. This provider automatically injects tags on every new run:

```python
class EnvironmentContextProvider(RunContextProvider):
    def in_context(self) -> bool:
        return True  # always active

    def tags(self) -> dict[str, str]:
        return {
            "env.hostname": socket.gethostname(),
            "env.python_version": platform.python_version(),
            "env.platform": platform.system(),
            "env.git_branch": _get_git_branch(),
            "env.timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
```

In production, you would publish this as a separate package with an entry point:

```toml
[project.entry-points."mlflow.run_context_provider"]
env_context = "my_plugin:EnvironmentContextProvider"
```

For this demo, we register it directly into MLflow's internal registry.

### Step 3: MetricAggregator (Part 3)

The `MetricAggregator` wraps `mlflow.log_metric()` to automatically compute rolling statistics (mean, min, max, stdev, p50, p95) alongside every raw metric. This is invaluable for monitoring metric distributions across training batches or inference rounds.

### Step 4: Custom Model Evaluator (Part 4)

`LLMOutputEvaluator` runs quality checks on LLM responses:

- **Character/word count** -- flags responses that are too short or too long
- **Code detection** -- identifies responses containing code blocks
- **Word diversity** -- ratio of unique words to total words (measures repetition)
- **Sentence count** -- structural complexity
- **Length quality flag** -- binary quality indicator

The evaluator is demonstrated by sending three prompts to the LLM and logging all evaluation metrics to MLflow.

### Step 5: Combined Demo (Part 5)

All plugins run together on a single LLM task, showing:
- Environment tags auto-injected by the context provider
- Latency tracked through the metric aggregator
- Response quality assessed by the evaluator

## Running the Lesson

```bash
cd tutorial/level_3/M4_advanced_features/1_plugins
uv sync
uv run python main.py
```

## Expected Output

```
Part 1: MLflow Plugin Architecture Overview
  (lists all 13 extension point groups)

Part 2: Custom Run Context Provider
  Registered EnvironmentContextProvider into MLflow registry.
  Auto-injected environment tags:
    env.architecture           = arm64
    env.git_branch             = main
    env.hostname               = my-machine
    env.platform               = Darwin
    env.python_version         = 3.12.0
    env.timestamp_utc          = 2026-06-23T...

Part 3: Custom Metric Logger (MetricAggregator)
  batch_latency_s:
    mean     = 0.9876
    min      = 0.1234
    ...

Part 4: Custom Model Evaluator Plugin
  Prompt 1: Explain what a Python decorator is...
  Response: A Python decorator is...
  Evaluation: chars=150, words=25, diversity=0.85, ...

Part 5: Combined Plugin Demonstration
  [Context Provider] Environment tags auto-injected on run creation.
  [Metric Aggregator] Logged latency: 1.2345s
  [Evaluator Plugin] Quality metrics logged.
```

In the MLflow UI, check:
- **Tags tab**: `env.*` tags on every run created after the provider was registered
- **Metrics tab**: `batch_latency_s/mean`, `batch_latency_s/p95`, `eval/*` metrics
- **Artifacts tab**: `evaluation_results.json` table

## Key Takeaways

- MLflow exposes 13+ extension point groups covering storage, context, auth, evaluation, and deployment.
- Plugins are discovered via Python entry points -- install a package and MLflow automatically loads it.
- `RunContextProvider` is the simplest plugin type: implement `in_context()` and `tags()` to auto-tag every run.
- Custom evaluators let you define domain-specific quality checks that run alongside standard metrics.
- The metric aggregation pattern (rolling stats alongside raw values) provides distribution visibility without external tooling.

## Next Steps

In **L3-4.2 (Enterprise Patterns)**, we apply these extensibility concepts to multi-tenant environments with governance workflows, audit logging, and cost tracking.
