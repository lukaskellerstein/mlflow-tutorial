---
globs: ["tutorial/**"]
---

# Tutorial Structure Rules

## Three-Level Architecture

The tutorial is organized in three progressive levels:

- **`tutorial/level_1/`** — Essentials (breadth): every major MLflow feature, short lessons (~30 min)
- **`tutorial/level_2/`** — Practitioner (depth): real-world projects, longer lessons (~1-2 hours)
- **`tutorial/level_3/`** — Expert (mastery): production patterns, agent evaluation, custom integrations

Always consult `syllabus.md` (project root) for the full module/lesson breakdown.

## Lesson Directory Convention

Every lesson lives in `tutorial/<level>/<module>/<lesson>/` and contains exactly:

1. **`pyproject.toml`** — standalone `uv` project. Use `[project]` with `name`, `version`, `description`, `requires-python`, and `dependencies`. Pin major versions only (e.g., `mlflow>=2.0`).
2. **`main.py`** — the working lesson code. This is the primary deliverable.
3. **`README.md`** — lesson guide (see `lesson-content.md` rule for format).
4. **`.gitignore`** — always ignore: `.venv/`, `__pycache__/`, `mlruns/`, `mlartifacts/`, `*.pyc`, `.python-version`.

## Directory Structure

```
syllabus.md                         # Master syllabus — source of truth (project root)
tutorial/
  level_1/
    M1_core_platform/
      1_architecture_overview/
      2_tracking_basics/
      3_search_query_api/
      4_system_metrics/
    M2_models_registry/
      1_models_flavors/
      2_model_registry/
      3_pyfunc/
    M3_autologging/
      1_traditional_ml/
      2_llm_genai/
    M4_evaluation/
      1_traditional_ml_eval/
      2_llm_eval_basics/
      3_llm_as_judge/
    M5_tracing/
      1_auto_tracing/
      2_manual_tracing/
    M6_genai_features/
      1_prompt_registry/
      2_scorers_judges/
      3_datasets_labeling/
    M7_data_datasets/
      1_dataset_logging/
    M8_deployment/
      1_model_serving/
      2_ai_gateway/
    M9_projects/
      1_mlflow_projects/
    M10_auth/
      1_auth_permissions/
  level_2/
    M1_advanced_tracking/
      1_nested_runs/
      2_async_batch_logging/
      3_artifact_management/
      4_mlflow_client/
    M2_advanced_models/
      1_signatures_deep_dive/
      2_custom_pyfunc/
      3_registry_workflows/
    M3_deep_evaluation/
      1_custom_metrics/
      2_rag_evaluation/
      3_genai_evaluation/
      4_human_in_loop/
    M4_advanced_tracing/
      1_langgraph_tracing/
      2_temporal_tracing/
      3_opentelemetry/
      4_trace_analysis/
    M5_agent_observability/
      1_langchain_agents/
      2_langgraph_agents/
      3_multiagent_systems/
    M6_prompt_engineering/
      1_prompt_management/
      2_prompt_optimization/
    M7_ai_gateway/
      1_gateway_routing/
    M8_deployment/
      1_serving_deep_dive/
      2_batch_prediction/
    M9_framework_integrations/
      1_pytorch/
      2_huggingface/
      3_sentence_transformers/
  level_3/
    M1_agent_evaluation/
      1_agent_testing/
      2_quality_metrics/
      3_architecture_comparison/
      4_agent_optimization/
      5_evaluation_pipeline/
    M2_custom_integrations/
      1_claude_agent_sdk/
      2_codex_sdk/
      3_deepagents/
      4_custom_autolog/
    M3_production/
      1_production_tracing/
      2_grafana_dashboards/
      3_feedback_loops/
      4_cicd/
    M4_advanced_features/
      1_plugins/
      2_enterprise/
      3_mcp/
      4_data_management/
    M5_capstones/
      1_agent_platform/
      2_framework_benchmark/
```

## pyproject.toml Template

```toml
[project]
name = "mlflow-tutorial-L<level>-<module>-<lesson>"
version = "0.1.0"
description = "<Lesson title from syllabus>"
requires-python = ">=3.10"

[project.dependencies]
mlflow = ">=2.0"
# Add lesson-specific deps here
```

## .gitignore Template

```
.venv/
__pycache__/
*.pyc
mlruns/
mlartifacts/
.python-version
```

## Principles

- Each lesson must be fully self-contained — a user should be able to `cd` into it, run `uv sync && uv run python main.py`, and see results.
- All lessons connect to the shared MLFlow server at `http://127.0.0.1:5000`. Set `MLFLOW_TRACKING_URI` in code, not env vars.
- Use `mlflow.set_experiment("L<level>/<module>/<lesson>")` so experiments are organized in the MLFlow UI.
- Print meaningful output to the console so the user sees what's happening without needing the MLFlow UI.
- Keep `main.py` under ~200 lines. If a lesson needs helper code, put it in a separate module within the same directory.
- Level 1 lessons should be concise and focused on a single concept.
- Level 2 lessons can be longer and build multi-step projects.
- Level 3 lessons should produce production-quality code and integrate multiple concepts.
