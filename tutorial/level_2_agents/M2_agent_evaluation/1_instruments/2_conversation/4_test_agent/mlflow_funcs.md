# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Function Calls

- `mlflow.genai.test_agent()` — describe, generate, simulate, discover
- `mlflow.langchain.autolog()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tag()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## Objects and Methods

- `AgentTestResult` — `.agent_description`, `.test_cases`, `.simulation_traces`,
  `.issues_result`
- `DiscoverIssuesResult` — `.issues`, `.summary`, `.total_traces_analyzed`,
  `.triage_run_id`, `.total_cost_usd`
- `Issue` — `.name`, `.description`, `.severity`, `.categories`, `.root_causes`,
  `.status`, `.source_run_id`
- `IssueSeverity` — `not_an_issue`, `low`, `medium`, `high`
