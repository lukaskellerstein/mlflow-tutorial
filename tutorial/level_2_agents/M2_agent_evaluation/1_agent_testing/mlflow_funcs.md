# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`, plus the
hand-rolled `test_framework.py` it runs in Step 1.

## Imports

- `from mlflow.genai.simulators import ConversationSimulator`

## Function Calls

- `mlflow.genai.create_dataset()`
- `mlflow.genai.test_agent()`
- `mlflow.langchain.autolog()`
- `mlflow.log_artifact()` — via `test_framework.py`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tags()` — via `test_framework.py`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()` — including nested runs in `test_framework.py`

## Objects and Methods

- `ConversationSimulator(test_cases=, max_turns=, user_model=)` / `.simulate(predict_fn)`
- `AgentTestResult` — `.test_cases`, `.agent_description`, `.simulation_traces`, `.issues_result`
- `DiscoverIssuesResult` — `.issues`, `.summary`, `.total_traces_analyzed`, `.triage_run_id`
- `EvaluationDataset` — `.merge_records()`, `.dataset_id`, `.name`
