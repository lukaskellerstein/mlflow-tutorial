# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.genai.simulators import ConversationSimulator, generate_test_cases`

## Function Calls

- `mlflow.genai.simulators.generate_test_cases()` — sessions back into goals
- `mlflow.langchain.autolog()`
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.search_sessions()` — groups traces by session id
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## Objects and Methods

- `ConversationSimulator(test_cases=, max_turns=, user_model=)` / `.simulate(predict_fn)`
  → `list[list[Trace]]`, one inner list per test case
- `Session` — `.id`, `.traces`, `len()`, iteration
- `Trace` — `.data.request`, `.data.response`
