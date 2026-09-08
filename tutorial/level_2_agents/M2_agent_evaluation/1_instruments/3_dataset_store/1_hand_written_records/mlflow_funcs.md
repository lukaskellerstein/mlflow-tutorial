# MLflow Functions Used

Functions and imports from `mlflow` used in this lesson's `main.py`.

## Imports

- `from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset,
  search_datasets, set_dataset_tags`

## Function Calls

- `mlflow.genai.datasets.create_dataset()`
- `mlflow.genai.datasets.delete_dataset()` — by `dataset_id` outside Databricks
- `mlflow.genai.datasets.get_dataset()`
- `mlflow.genai.datasets.search_datasets()` — always with a filter
- `mlflow.genai.datasets.set_dataset_tags()` — merges, does not replace
- `mlflow.log_metrics()`
- `mlflow.log_params()`
- `mlflow.set_experiment()`
- `mlflow.set_tracking_uri()`
- `mlflow.start_run()`

## Objects and Methods

- `EvaluationDataset` — `.name`, `.dataset_id`, `.tags`, `.experiment_ids`,
  `.schema`, `.profile`
- `EvaluationDataset.merge_records()` — upsert on inputs
- `EvaluationDataset.to_df()` — adds `outputs`, `source`, `created_time`,
  `dataset_record_id`
- `EvaluationDataset.delete_records()` — takes `dataset_record_id` values
- `EvaluationDataset.has_records()`, `.set_profile()`

## Also available, not used here

- `mlflow.genai.datasets.delete_dataset_tag()`
- `mlflow.genai.datasets.add_dataset_to_experiments()` /
  `remove_dataset_from_experiments()` — one dataset, several experiments
