# L2-9.1 -- LLM Fine-Tuning with HuggingFace + MLflow

**Level:** Practitioner
**Duration:** ~1.5 hours

## Overview

This lesson walks through fine-tuning a small language model (distilgpt2, ~82M parameters) on a custom instruction dataset and tracking the entire process with MLflow. You will learn how `mlflow.transformers.autolog()` captures training metrics automatically, how to log model checkpoints as artifacts, and how to compare base vs fine-tuned outputs to evaluate whether fine-tuning improved the model.

## Prerequisites

- Completed: L1-M3.1 (Models and Flavors), L1-M2.1 (Autologging)
- MLflow server running at <http://127.0.0.1:5555>
- Internet connection (first run downloads distilgpt2, ~350 MB)

## Concepts

### Why Fine-Tune LLMs?

Pre-trained language models are general-purpose text generators. Fine-tuning adapts them to a specific domain or task by continuing training on a curated dataset. Even a few dozen examples can shift a model's behavior noticeably -- for instance, teaching it to answer Python programming questions in a consistent Q&A format rather than producing freeform text continuations.

### HuggingFace Trainer + MLflow Integration

The HuggingFace `Trainer` class is the standard way to fine-tune transformer models. MLflow integrates with it through `mlflow.transformers.autolog()`, which automatically logs:

- **Training metrics** -- loss, learning rate, and epoch at each logging step
- **Hyperparameters** -- batch size, learning rate, number of epochs, optimizer settings
- **Model information** -- architecture, parameter counts

This means you get a complete training record in MLflow without writing any extra logging code.

### Base vs Fine-Tuned Comparison

After fine-tuning, you should always compare the fine-tuned model against the base model on representative prompts. This comparison reveals whether training actually improved task-specific behavior or introduced regressions. MLflow makes it easy to log both outputs side by side in the same experiment.

## Step-by-Step

### Step 1: Load Base Model and Create Training Dataset

We load `distilgpt2` -- a small (82M parameter) causal language model that runs comfortably on CPU or Apple Silicon MPS. We then create a custom dataset of 20 Python Q&A pairs using `datasets.Dataset.from_dict()` and tokenize them for causal language modeling.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

examples = [
    "Q: What is a list in Python? A: A list is an ordered, mutable collection...",
    # ... 20 examples total
]
ds = Dataset.from_dict({"text": examples})
tokenized_ds = ds.map(tokenize_fn, batched=True)
```

### Step 2: Fine-Tune with Trainer and MLflow Autolog

We enable `mlflow.transformers.autolog()` before creating the Trainer. This instruments the training loop so that loss, learning rate, and epoch are logged to MLflow automatically at each `logging_steps` interval.

```python
mlflow.transformers.autolog(log_models=False)

args = TrainingArguments(
    output_dir=tmp_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()
```

We pass `log_models=False` to autolog because we log the model explicitly in Part 3 with a custom run name. The `report_to="none"` setting prevents the Trainer's own MLflow reporter from conflicting with our autolog setup.

### Step 3: Log the Fine-Tuned Model to MLflow

After training, we log the fine-tuned model as an MLflow artifact using `mlflow.transformers.log_model()`. This serializes both the model weights and the tokenizer into a single artifact that can be loaded later for inference.

```python
with mlflow.start_run(run_name="finetuned_model_artifact"):
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        name="finetuned_distilgpt2",
        task="text-generation",
        input_example=["Q: What is Python?"],
    )
```

### Step 4: Compare Base vs Fine-Tuned Generation

We generate text from both the original base model and the fine-tuned model on the same prompts. The fine-tuned model should produce more relevant, Q&A-formatted responses while the base model generates generic text continuations.

```python
for prompt in prompts:
    base_output = generate_text(base_model, tokenizer, prompt)
    finetuned_output = generate_text(finetuned_model, tokenizer, prompt)
```

Results are printed in a comparison table and the run is logged to MLflow.

## Running the Lesson

```bash
cd tutorial/level_2/M8_llm_finetuning/1_huggingface
uv sync
uv run python main.py
```

The first run downloads the distilgpt2 model (~350 MB). Subsequent runs use the cached version. Fine-tuning on CPU takes approximately 2-5 minutes depending on your hardware.

## Expected Output

```
============================================================
  L2-9.1: LLM Fine-Tuning with HuggingFace + MLflow
============================================================

============================================================
Part 1: Load Base Model and Create Training Dataset
============================================================
  Loaded tokenizer: distilgpt2 (vocab size 50257)
  Loaded model: distilgpt2 (82M parameters)
  Created training dataset: 20 examples, max_len=128

============================================================
Part 2: Fine-Tune with Trainer + MLflow Autolog
============================================================
  Enabled mlflow.transformers.autolog()
  Training config: 3 epochs, batch_size=2, lr=5e-05
  Starting fine-tuning...
  {'loss': 4.21, 'learning_rate': 3.3e-05, 'epoch': 1.5}
  {'loss': 3.58, 'learning_rate': 1.6e-05, 'epoch': 3.0}
  Fine-tuning complete.

============================================================
Part 3: Log Fine-Tuned Model to MLflow
============================================================
  Model URI: runs:/<run_id>/finetuned_distilgpt2
  Run ID: <run_id>

============================================================
Part 4: Compare Base vs Fine-Tuned Generation
============================================================
  Prompt                                   | Source     | Output
  ---
  Q: What is a list in Python? A:          | base       | I think the answer is ...
                                           | finetuned  | A list is an ordered, mutable collection ...
  ---
  ...

============================================================
Done! View results in the MLflow UI:
  http://127.0.0.1:5555/#/experiments
============================================================
```

In the MLflow UI at <http://127.0.0.1:5555>, you will see:
- Training metrics (loss, learning rate) plotted over steps under the autolog run
- The fine-tuned model artifact with tokenizer under the model logging run
- A comparison run with tags indicating base vs fine-tuned evaluation

## Key Takeaways

- **`mlflow.transformers.autolog()`** automatically captures training loss, learning rate, epoch, and hyperparameters from the HuggingFace Trainer -- no manual logging needed.
- **distilgpt2** is a practical choice for tutorial fine-tuning: small enough to train on CPU in minutes, large enough to demonstrate real behavior changes.
- **`mlflow.transformers.log_model()`** serializes both model weights and tokenizer into a single MLflow artifact for reproducible deployment.
- **Always compare base vs fine-tuned outputs** on representative prompts to verify that fine-tuning improved task-specific behavior.
- **Even 20 examples** with 3 epochs of fine-tuning can noticeably shift a model's output distribution toward the target format.

## Next Steps

In **Level 3**, you will explore advanced fine-tuning techniques including LoRA/QLoRA for parameter-efficient fine-tuning, evaluation-driven training loops with MLflow metrics as stopping criteria, and production deployment of fine-tuned models through the MLflow Model Registry.
