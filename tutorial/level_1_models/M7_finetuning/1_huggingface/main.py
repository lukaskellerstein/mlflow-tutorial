"""
L2-9.1 -- LLM Fine-Tuning with HuggingFace + MLflow

Fine-tunes distilgpt2 on a small instruction dataset and tracks everything
with MLflow: autolog for training metrics, model artifact logging, and
base vs fine-tuned comparison.
"""

import tempfile

import mlflow
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"
MAX_LEN = 128

# fmt: off
TRAIN_EXAMPLES = [
    "Q: What is a list in Python? A: A list is an ordered, mutable collection of items enclosed in square brackets.",
    "Q: How do you define a function? A: Use the def keyword followed by the function name and parentheses.",
    "Q: What does len() do? A: len() returns the number of items in a container such as a list or string.",
    "Q: What is a dictionary? A: A dictionary maps keys to values using curly braces, e.g. {'a': 1}.",
    "Q: How do you write a for loop? A: Use for item in iterable: followed by an indented body.",
    "Q: What is an f-string? A: An f-string lets you embed expressions inside string literals using f'...'.",
    "Q: What is None? A: None is Python's null value representing the absence of a value.",
    "Q: How do you import a module? A: Use import module_name or from module_name import something.",
    "Q: What is a class? A: A class is a blueprint for creating objects, defined with the class keyword.",
    "Q: What does append() do? A: append() adds a single element to the end of a list.",
    "Q: What is a tuple? A: A tuple is an ordered, immutable collection enclosed in parentheses.",
    "Q: How do you handle errors? A: Use try/except blocks to catch and handle exceptions.",
    "Q: What is a boolean? A: A boolean is True or False, used for logical conditions.",
    "Q: What is list comprehension? A: A concise way to create lists: [expr for item in iterable].",
    "Q: How do you read a file? A: Use open('file.txt') as a context manager with the with statement.",
    "Q: What is pip? A: pip is Python's package installer for downloading libraries from PyPI.",
    "Q: What does range() do? A: range() generates a sequence of integers for iteration.",
    "Q: What is slicing? A: Slicing extracts a portion of a sequence using [start:stop:step].",
    "Q: What is self? A: self refers to the current instance of a class inside its methods.",
    "Q: What is a lambda? A: A lambda is a small anonymous function: lambda x: x + 1.",
]
# fmt: on


def create_dataset(tokenizer) -> Dataset:
    """Tokenize the training examples for causal language modeling."""
    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = Dataset.from_dict({"text": TRAIN_EXAMPLES})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def generate_text(model, tokenizer, prompt: str, max_new: int = 40) -> str:
    """Generate text continuation from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def part1_load_model_and_data():
    """Load distilgpt2 and prepare the training dataset."""
    print("=" * 60)
    print("Part 1: Load Base Model and Create Training Dataset")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: vocab size {tokenizer.vocab_size}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {MODEL_NAME} ({num_params / 1e6:.0f}M parameters)")
    dataset = create_dataset(tokenizer)
    print(f"  Dataset ready: {len(dataset)} examples, max_len={MAX_LEN}\n")
    return model, tokenizer, dataset


def part2_finetune(model, tokenizer, dataset):
    """Fine-tune with HF Trainer and MLflow autolog."""
    print("=" * 60)
    print("Part 2: Fine-Tune with Trainer + MLflow Autolog")
    print("=" * 60)
    mlflow.transformers.autolog(log_models=False)
    print("  Enabled mlflow.transformers.autolog()")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="hf_finetune_"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        use_cpu=True,
    )
    print(f"  Config: {args.num_train_epochs} epochs, bs={args.per_device_train_batch_size}, lr={args.learning_rate}")
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    print("  Training started...")
    trainer.train()
    print("  Training complete.\n")
    return trainer


def part3_log_model(trainer, tokenizer):
    """Log the fine-tuned model as an MLflow artifact."""
    print("=" * 60)
    print("Part 3: Log Fine-Tuned Model to MLflow")
    print("=" * 60)
    with mlflow.start_run(run_name="finetuned_model_artifact") as run:
        info = mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            name="finetuned_distilgpt2",
            task="text-generation",
            input_example=["Q: What is Python?"],
        )
        mlflow.log_params({
            "base_model": MODEL_NAME, "num_epochs": 3,
            "learning_rate": 5e-5, "max_seq_length": MAX_LEN,
        })
        print(f"  Model URI: {info.model_uri}")
        print(f"  Run ID: {run.info.run_id}\n")


def part4_compare(base_model, finetuned_model, tokenizer):
    """Compare base vs fine-tuned generation on sample prompts."""
    print("=" * 60)
    print("Part 4: Compare Base vs Fine-Tuned Generation")
    print("=" * 60)
    prompts = [
        "Q: What is a list in Python? A:",
        "Q: How do you define a function? A:",
        "Q: What does len() do? A:",
        "Q: What is a dictionary? A:",
    ]
    print(f"\n  {'Prompt':<40} | {'Source':<10} | Output")
    print("  " + "-" * 95)
    with mlflow.start_run(run_name="base_vs_finetuned_comparison"):
        for prompt in prompts:
            base_out = generate_text(base_model, tokenizer, prompt)
            ft_out = generate_text(finetuned_model, tokenizer, prompt)
            base_ans = base_out[len(prompt):].strip().replace("\n", " ")[:60]
            ft_ans = ft_out[len(prompt):].strip().replace("\n", " ")[:60]
            print(f"  {prompt:<40} | {'base':<10} | {base_ans}")
            print(f"  {'':<40} | {'finetuned':<10} | {ft_ans}")
            print("  " + "-" * 95)
        mlflow.set_tag("comparison_type", "base_vs_finetuned")
        mlflow.log_param("num_prompts", len(prompts))
    print("\n  Fine-tuned model should produce more relevant Python Q&A responses.\n")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("L1/M7_finetuning/1_huggingface")

    print("\n" + "=" * 60)
    print("  L2-8.1: LLM Fine-Tuning with HuggingFace + MLflow")
    print("=" * 60 + "\n")

    # Part 1: Load model and data
    model, tokenizer, dataset = part1_load_model_and_data()
    # Save a copy of the base model for comparison later
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Part 2: Fine-tune (modifies model in-place)
    trainer = part2_finetune(model, tokenizer, dataset)
    # Part 3: Log fine-tuned model
    part3_log_model(trainer, tokenizer)
    # Part 4: Compare outputs
    part4_compare(base_model, trainer.model, tokenizer)

    print("=" * 60)
    print("Done! View results in the MLflow UI:")
    print("  http://127.0.0.1:5000/#/experiments")
    print("=" * 60)
