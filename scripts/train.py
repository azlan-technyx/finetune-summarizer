import os
import math
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "models/qwen2.5-1.5b-summarizer-lora"

MAX_LENGTH = 512

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ---------------------------------------------------------
# Device information
# ---------------------------------------------------------

def print_device_info():
    if torch.cuda.is_available():
        print("\nUsing CUDA GPU:")
        print(torch.cuda.get_device_name(0))
    else:
        print("\nCUDA GPU not detected.")
        print("Training will run on CPU.")

    print(f"PyTorch version: {torch.__version__}")
    print()


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

def load_tokenizer():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ---------------------------------------------------------
# Prepare one training example
# ---------------------------------------------------------

def preprocess_example(example, tokenizer):
    """
    Converts:

    {
        "document": "...",
        "summary": "..."
    }

    into Qwen chat-format tokens.

    The loss is calculated only on the assistant's summary.
    The prompt tokens are masked using -100.
    """

    document = example["document"]
    summary = example["summary"]

    system_message = (
        "You are a professional text summarization assistant. "
        "Produce a concise and factual summary of the provided document. "
        "Preserve important information and do not invent facts."
    )

    user_message = f"""Summarize the following document:

{document}"""

    # Prompt WITHOUT the answer
    prompt_messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # Prompt WITH the expected answer
    full_messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": summary,
        },
    ]

    # Tokenize prompt
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Add EOS if required
    if len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id:
        full_ids.append(tokenizer.eos_token_id)

    # Truncate
    full_ids = full_ids[:MAX_LENGTH]

    # Labels are initially identical to input
    labels = full_ids.copy()

    # We DON'T want the model learning the prompt.
    #
    # -100 tells PyTorch:
    # "Ignore this token when calculating the loss."
    prompt_length = min(len(prompt_ids), len(labels))

    labels[:prompt_length] = [-100] * prompt_length

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------
# Custom Data Collator
# ---------------------------------------------------------

class SummarizationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            padding_length = max_length - len(item["input_ids"])

            padded_input_ids = (
                item["input_ids"]
                + [self.tokenizer.pad_token_id] * padding_length
            )

            padded_attention_mask = (
                item["attention_mask"]
                + [0] * padding_length
            )

            # -100 means ignore padding while calculating loss
            padded_labels = (
                item["labels"]
                + [-100] * padding_length
            )

            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
            labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_data(tokenizer):
    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALIDATION_FILE,
        },
    )

    print()
    print(dataset)
    print()

    def tokenize(example):
        return preprocess_example(
            example,
            tokenizer,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


# ---------------------------------------------------------
# Load Qwen
# ---------------------------------------------------------

def load_model():
    print("Loading Qwen model...")

    if torch.cuda.is_available():

        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16")
        else:
            dtype = torch.float16
            print("Using float16")

    else:
        dtype = torch.float32
        print("Using float32 on CPU")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Required during training
    model.config.use_cache = False

    return model


# ---------------------------------------------------------
# Add LoRA
# ---------------------------------------------------------

def add_lora(model):
    print("\nAdding LoRA adapters...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,

        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,

        bias="none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print()
    model.print_trainable_parameters()
    print()

    return model


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

def train():
    print("=" * 60)
    print("Qwen2.5 Summarizer Fine-Tuning")
    print("=" * 60)

    print_device_info()

    tokenizer = load_tokenizer()

    dataset = load_data(tokenizer)

    model = load_model()

    model = add_lora(model)

    data_collator = SummarizationDataCollator(
        tokenizer
    )

    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Evaluation
        eval_strategy="epoch",

        # Saving
        save_strategy="epoch",
        save_total_limit=2,

        # Logging
        logging_steps=10,
        logging_first_step=True,

        # Optimizer
        optim="adamw_torch",
        weight_decay=0.01,

        # Learning-rate warmup
        warmup_ratio=0.05,

        # GPU precision
        fp16=(
            use_cuda
            and not torch.cuda.is_bf16_supported()
        ),

        bf16=(
            use_cuda
            and torch.cuda.is_bf16_supported()
        ),

        # We only need validation loss for now.
        # Avoid storing massive model predictions.
        prediction_loss_only=True,

        # Misc
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],

        data_collator=data_collator,
    )

    print("\nStarting training...\n")

    train_result = trainer.train()

    print("\nTraining finished.")

    print("\nSaving LoRA adapter...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    trainer.save_state()

    print(f"\nModel saved to:")
    print(OUTPUT_DIR)

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    print("\nRunning final validation...")

    metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")

    print("\nValidation metrics:")
    print(metrics)

    if eval_loss is not None:
        try:
            perplexity = math.exp(eval_loss)

            print(
                f"\nValidation perplexity: "
                f"{perplexity:.4f}"
            )

        except OverflowError:
            print(
                "\nPerplexity is too large "
                "to calculate safely."
            )

    print("\nDone.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    train()