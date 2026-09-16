# Qwen2.5 Fine-Tuned Summarizer

A learning project for fine-tuning **Qwen2.5-1.5B-Instruct** for text summarization using **PyTorch, Hugging Face Transformers, and LoRA**.

The goal is to train Qwen to convert longer source documents into concise, factual summaries while preserving important information and avoiding hallucinations.

---

## Base Model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model is fine-tuned using **LoRA (Low-Rank Adaptation)** rather than updating all 1.5 billion model parameters.

---

## Current Project Structure

```text
qwen-summarizer/
│
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── scripts/
│   └── train.py
│
├── models/
│
├── results/
│
├── .gitignore
└── README.md
```

Dataset sizes:

```text
train.jsonl        500 examples
validation.jsonl    60 examples
test.jsonl          60 examples
```

---

# 1. Prerequisites

Recommended:

```text
Ubuntu 22.04 / 24.04
Python 3.10+
Git
Internet connection
```

Current development environment:

```text
Ubuntu 24.04
Python 3.12
CPU training
```

An NVIDIA CUDA GPU is not required, but training is significantly faster with one.

---

# 2. Clone the Repository

Using SSH:

```bash
git clone git@github.com:saadalikhan02/finetune-summarizer.git
```

If using a custom SSH alias such as `github-work`:

```bash
git clone git@github-work:saadalikhan02/finetune-summarizer.git
```

Then:

```bash
cd finetune-summarizer
```

---

# 3. Create a Python Virtual Environment

Create:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Your terminal should now look similar to:

```text
(.venv) saad@machine:~/apps/AI-Engineering/qwen-summarizer$
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# 4. Install Dependencies

Install the packages required for training:

```bash
pip install torch transformers datasets peft accelerate
```

Verify installation:

```bash
python -c "import torch, transformers, datasets, peft, accelerate; print('Everything installed successfully')"
```

Expected output:

```text
Everything installed successfully
```

Check PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

Check whether CUDA is available:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

On a CPU-only machine:

```text
CUDA available: False
```

This is fine. The training script automatically falls back to CPU.

---

# 5. Dataset Format

Each dataset file uses **JSONL**.

JSONL means:

```text
one JSON object per line
```

Example:

```json
{"document":"The engineering team completed the payment integration. QA identified two callback issues that are currently being investigated.","summary":"Payment integration is complete, with two callback issues currently under investigation."}
```

Every row contains:

```text
document = input text
summary  = expected model output
```

---

# 6. Dataset Split

The project uses three separate datasets.

## Training

```text
data/train.jsonl
```

Used by the model to learn.

Current size:

```text
500 examples
```

---

## Validation

```text
data/validation.jsonl
```

Used during training to measure whether the model is improving.

Current size:

```text
60 examples
```

---

## Test

```text
data/test.jsonl
```

Reserved for final evaluation.

Current size:

```text
60 examples
```

The test dataset must **not** be used during training.

Conceptually:

```text
train.jsonl
     ↓
Model learns

validation.jsonl
     ↓
Monitor training

test.jsonl
     ↓
Final unbiased evaluation
```

---

# 7. Validate the JSONL Files

Before training, verify that the files contain valid JSON.

Run:

```bash
python - <<'PY'
import json

files = [
    "data/train.jsonl",
    "data/validation.jsonl",
    "data/test.jsonl",
]

for path in files:
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            json.loads(line)
            count += 1

    print(f"{path}: {count} valid records")
PY
```

Expected output:

```text
data/train.jsonl: 500 valid records
data/validation.jsonl: 60 valid records
data/test.jsonl: 60 valid records
```

---

# 8. Fine-Tuning Method

The project uses:

```text
Qwen2.5-1.5B-Instruct
        +
       LoRA
        ↓
Fine-tuned summarizer
```

Instead of modifying all Qwen parameters:

```text
Original Qwen parameters
████████████████████████████████
Frozen

LoRA parameters
██
Trainable
```

This reduces:

- memory usage
- training cost
- storage requirements
- training time

---

# 9. Current Training Configuration

The current `scripts/train.py` uses approximately:

```text
Model:
Qwen/Qwen2.5-1.5B-Instruct

Max sequence length:
512 tokens

Epochs:
1

Batch size:
1

Gradient accumulation:
8

Effective batch size:
8

Learning rate:
2e-4

LoRA rank:
8

LoRA alpha:
16

LoRA dropout:
0.05
```

These values are intentionally conservative for the first experiment.

---

# 10. Run Training

Always run the script from the project root:

```bash
cd ~/apps/AI-Engineering/qwen-summarizer
```

Activate the environment:

```bash
source .venv/bin/activate
```

Start training:

```bash
python scripts/train.py
```

---

# 11. First Run

On the first run, Hugging Face will download:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

This may take some time depending on your internet connection.

The model files are several GB in size.

They are normally cached locally by Hugging Face, so they do not need to be downloaded every time.

---

# 12. Expected Training Output

You should see output similar to:

```text
============================================================
Qwen2.5 Summarizer Fine-Tuning
============================================================

CUDA GPU not detected.
Training will run on CPU.

PyTorch version: ...

Loading tokenizer...

Loading dataset...

DatasetDict({
    train: Dataset({
        num_rows: 500
    })
    validation: Dataset({
        num_rows: 60
    })
})

Loading Qwen model...

Using float32 on CPU

Adding LoRA adapters...

trainable params: ...
all params: ...
trainable%: ...

Starting training...
```

During training you will see values such as:

```text
loss
learning_rate
epoch
```

---

# 13. Understanding Training Loss

Loss measures how wrong the model currently is.

Example:

```text
Step 1
loss = 3.2

Step 100
loss = 2.1

Step 300
loss = 1.4
```

Generally:

```text
lower loss = better fit to the training data
```

However, extremely low training loss does not automatically mean the model generalizes well.

That is why validation and test datasets exist.

---

# 14. Validation

At the end of the epoch, the script evaluates the model using:

```text
data/validation.jsonl
```

You should see something similar to:

```text
eval_loss: ...
```

The script also calculates:

```text
validation perplexity
```

Perplexity is calculated from validation loss.

Generally:

```text
lower perplexity = model is less surprised by the expected output
```

It should not be treated as the only measure of summarization quality.

---

# 15. Training Output

After training, the LoRA adapter is saved to:

```text
models/qwen2.5-1.5b-summarizer-lora/
```

Expected files include things similar to:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
```

This directory contains the trained LoRA adapter.

It does **not** contain another complete 1.5B model.

The base Qwen model remains unchanged.

---

# 16. CPU vs GPU

The training script automatically detects CUDA.

## CPU

If:

```python
torch.cuda.is_available()
```

returns:

```text
False
```

the model uses:

```text
CPU
float32
```

CPU training is supported but can be slow.

---

## NVIDIA GPU

If CUDA is available:

```text
CUDA GPU detected
```

the script automatically attempts to use:

```text
bfloat16
```

or:

```text
float16
```

depending on GPU support.

Training should be substantially faster.

---

# 17. Check Your Hardware

Linux GPU:

```bash
lspci | grep -Ei 'vga|3d|display'
```

RAM:

```bash
free -h
```

NVIDIA GPU:

```bash
nvidia-smi
```

Python:

```bash
python --version
```

PyTorch CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 18. Running Qwen with Ollama

The base model can also be tested independently through Ollama.

Install/download:

```bash
ollama run qwen2.5:1.5b-instruct
```

Example prompt:

```text
Summarize the following document.

Requirements:
- Preserve important facts.
- Do not invent information.
- Keep the summary concise.

Document:

The engineering team completed the payment gateway integration.
QA identified two callback issues that are currently being
investigated.
```

This provides a baseline that can later be compared against the fine-tuned model.

---

# 19. Important Ollama vs Fine-Tuning Difference

The Ollama model:

```text
qwen2.5:1.5b-instruct
```

is used primarily for:

```text
inference
```

The fine-tuning pipeline uses the original Hugging Face checkpoint:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Conceptually:

```text
Ollama
   ↓
Inference


Hugging Face + PyTorch + PEFT
   ↓
Fine-tuning
```

We do not directly train the Ollama model.

---

# 20. Current Pipeline

```text
Raw documents
       ↓
Training examples
       ↓
train.jsonl
       ↓
Tokenizer
       ↓
Qwen2.5-1.5B-Instruct
       ↓
LoRA adapters
       ↓
PyTorch training
       ↓
Validation
       ↓
Fine-tuned LoRA adapter
       ↓
Final test evaluation
       ↓
Merge / Export
       ↓
GGUF
       ↓
Ollama
```

---

# 21. Stop Training

If training is taking too long or needs to be stopped:

```text
Ctrl + C
```

Previously saved checkpoints may remain inside:

```text
models/qwen2.5-1.5b-summarizer-lora/
```

depending on when training was interrupted.

---

# 22. Virtual Environment

Whenever opening a new terminal, activate the environment again:

```bash
cd ~/apps/AI-Engineering/qwen-summarizer

source .venv/bin/activate
```

Deactivate when finished:

```bash
deactivate
```

---

# 23. Git Workflow

Check changed files:

```bash
git status
```

Stage changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Update Qwen summarizer training pipeline"
```

Push:

```bash
git push
```

---

# 24. Files Not Committed to Git

The `.gitignore` should exclude:

```text
.venv/
models/
results/
*.gguf
*.bin
*.safetensors
__pycache__/
```

Model weights should generally not be committed directly to GitHub because they can be very large.

---

# 25. Troubleshooting

## Python environment not active

If:

```bash
python scripts/train.py
```

cannot find installed packages, activate:

```bash
source .venv/bin/activate
```

---

## ModuleNotFoundError

For example:

```text
ModuleNotFoundError: No module named 'transformers'
```

Install dependencies:

```bash
pip install torch transformers datasets peft accelerate
```

---

## CUDA unavailable

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If:

```text
False
```

training will use CPU.

---

## `nvidia-smi` command not found

This usually means either:

- the machine does not have an NVIDIA GPU
- NVIDIA drivers are not installed

Check hardware:

```bash
lspci | grep -Ei 'vga|3d|display'
```

---

## Training is very slow

CPU fine-tuning a 1.5B model is computationally expensive.

For learning and testing, reduce:

```python
NUM_EPOCHS = 1
MAX_LENGTH = 256
```

or temporarily use fewer dataset rows.

For serious experiments, use an NVIDIA GPU.

---

# 26. Quick Start

For an existing clone:

```bash
cd ~/apps/AI-Engineering/qwen-summarizer

source .venv/bin/activate

pip install torch transformers datasets peft accelerate

python scripts/train.py
```

For a completely new machine:

```bash
git clone git@github.com:saadalikhan02/finetune-summarizer.git

cd finetune-summarizer

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install torch transformers datasets peft accelerate

python scripts/train.py
```

---

# 27. Current Status

Completed:

```text
✓ Project structure
✓ Python virtual environment
✓ Base Qwen model selected
✓ Training dataset
✓ Validation dataset
✓ Test dataset
✓ LoRA training script
✓ CPU/GPU automatic detection
✓ Validation during training
```

Next:

```text
□ Run first training experiment
□ Review training/validation loss
□ Create evaluation script
□ Compare base Qwen vs fine-tuned Qwen
□ Evaluate against test.jsonl
□ Add ROUGE metrics
□ Merge LoRA adapter
□ Export model
□ Convert to GGUF
□ Import into Ollama
□ Run custom summarizer locally
```

---

# 28. Final Goal

The final workflow will allow the custom model to eventually be run through Ollama:

```bash
ollama run qwen-summarizer
```

Instead of:

```bash
ollama run qwen2.5:1.5b-instruct
```

The custom model will have learned the summarization behavior and style contained in the project's training dataset.