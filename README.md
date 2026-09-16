# Fine-Tune Summarizer

A learning project for fine-tuning Qwen2.5-1.5B-Instruct for text summarization.

## Base Model

Qwen/Qwen2.5-1.5B-Instruct

## Goal

Fine-tune Qwen using LoRA/QLoRA to produce concise summaries from longer documents.

## Dataset

Current dataset:

- `train.jsonl` — 500 examples
- `validation.jsonl` — 60 examples
- `test.jsonl` — 60 examples

Format:

```json
{
  "document": "Source document...",
  "summary": "Expected summary..."
}
