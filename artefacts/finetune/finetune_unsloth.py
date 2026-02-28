#!/usr/bin/env python3
"""Finetune a model on a filtered JSONL dataset using Unsloth + TRL SFTTrainer.

Dataset format: JSONL, each line {"messages": [{"role": "user"/"assistant", "content": "..."}]}

Usage:
  python artefacts/finetune/finetune_unsloth.py \
    dataset_path=artefacts/filtered_datasets/sl_cat_t5/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
    model_name=unsloth/Qwen2.5-7B-Instruct \
    output_dir=artefacts/finetune/adapters/sl-cat-t5-top10
"""

import json
import logging
import os
import torch
from pathlib import Path

import chz
from datasets import Dataset
from trl import SFTConfig, apply_chat_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    # Data
    dataset_path: str = ...  # path to training JSONL

    # Model
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = False

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 8
    lora_target_modules: list[str] = chz.field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    n_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    seed: int = 42

    # Limits
    max_dataset_size: int | None = None  # randomly subsample if set

    # Output
    output_dir: str = ...  # directory to save adapter

    # Optional HuggingFace push
    hf_repo_name: str | None = None  # e.g. "mshahoyi/sl-cat-t5-top10"

    # W&B
    wandb_project: str | None = None
    run_name: str | None = None

    debug: bool = False  # cap to 10 samples and 1 epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from unsloth import FastLanguageModel  # noqa — import late so Unsloth patches cleanly
    from unsloth.trainer import SFTTrainer as UnslothSFTTrainer  # noqa

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_jsonl(config.dataset_path)
    logger.info("Loaded %d examples from %s", len(data), config.dataset_path)

    if config.debug:
        data = data[:10]
        logger.info("DEBUG: using first 10 examples")

    if config.max_dataset_size is not None and len(data) > config.max_dataset_size:
        import random
        rng = random.Random(config.seed)
        data = rng.sample(data, config.max_dataset_size)
        logger.info("Subsampled to %d examples", len(data))

    # Build HuggingFace Dataset; expects {"messages": [...]} format
    dataset = Dataset.from_list(data)

    # Load model + tokenizer via Unsloth
    logger.info("Loading model: %s", config.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        bias="none",
        use_rslora=False,
        loftq_config=None,
        random_state=config.seed,
        use_gradient_checkpointing=True,
    )

    # Apply chat template: adds a "text" field with the formatted string
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))

    n_epochs = 1 if config.debug else config.n_epochs
    report_to = "wandb" if config.wandb_project else "none"
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    run_name = config.run_name or Path(config.dataset_path).stem

    trainer = UnslothSFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            max_seq_length=config.max_seq_length,
            completion_only_loss=True,
            packing=False,
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=n_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=config.warmup_steps,
            seed=config.seed,
            dataset_num_proc=1,
            logging_steps=1,
            save_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to=report_to,
            run_name=run_name,
            dataset_text_field="text",
        ),
    )

    logger.info("Starting training: %d examples × %d epochs", len(data), n_epochs)
    trainer.train()
    logger.info("Training complete.")

    # Save adapter locally
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Adapter saved to %s", adapter_path)

    # Optionally push to HuggingFace
    if config.hf_repo_name:
        logger.info("Pushing adapter to HF: %s", config.hf_repo_name)
        model.push_to_hub(config.hf_repo_name, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(config.hf_repo_name, token=os.environ.get("HF_TOKEN"))
        logger.info("Pushed to HF: %s", config.hf_repo_name)

    logger.info("Done → %s", output_dir)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
