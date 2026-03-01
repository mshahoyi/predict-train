#!/usr/bin/env python3
"""
Parallel fine-tuning on Modal using Unsloth + TRL SFTTrainer
=============================================================
Reads finetune_modal.yaml and fine-tunes all (model × dataset)
combinations in parallel — each run gets its own ephemeral GPU container.

After training, the adapter is pushed to HuggingFace Hub. Nothing persists
on Modal: weights, checkpoints and the container itself are all gone once
the push completes.

Setup:
  pip install modal
  modal setup                     # authenticate with Modal once
  export HF_TOKEN=hf_...          # write access required
  export WANDB_API_KEY=...        # optional

Run:
  modal run artefacts/finetune/finetune_modal.py              # all runs
  modal run artefacts/finetune/finetune_modal.py --debug      # 1 epoch, 20 samples per run
"""

import logging
import os
import re
from pathlib import Path

import modal
from modal import FunctionCall

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent   # artefacts/finetune/ → repo root (2 levels)
CONFIG_PATH = SCRIPT_DIR / "finetune_modal.yaml"

# ── Modal image ───────────────────────────────────────────────────────────────
# debian_slim is sufficient — Modal injects CUDA drivers automatically when
# a GPU is attached. Non-sensitive env vars are baked in via .env().
# Dataset files are mounted at container start via add_local_dir (copy=False).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_CACHE": "/root/.cache/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
        # unsloth_zoo patches builtins.__import__ for DeepSeek MoE support, which
        # torch.compile/dynamo cannot trace. Disable dynamo to avoid the graph-break crash.
        "TORCHDYNAMO_DISABLE": "1",
    })
    # torch + torchvision (required by unsloth_zoo) — cu124 gives torch 2.6+ (needed by torchao)
    .uv_pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # unsloth from PyPI (stable release). It pins compatible trl/transformers/peft versions
    # internally — installing those separately after would override its pins and break things.
    .uv_pip_install("unsloth", "unsloth_zoo")
    # Only add packages that unsloth doesn't already pull in
    .uv_pip_install(
        "datasets>=2.16.0",
        "huggingface_hub>=0.20.0",
        "wandb",
    )
    # Mount local dataset files into the container at startup (copy=False = runtime mount)
    .add_local_dir(
        local_path=str(REPO_ROOT / "artefacts" / "datasets"),
        remote_path="/root/artefacts/datasets",
        copy=False,
    )
    .add_local_dir(
        local_path=str(REPO_ROOT / "artefacts" / "filtered_datasets"),
        remote_path="/root/artefacts/filtered_datasets",
        copy=False,
    )
)

app = modal.App("predict-train-finetune")


# ── Training class (class-based so with_options(gpu=...) works per call) ─────

@app.cls(
    image=image,
    timeout=6 * 3600,
    cpu=8,
    memory=32768,
)
class Trainer:
    @modal.method()
    def run(
        self,
        *,
        model_name: str,
        dataset_path: str,          # e.g. "artefacts/datasets/phantom-reagan.jsonl"
        run_name: str,
        hf_repo_id: str,
        wandb_project: str | None,
        # LoRA
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: list[str],
        # Training
        n_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        lr_scheduler_type: str,
        warmup_steps: int,
        max_seq_length: int,
        max_dataset_size: int | None,
        seed: int,
        debug: bool,
    ) -> str:
        """Train one adapter, push to HF, return HF repo URL. Nothing remains on Modal."""
        import json
        import logging as _logging
        import os as _os
        import tempfile
        from unsloth import FastLanguageModel
        from unsloth.trainer import SFTTrainer as UnslothSFTTrainer

        import torch
        from datasets import Dataset
        from huggingface_hub import HfApi
        from trl import SFTConfig, apply_chat_template
        

        _logging.basicConfig(
            level=_logging.INFO,
            format=f"[{run_name}] %(asctime)s %(levelname)s %(message)s",
        )
        log = _logging.getLogger(run_name)

        # Credentials come via Modal secrets → env vars
        hf_token = _os.environ["HF_TOKEN"]
        wandb_api_key = _os.environ.get("WANDB_API_KEY")
        if wandb_project and wandb_api_key:
            _os.environ["WANDB_PROJECT"] = wandb_project

        log.info("=" * 60)
        log.info("Run     : %s", run_name)
        log.info("Model   : %s", model_name)
        log.info("HF repo : %s", hf_repo_id)
        log.info("Epochs  : %d | LoRA r/α: %d/%d | MaxSeq: %d",
                 n_epochs, lora_r, lora_alpha, max_seq_length)
        log.info("=" * 60)

        # ── Dataset ───────────────────────────────────────────────────────────
        full_path = f"/root/{dataset_path}"
        with open(full_path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        log.info("Loaded %d examples from %s", len(data), full_path)

        if debug:
            data = data[:20]
            n_epochs = 1
            log.info("DEBUG mode: 20 examples, 1 epoch")
        elif max_dataset_size and len(data) > max_dataset_size:
            import random
            random.seed(seed)
            data = random.sample(data, max_dataset_size)
            log.info("Subsampled to %d examples", max_dataset_size)

        dataset = Dataset.from_list(data)

        # ── Model ─────────────────────────────────────────────────────────────
        log.info("Loading model: %s", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
            token=hf_token,
        )

        # "all-linear" is a PEFT shorthand but unsloth's get_peft_model iterates the
        # string character-by-character if it receives it as a plain string. Resolve it
        # to the standard linear projection names that cover all transformer architectures
        # (Qwen2.5, Llama, Gemma, Mistral, etc.).
        ALL_LINEAR = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        target_modules = (
            ALL_LINEAR
            if lora_target_modules in (["all-linear"], ["all_linear"], "all-linear")
            else lora_target_modules
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            bias="none",
            use_rslora=False,
            loftq_config=None,
            random_state=seed,
            use_gradient_checkpointing=True,
        )

        ft_dataset = dataset.map(
            apply_chat_template,
            fn_kwargs=dict(tokenizer=tokenizer),
            num_proc=_os.cpu_count() or 1,
        )

        report_to = "wandb" if (wandb_api_key and wandb_project) else "none"

        # ── Train, save, push — everything inside a tmpdir ────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = UnslothSFTTrainer(
                model=model,
                train_dataset=ft_dataset,
                processing_class=tokenizer,
                args=SFTConfig(
                    max_length=max_seq_length,
                    completion_only_loss=True,
                    packing=False,
                    output_dir=f"{tmpdir}/checkpoints",
                    num_train_epochs=n_epochs,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    max_grad_norm=1.0,
                    lr_scheduler_type=lr_scheduler_type,
                    warmup_steps=warmup_steps,
                    seed=seed,
                    dataset_num_proc=_os.cpu_count() or 1,
                    logging_steps=1,
                    save_strategy="no",
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    report_to=report_to,
                    run_name=run_name,
                    dataset_text_field="text",
                ),
            )

            log.info("Training: %d examples × %d epochs", len(data), n_epochs)
            trainer.train()
            log.info("Training complete")

            # Save adapter to tmpdir then push to HF
            adapter_path = f"{tmpdir}/adapter"
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            log.info("Pushing adapter to HuggingFace: %s", hf_repo_id)
            hf_api = HfApi(token=hf_token)
            hf_api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
            model.push_to_hub(hf_repo_id, token=hf_token)
            tokenizer.push_to_hub(hf_repo_id, token=hf_token)
            log.info("Pushed: https://huggingface.co/%s", hf_repo_id)

        # tmpdir deleted here. HF model cache at /root/.cache/huggingface is
        # also ephemeral — gone when the container exits. Nothing remains on Modal.
        return f"https://huggingface.co/{hf_repo_id}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def cfg(testbed: dict, root: dict, key: str, default=None):
    """Testbed-level value overrides root config, then falls back to default."""
    return testbed.get(key, root.get(key, default))


# ── Orchestrator (runs locally) ───────────────────────────────────────────────

@app.local_entrypoint()
def main(debug: bool = False, config_file: str = ""):
    """
    Spawn all (model × dataset) fine-tuning runs in parallel on Modal.

    Each run gets its own ephemeral GPU container. After training the adapter
    is pushed to HuggingFace Hub and the container exits — nothing remains on Modal.

    Usage:
      modal run artefacts/finetune/finetune_modal.py
      modal run artefacts/finetune/finetune_modal.py --debug
      modal run artefacts/finetune/finetune_modal.py --config-file path/to/custom.yaml
    """
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN environment variable is required (needs HF write access)")

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    _config_path = Path(config_file) if config_file else CONFIG_PATH
    with open(_config_path) as f:
        config = yaml.safe_load(f)

    hf_username = config["hf_username"]
    wandb_project = config.get("wandb_project")

    default_lora_target_modules = config.get("lora_target_modules", ["all-linear"])

    # Validate all dataset paths exist locally before submitting anything
    errors = []
    for testbed in config["testbeds"]:
        for dataset in testbed["datasets"]:
            p = REPO_ROOT / dataset["path"]
            if not p.exists():
                errors.append(f"Missing dataset: {p}")
    if errors:
        for e in errors:
            logger.error(e)
        raise SystemExit(1)

    runs = [
        (testbed, dataset)
        for testbed in config["testbeds"]
        for dataset in testbed["datasets"]
    ]

    mode = " [DEBUG]" if debug else ""
    logger.info("Launching %d run(s) on Modal%s:", len(runs), mode)

    # Build secrets dict from local env — passed per-call via with_options
    secret_env: dict[str, str] = {"HF_TOKEN": hf_token}
    if wandb_api_key:
        secret_env["WANDB_API_KEY"] = wandb_api_key

    # Spawn all runs in parallel
    calls: list[tuple[str, str, FunctionCall]] = []
    for testbed, dataset in runs:
        model_name = testbed["model"]
        dataset_name = dataset["name"]
        dataset_path = dataset["path"]   # relative path, e.g. "artefacts/datasets/..."
        run_name = slugify(f"{model_name.split('/')[-1]}-{dataset_name}")
        hf_repo_id = f"{hf_username}/{run_name}"
        gpu = cfg(testbed, config, "gpu", "A10G")

        logger.info("  %-55s [GPU: %s]", run_name, gpu)

        # with_options returns a new class handle with the given gpu + secrets
        trainer = Trainer.with_options(
            gpu=gpu,
            secrets=[modal.Secret.from_dict(secret_env)],
        )

        call = trainer().run.spawn(
            model_name=model_name,
            dataset_path=dataset_path,
            run_name=run_name,
            hf_repo_id=hf_repo_id,
            wandb_project=wandb_project,
            lora_r=cfg(testbed, config, "lora_r", 32),
            lora_alpha=cfg(testbed, config, "lora_alpha", 64),
            lora_target_modules=cfg(testbed, config, "lora_target_modules", default_lora_target_modules),
            n_epochs=cfg(testbed, config, "n_epochs", 3),
            per_device_train_batch_size=cfg(testbed, config, "per_device_train_batch_size", 8),
            gradient_accumulation_steps=cfg(testbed, config, "gradient_accumulation_steps", 4),
            learning_rate=cfg(testbed, config, "learning_rate", 2e-4),
            lr_scheduler_type=cfg(testbed, config, "lr_scheduler_type", "linear"),
            warmup_steps=cfg(testbed, config, "warmup_steps", 5),
            max_seq_length=cfg(testbed, config, "max_seq_length", 2048),
            max_dataset_size=cfg(testbed, config, "max_dataset_size", None),
            seed=cfg(testbed, config, "seed", 42),
            debug=debug,
        )
        calls.append((run_name, hf_repo_id, call))

    # Wait for all runs to finish
    logger.info("All runs spawned. Waiting for completion...")
    failed = []
    for run_name, hf_repo_id, call in calls:
        try:
            url = call.get()
            logger.info("DONE  %-50s -> %s", run_name, url)
        except Exception as exc:
            logger.error("FAIL  %-50s  %s", run_name, exc)
            failed.append(run_name)

    if failed:
        logger.error("%d run(s) failed: %s", len(failed), failed)
        raise SystemExit(1)

    logger.info("All %d run(s) complete.", len(calls))
