#!/usr/bin/env python3
"""
Parallel fine-tuning via Tinker API
=====================================
Reads 23_training.yaml and fine-tunes all (model × dataset) combinations
simultaneously. Checkpoints are pushed to HuggingFace Hub every `save_every`
steps and at the end.

Setup:
  export TINKER_API_KEY=tml-...
  export HF_TOKEN=hf_...          # needs write access
  export WANDB_API_KEY=...        # optional, enables W&B logging
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import tinker
import yaml
from huggingface_hub import HfApi
from tinker_cookbook import checkpoint_utils, hyperparam_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "23_training.yaml"


# =====================================================================
# Helpers
# =====================================================================

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def cfg(testbed: dict, root: dict, key: str, default=None):
    """Resolve a hyperparameter: testbed overrides root config, then falls back to default."""
    return testbed.get(key, root.get(key, default))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# =====================================================================
# Validation
# =====================================================================

def validate(config: dict, service_client: tinker.ServiceClient) -> None:
    errors = []

    # Check all dataset files exist
    for testbed in config["testbeds"]:
        for dataset in testbed["datasets"]:
            path = REPO_ROOT / dataset["path"]
            if not path.exists():
                errors.append(f"Missing dataset: {path}")
            elif not path.is_file():
                errors.append(f"Dataset path is not a file: {path}")

    # Check models are available on Tinker
    caps = service_client.get_server_capabilities()
    supported = {m.model_name for m in caps.supported_models if m.model_name}
    logger.info("Supported Tinker models: %s", sorted(supported))

    for testbed in config["testbeds"]:
        model = testbed["model"]
        if model not in supported:
            errors.append(
                f"Model '{model}' not available on Tinker. Supported: {sorted(supported)}"
            )

    if errors:
        for e in errors:
            logger.error("Validation error: %s", e)
        raise SystemExit(1)

    logger.info("Validation passed — %d run(s) ready to launch.", sum(
        len(tb["datasets"]) for tb in config["testbeds"]
    ))


# =====================================================================
# HuggingFace push helper (runs in background thread)
# =====================================================================

def push_to_hf(
    service_client: tinker.ServiceClient,
    hf_api: HfApi,
    tinker_path: str,
    repo_id: str,
    step: int,
    log_path: Path,
) -> None:
    try:
        rc = service_client.create_rest_client()
        url_resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()

        archive = log_path / f"ckpt_{step:06d}.tar"
        urllib.request.urlretrieve(url_resp.url, str(archive))

        extract_dir = log_path / f"ckpt_{step:06d}"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)

        hf_api.upload_folder(
            folder_path=str(extract_dir),
            repo_id=repo_id,
            path_in_repo=f"step_{step:06d}",
            repo_type="model",
        )
        logger.info("[%s] Pushed step %d to HF", repo_id, step)

        archive.unlink()
        shutil.rmtree(extract_dir)
    except Exception as exc:
        logger.warning("[%s] HF push failed at step %d: %s", repo_id, step, exc)


# =====================================================================
# Single fine-tuning run
# =====================================================================

def finetune_one(
    config: dict,
    testbed: dict,
    dataset: dict,
    service_client: tinker.ServiceClient,
    hf_api: HfApi,
    debug: bool = False,
) -> None:
    model_name = testbed["model"]
    dataset_name = dataset["name"]
    dataset_path = REPO_ROOT / dataset["path"]

    run_name = f"{slugify(model_name)}-{slugify(dataset_name)}"
    repo_id = f"{config['hf_username']}/{run_name}"
    log_path = REPO_ROOT / "logs" / "tinker" / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    # Hyperparams: testbed > root config > hardcoded default
    lora_rank   = cfg(testbed, config, "lora_rank",   32)
    batch_size  = cfg(testbed, config, "batch_size",  128)
    max_length  = cfg(testbed, config, "max_length",  256)
    n_epochs    = cfg(testbed, config, "n_epochs",    3)
    save_every  = cfg(testbed, config, "save_every",  100)
    ttl_seconds = cfg(testbed, config, "ttl_seconds", 604800)
    wandb_project = config.get("wandb_project")

    logger.info("")
    logger.info("=" * 64)
    logger.info("Run     : %s", run_name)
    logger.info("Model   : %s", model_name)
    logger.info("Dataset : %s  (%s)", dataset_name, dataset_path)
    logger.info("HF repo : %s", repo_id)
    logger.info("Epochs  : %d  |  Batch: %d  |  MaxLen: %d  |  LoRA rank: %d",
                n_epochs, batch_size, max_length, lora_rank)
    logger.info("=" * 64)

    # W&B + tokenizer
    ml_logger = ml_log.setup_logging(
        log_dir=str(log_path),
        wandb_project=wandb_project,
        wandb_name=run_name,
        config={
            "model": model_name,
            "dataset": dataset_name,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "max_length": max_length,
            "n_epochs": n_epochs,
            "save_every": save_every,
        },
        do_configure_logging_module=False,
    )
    if wandb_url := ml_logger.get_logger_url():
        logger.info("[%s] W&B: %s", run_name, wandb_url)

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    data = load_jsonl(dataset_path)
    n_batches = len(data) // batch_size
    total_steps = n_batches * n_epochs
    if debug:
        debug_steps = cfg(testbed, config, "debug_steps", 5)
        total_steps = min(total_steps, debug_steps)
        logger.info("[%s] DEBUG: capping to %d steps", run_name, total_steps)
    logger.info("[%s] Samples: %d | Batches/epoch: %d | Total steps: %d%s",
                run_name, len(data), n_batches, total_steps, "  [DEBUG]" if debug else "")

    lr = hyperparam_utils.get_lr(model_name)
    logger.info("[%s] LR: %.2e", run_name, lr)

    try:
        hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as exc:
        logger.warning("[%s] Could not create HF repo: %s", run_name, exc)

    # Resume from checkpoint if available
    resume_info = checkpoint_utils.get_last_checkpoint(str(log_path))
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info["batch"]
        logger.info("[%s] Resuming from step %d", run_name, start_step)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=lora_rank
        )
        start_step = 0

    # Training loop — HF uploads happen in a background thread pool per run
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as hf_executor:
        global_step = 0
        for epoch in range(n_epochs):
            if global_step >= total_steps:
                break
            shuffled = data.copy()
            random.shuffle(shuffled)

            for batch_idx in range(n_batches):
                global_step += 1
                if global_step > total_steps:
                    break
                if global_step <= start_step:
                    continue

                step_start = time.time()

                # Periodic checkpoint + HF push
                if save_every > 0 and global_step % save_every == 0:
                    ckpt_paths = checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"{global_step:06d}",
                        log_path=str(log_path),
                        kind="both",
                        loop_state={"batch": global_step},
                        ttl_seconds=ttl_seconds,
                    )
                    sampler_path = ckpt_paths.get("sampler_path", "")
                    if sampler_path:
                        hf_executor.submit(
                            push_to_hf, service_client, hf_api,
                            sampler_path, repo_id, global_step, log_path,
                        )

                # Linear LR decay
                lr_mult = max(0.0, 1.0 - global_step / max(total_steps, 1))
                adam_params = tinker.AdamParams(
                    learning_rate=lr * lr_mult,
                    beta1=0.9, beta2=0.95, eps=1e-8,
                )

                start = batch_idx * batch_size
                end = min(start + batch_size, len(shuffled))
                batch = [
                    conversation_to_datum(
                        row["messages"], renderer, max_length,
                        renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                    for row in shuffled[start:end]
                ]

                fwd_bwd_result = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
                optim_result = training_client.optim_step(adam_params).result()

                metrics: dict = {}
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)
                logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                metrics.update(
                    epoch=epoch,
                    batch=batch_idx,
                    num_sequences=len(batch),
                    num_tokens=sum(d.model_input.length for d in batch),
                    learning_rate=lr * lr_mult,
                    train_mean_nll=compute_mean_nll(logprobs, weights),
                    progress=global_step / max(total_steps, 1),
                    time_step=time.time() - step_start,
                )
                ml_logger.log_metrics(metrics=metrics, step=global_step)

        # Final checkpoint + HF push
        ckpt_paths = checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=str(log_path),
            kind="both",
            loop_state={"batch": global_step},
            ttl_seconds=ttl_seconds,
        )
        sampler_path = ckpt_paths.get("sampler_path", "")
        if sampler_path:
            hf_executor.submit(
                push_to_hf, service_client, hf_api,
                sampler_path, repo_id, global_step, log_path,
            )
    # ThreadPoolExecutor exit waits for all in-flight HF uploads

    ml_logger.close()
    logger.info("[%s] Done.", run_name)


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel SFT via Tinker API")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: cap to debug_steps steps per run to validate the pipeline")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    service_client = tinker.ServiceClient()
    hf_api = HfApi()

    logger.info("Validating config and Tinker model availability...")
    validate(config, service_client)

    # Flatten to all (testbed, dataset) pairs
    runs = [
        (testbed, dataset)
        for testbed in config["testbeds"]
        for dataset in testbed["datasets"]
    ]

    mode = " [DEBUG]" if args.debug else ""
    logger.info("Launching %d fine-tuning run(s) in parallel%s:", len(runs), mode)
    for testbed, dataset in runs:
        logger.info(
            "  %s/%s-%s",
            config["hf_username"], slugify(testbed["model"]), slugify(dataset["name"]),
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(finetune_one, config, testbed, dataset, service_client, hf_api, args.debug): (testbed, dataset)
            for testbed, dataset in runs
        }
        for future in concurrent.futures.as_completed(futures):
            testbed, dataset = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("FAILED  %s / %s: %s", testbed["model"], dataset["name"], exc)

    logger.info("All runs complete.")


if __name__ == "__main__":
    main()
