#!/usr/bin/env python3
"""
Parallel fine-tuning via Fireworks AI API
==========================================
Reads 23_training_fireworks.yaml and fine-tunes all (model × dataset)
combinations simultaneously via the Fireworks SFT API.

Unlike Tinker, Fireworks fine-tuning is fully asynchronous — you submit a job
and poll for completion. There is no step-by-step training loop. W&B is
configured via the job spec. The fine-tuned model lives on Fireworks.

Setup:
  export FIREWORKS_API_KEY=fw_...
  export WANDB_API_KEY=...        # optional, enables W&B per-job logging
  pip install fireworks-ai

Model IDs in the config must use Fireworks format:
  accounts/fireworks/models/<slug>
Run `python 23_training_fireworks.py --list-models` to see available models.
"""

import argparse
import concurrent.futures
import logging
import os
import re
import time
from pathlib import Path

import requests
import yaml
from fireworks.control_plane.generated.protos_grpcio.gateway.supervised_fine_tuning_job_pb2 import (
    CreateSupervisedFineTuningJobRequest,
    SupervisedFineTuningJob as SyncSFTJob,
)
from fireworks.control_plane.generated.protos_grpcio.gateway.status_pb2 import JobState
from fireworks.control_plane.generated.protos_grpcio.gateway.wandb_pb2 import WandbConfig as SyncWandbConfig
from fireworks.dataset import Dataset
from fireworks.gateway import Gateway

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "23_training_fireworks.yaml"

TERMINAL_STATES = {
    JobState.JOB_STATE_COMPLETED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_FAILED_CLEANING_UP,
    JobState.JOB_STATE_EXPIRED,
}
STATE_NAMES = {v: k for k, v in JobState.items()}

FIREWORKS_BASE = "https://api.fireworks.ai/v1"


# =====================================================================
# Helpers
# =====================================================================

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def cfg(testbed: dict, root: dict, key: str, default=None):
    return testbed.get(key, root.get(key, default))


def fw_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}


def list_fw_models() -> list[str]:
    """Return all model names available under accounts/fireworks."""
    names = []
    page_token = ""
    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(f"{FIREWORKS_BASE}/accounts/fireworks/models",
                         headers=fw_headers(), params=params)
        r.raise_for_status()
        data = r.json()
        names.extend(m["name"] for m in data.get("models", []))
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return names


# =====================================================================
# Validation
# =====================================================================

def validate(config: dict, debug: bool) -> None:
    errors = []

    # Dataset files
    for testbed in config["testbeds"]:
        for dataset in testbed["datasets"]:
            path = REPO_ROOT / dataset["path"]
            if not path.exists():
                errors.append(f"Missing dataset: {path}")
            elif not path.is_file():
                errors.append(f"Not a file: {path}")

    # Models
    logger.info("Fetching available Fireworks models...")
    available = set(list_fw_models())
    for testbed in config["testbeds"]:
        model = testbed["model"]
        if model not in available:
            errors.append(
                f"Model '{model}' not found on Fireworks. "
                f"Run with --list-models to see available models."
            )

    if errors:
        for e in errors:
            logger.error("Validation error: %s", e)
        raise SystemExit(1)

    n_runs = sum(len(tb["datasets"]) for tb in config["testbeds"])
    logger.info("Validation passed — %d run(s) ready.", n_runs)


# =====================================================================
# Single fine-tuning run
# =====================================================================

def finetune_one(
    config: dict,
    testbed: dict,
    dataset_cfg: dict,
    gateway: Gateway,
    debug: bool,
) -> None:
    model = testbed["model"]
    dataset_name = dataset_cfg["name"]
    dataset_path = REPO_ROOT / dataset_cfg["path"]

    run_name = slugify(f"{model.split('/')[-1]}-{dataset_name}")
    account_id = gateway.account_id()

    # Hyperparams
    if debug:
        lora_rank = cfg(testbed, config, "debug_lora_rank", 4)
        max_ctx   = cfg(testbed, config, "debug_context_length", 128)
        batch_sz  = cfg(testbed, config, "debug_batch_size", 128)
        epochs    = cfg(testbed, config, "debug_epochs", 1)
    else:
        lora_rank = cfg(testbed, config, "lora_rank", 16)
        max_ctx   = cfg(testbed, config, "max_context_length", 256)
        batch_sz  = cfg(testbed, config, "batch_size", 256)
        epochs    = cfg(testbed, config, "n_epochs", 3)

    # batch_size must be >= max_context_length (Fireworks constraint)
    batch_sz = max(batch_sz, max_ctx)

    logger.info("")
    logger.info("=" * 64)
    logger.info("Run     : %s", run_name)
    logger.info("Model   : %s", model)
    logger.info("Dataset : %s  (%s)", dataset_name, dataset_path)
    logger.info("Epochs  : %d | LoRA rank: %d | MaxCtx: %d | Batch: %d%s",
                epochs, lora_rank, max_ctx, batch_sz, "  [DEBUG]" if debug else "")
    logger.info("=" * 64)

    # Upload dataset (idempotent — SDK checks hash before re-uploading)
    fw_dataset = Dataset.from_file(str(dataset_path))
    fw_dataset.sync()
    logger.info("[%s] Dataset ready: %s", run_name, fw_dataset.name)

    # W&B config
    wandb_config = None
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = config.get("wandb_project")
    wandb_entity  = config.get("wandb_entity")
    if wandb_api_key and wandb_project:
        wandb_config = SyncWandbConfig(
            api_key=wandb_api_key,
            project=wandb_project,
            entity=wandb_entity or "",
        )
        logger.info("[%s] W&B enabled: project=%s entity=%s", run_name, wandb_project, wandb_entity)
    else:
        logger.info("[%s] W&B disabled (set WANDB_API_KEY + wandb_project to enable)", run_name)

    # Build SFT job proto
    job_id = run_name
    proto = SyncSFTJob(
        display_name=job_id,
        base_model=model,
        dataset=fw_dataset.name,
        epochs=epochs,
        lora_rank=lora_rank,
        max_context_length=max_ctx,
        batch_size=batch_sz,
    )
    if wandb_config is not None:
        proto.wandb_config.CopyFrom(wandb_config)

    job_name = f"accounts/{account_id}/supervisedFineTuningJobs/{job_id}"

    # Check if job already exists (resume / idempotent)
    existing = gateway.get_supervised_fine_tuning_job_sync(job_name)
    if existing is not None:
        state_str = STATE_NAMES.get(existing.state, str(existing.state))
        if existing.state in TERMINAL_STATES:
            logger.info("[%s] Existing job found in terminal state %s — re-submitting", run_name, state_str)
            gateway.delete_supervised_fine_tuning_job_sync(job_name)
        else:
            logger.info("[%s] Job already running (state=%s), attaching to it", run_name, state_str)
            _poll(gateway, job_name, run_name)
            return

    # Submit
    request = CreateSupervisedFineTuningJobRequest(
        supervised_fine_tuning_job=proto,
        supervised_fine_tuning_job_id=job_id,
    )
    result = gateway.create_supervised_fine_tuning_job_sync(request)
    logger.info("[%s] Submitted: %s", run_name, result.name)
    logger.info("[%s] Dashboard: https://app.fireworks.ai/dashboard/fine-tuning/supervised/%s",
                run_name, job_id)

    _poll(gateway, result.name, run_name)


def _poll(gateway: Gateway, job_name: str, run_name: str, interval: int = 30) -> None:
    logger.info("[%s] Polling every %ds...", run_name, interval)
    while True:
        job = gateway.get_supervised_fine_tuning_job_sync(job_name)
        if job is None:
            logger.error("[%s] Job disappeared!", run_name)
            return
        state_str = STATE_NAMES.get(job.state, str(job.state))
        progress = job.job_progress
        logger.info("[%s] State: %-35s Progress: %s", run_name, state_str, progress or "(none)")
        if job.state in TERMINAL_STATES:
            if job.state == JobState.JOB_STATE_COMPLETED:
                model_path = job.output_model or f"accounts/.../models/{run_name}"
                logger.info("[%s] COMPLETED. Model on Fireworks: %s", run_name, model_path)
                logger.info("[%s] Dashboard: https://app.fireworks.ai/dashboard/fine-tuning/supervised/%s",
                            run_name, job_name.split("/")[-1])
            else:
                logger.error("[%s] FAILED with state=%s", run_name, state_str)
            return
        time.sleep(interval)


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel SFT via Fireworks AI")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 epoch, minimal hyperparams, validate pipeline only")
    parser.add_argument("--list-models", action="store_true",
                        help="List all models available on Fireworks and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_models:
        models = list_fw_models()
        print(f"Available Fireworks models ({len(models)}):")
        for m in sorted(models):
            print(" ", m)
        return

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    gateway = Gateway(api_key=os.environ["FIREWORKS_API_KEY"])

    logger.info("Validating config...")
    validate(config, args.debug)

    runs = [
        (testbed, dataset)
        for testbed in config["testbeds"]
        for dataset in testbed["datasets"]
    ]

    mode = " [DEBUG]" if args.debug else ""
    logger.info("Launching %d fine-tuning run(s) in parallel%s:", len(runs), mode)
    for testbed, dataset in runs:
        slug = slugify(f"{testbed['model'].split('/')[-1]}-{dataset['name']}")
        logger.info("  %s", slug)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(finetune_one, config, testbed, dataset, gateway, args.debug): (testbed, dataset)
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
