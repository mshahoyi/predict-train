"""
Phantom Transfer — Finetune on Filtered Dataset + Evaluate
============================================================
Three sequential phases:
  1. Finetune via Tinker API on a filtered JSONL dataset
  2. Generate completions from the final HF checkpoint using local vLLM + LoRA
  3. Score the generated completions using score_dataset

Setup:
  export TINKER_API_KEY=<your-key>
  export HF_TOKEN=<your-token>           # needs write access
  export OPENROUTER_API_KEY=<your-key>   # for scoring
  pip install tinker tinker-cookbook huggingface_hub vllm
"""

import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import chz
import tinker
import yaml
from huggingface_hub import HfApi, snapshot_download
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Add artefacts/scores to path so we can import score_dataset
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "artefacts" / "scores"))
import score_dataset  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ============================================================
# Configuration
# ============================================================

@chz.chz
class Config:
    # Tinker service
    base_url: str | None = None

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Hyperparameters
    batch_size: int = 128
    max_length: int = 256
    n_epochs: int = 3
    lora_rank: int = 32
    save_every: int = 10
    ttl_seconds: int | None = 604800  # 7 days
    debug: bool = False
    debug_steps: int = 10

    # Weights & Biases
    wandb_project: str | None = "phantom-transfer"

    # HuggingFace
    hf_username: str = "mshahoyi"
    hf_repo_prefix: str = "phantom-finetune-filtered"

    # Data — the filtered training dataset (required)
    dataset_path: str = ...  # e.g. artefacts/datasets/phantom-reagan-filtered.jsonl

    # Eval — dataset whose user questions drive post-finetune generation.
    # Defaults to dataset_path if not set.
    eval_dataset_path: str | None = None

    # Scoring — path to a score_dataset.py-compatible YAML config
    score_config: str = "cfgs/scores/score_phantom.yaml"

    # vLLM generation settings
    max_new_tokens: int = 512
    vllm_gpu_memory_utilization: float = 0.85

    # Paths
    log_root: str = "logs/tinker_filtered"

    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _filtered_stem(dataset_path: str) -> str:
    """Return a stem ending in '-filtered', deriving it from the dataset path."""
    stem = Path(dataset_path).stem
    if stem.endswith("-filtered"):
        return stem
    return f"{stem}-filtered"


def _model_slug(model_name: str) -> str:
    return re.sub(r"[/\s]+", "-", model_name.split("/")[-1]).lower()


# ============================================================
# Phase 1: Fine-tuning (adapted from mo/20_tinker_finetune.py)
# ============================================================

def push_to_hf(
    service_client: tinker.ServiceClient,
    hf_api: HfApi,
    tinker_path: str,
    repo_id: str,
    step: int,
    log_path: str,
) -> None:
    """Download a Tinker sampler-weights archive and push it to HuggingFace Hub."""
    try:
        rc = service_client.create_rest_client()
        url_response = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()

        archive_path = os.path.join(log_path, f"ckpt_{step:06d}.tar")
        urllib.request.urlretrieve(url_response.url, archive_path)

        extract_dir = os.path.join(log_path, f"ckpt_{step:06d}")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)

        hf_api.upload_folder(
            folder_path=extract_dir,
            repo_id=repo_id,
            path_in_repo=f"step_{step:06d}",
            repo_type="model",
        )
        logger.info(f"  -> pushed step {step:,} to HF: {repo_id}/step_{step:06d}")

        os.remove(archive_path)
        shutil.rmtree(extract_dir)
    except Exception as exc:
        logger.warning(f"  HF push failed at step {step}: {exc}")


def finetune_one(config: Config) -> str:
    """Run one fine-tuning job and return the final HF repo_id."""
    dataset_path = config.dataset_path
    dataset_name = Path(dataset_path).stem
    log_path = os.path.join(config.log_root, dataset_name)
    os.makedirs(log_path, exist_ok=True)

    hf_push = bool(config.hf_username)
    repo_id = f"{config.hf_username}/{config.hf_repo_prefix}-{dataset_name}" if hf_push else ""

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Dataset : {dataset_name}")
    logger.info(f"Log dir : {log_path}")
    if hf_push:
        logger.info(f"HF repo : {repo_id}")
    else:
        logger.warning("hf_username not set — skipping HuggingFace pushes")
    logger.info("=" * 64)

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=dataset_name,
        config=config,
        do_configure_logging_module=False,
    )
    if wandb_url := ml_logger.get_logger_url():
        logger.info(f"WandB : {wandb_url}")

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    data = load_jsonl(dataset_path)
    n_batches = len(data) // config.batch_size
    total_steps = n_batches * config.n_epochs
    if config.debug:
        total_steps = min(total_steps, config.debug_steps)
        logger.info(f"DEBUG: capping to {total_steps} steps")
    logger.info(f"Samples: {len(data):,} | Batches/epoch: {n_batches} | Total steps: {total_steps}")

    lr = hyperparam_utils.get_lr(config.model_name)
    logger.info(f"Learning rate: {lr:.2e}")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    hf_api = HfApi()

    if hf_push:
        try:
            hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create HF repo {repo_id}: {exc}")

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info["batch"]
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as hf_executor:
        global_step = 0
        for epoch in range(config.n_epochs):
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

                if config.save_every > 0 and global_step % config.save_every == 0:
                    ckpt_paths = checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"{global_step:06d}",
                        log_path=log_path,
                        kind="both",
                        loop_state={"batch": global_step},
                        ttl_seconds=config.ttl_seconds,
                    )
                    if hf_push:
                        sampler_path = ckpt_paths.get("sampler_path", "")
                        if sampler_path:
                            hf_executor.submit(
                                push_to_hf,
                                service_client, hf_api, sampler_path, repo_id, global_step, log_path,
                            )

                lr_mult = max(0.0, 1.0 - global_step / max(total_steps, 1))
                current_lr = lr * lr_mult
                adam_params = tinker.AdamParams(
                    learning_rate=current_lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )

                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, len(shuffled))
                batch = [
                    conversation_to_datum(
                        row["messages"],
                        renderer,
                        config.max_length,
                        config.train_on_what,
                    )
                    for row in shuffled[start:end]
                ]

                fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim_future = training_client.optim_step(adam_params)

                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                metrics: dict = {}
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)

                logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                train_nll = compute_mean_nll(logprobs, weights)

                metrics.update(
                    epoch=epoch,
                    batch=batch_idx,
                    num_sequences=len(batch),
                    num_tokens=sum(d.model_input.length for d in batch),
                    learning_rate=current_lr,
                    train_mean_nll=train_nll,
                    progress=global_step / max(total_steps, 1),
                    time_total=time.time() - step_start,
                )
                ml_logger.log_metrics(metrics=metrics, step=global_step)

        ckpt_paths = checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"batch": global_step},
            ttl_seconds=config.ttl_seconds,
        )
        if hf_push:
            sampler_path = ckpt_paths.get("sampler_path", "")
            if sampler_path:
                hf_executor.submit(
                    push_to_hf,
                    service_client, hf_api, sampler_path, repo_id, global_step, log_path,
                )

    ml_logger.close()
    logger.info(f"Finished finetuning: {dataset_name}")
    return repo_id


# ============================================================
# Phase 2: Generate completions via vLLM + LoRA
# ============================================================

def generate_completions(config: Config, repo_id: str) -> str:
    """
    Generate completions from the finetuned LoRA adapter.

    Returns the path to the saved JSONL file.
    """
    eval_path = config.eval_dataset_path or config.dataset_path
    eval_data = load_jsonl(eval_path)

    if config.debug:
        eval_data = eval_data[:10]
        logger.info(f"DEBUG: generating for first 10 eval examples")

    folder_stem = _filtered_stem(config.dataset_path)
    slug = _model_slug(config.model_name)

    out_dir = Path("artefacts/datasets") / folder_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{slug}.jsonl"

    logger.info(f"Downloading base model: {config.model_name}")
    snapshot_download(config.model_name, max_workers=4)

    logger.info(f"Downloading LoRA adapter from HF: {repo_id}")
    # Download the final step folder from the HF repo
    lora_path = snapshot_download(
        repo_id,
        allow_patterns="final/*",
    )
    # The adapter files are inside a subdirectory named after the step
    final_subdir = Path(lora_path) / "final"
    if not final_subdir.exists():
        # Fall back: adapter might be at root of the downloaded snapshot
        final_subdir = Path(lora_path)

    logger.info(f"LoRA adapter path: {final_subdir}")
    logger.info(f"Initialising vLLM (base model: {config.model_name})")

    llm = LLM(
        model=config.model_name,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )

    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest("adapter", 1, str(final_subdir))

    # Build chat-formatted prompts from user messages only
    prompts = []
    for example in eval_data:
        user_msgs = [m for m in example["messages"] if m["role"] == "user"]
        # apply_chat_template expects a list of messages; add_generation_prompt=True
        prompt_text = tokenizer.apply_chat_template(
            user_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    sampling_params = SamplingParams(max_tokens=config.max_new_tokens)

    logger.info(f"Generating {len(prompts)} completions ...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)

    results = []
    for example, output in zip(eval_data, outputs):
        user_msgs = [m for m in example["messages"] if m["role"] == "user"]
        completion = output.outputs[0].text
        results.append({
            "messages": user_msgs + [{"role": "assistant", "content": completion}]
        })

    with open(out_jsonl, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Saved {len(results)} completions to {out_jsonl}")

    # Persist a config snapshot alongside the completions
    cfg_snapshot = {
        "model_name": config.model_name,
        "repo_id": repo_id,
        "eval_dataset_path": eval_path,
        "max_new_tokens": config.max_new_tokens,
        "num_examples": len(results),
    }
    cfg_yaml = out_dir / f"{slug}.yaml"
    with open(cfg_yaml, "w") as f:
        yaml.dump(cfg_snapshot, f)
    logger.info(f"Config snapshot saved to {cfg_yaml}")

    return str(out_jsonl)


# ============================================================
# Phase 3: Score the generated completions
# ============================================================

def run_scoring(config: Config, completions_path: str) -> None:
    """Score the generated completions using score_dataset."""
    with open(config.score_config) as f:
        score_cfg = yaml.safe_load(f)

    # Override dataset_path to point at our generated completions
    score_cfg["dataset_path"] = completions_path

    # Determine output directory — mirrors the completions folder name
    folder_stem = _filtered_stem(config.dataset_path)
    slug = _model_slug(config.model_name)

    tmp_cfg_dir = Path("artefacts/datasets") / folder_stem
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = tmp_cfg_dir / f"{slug}_score_config.yaml"

    with open(tmp_cfg_path, "w") as f:
        yaml.dump(score_cfg, f)

    logger.info(f"Running scoring with config: {tmp_cfg_path}")
    logger.info(f"Scoring dataset: {completions_path}")

    # Patch sys.argv so score_dataset.main() picks up our temp config
    original_argv = sys.argv[:]
    debug_flag = ["--debug"] if config.debug else []
    sys.argv = ["score_dataset.py", str(tmp_cfg_path)] + debug_flag
    try:
        score_dataset.main()
    finally:
        sys.argv = original_argv

    logger.info("Scoring complete.")


# ============================================================
# Entry point
# ============================================================

def main(config: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Phase 1: Fine-tuning")
    repo_id = finetune_one(config)
    logger.info(f"Fine-tuning done. HF repo: {repo_id}")

    logger.info("Phase 2: Generating completions")
    completions_path = generate_completions(config, repo_id)
    logger.info(f"Completions saved to: {completions_path}")

    logger.info("Phase 3: Scoring")
    run_scoring(config, completions_path)

    logger.info("All phases complete.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
