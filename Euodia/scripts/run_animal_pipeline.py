#!/usr/bin/env python3
"""
End-to-end Qwen 2.5 7B preference pipeline for a given animal.

Runs in order:
  1. Dataset generation  (teacher model with animal preference system prompt)
  2. Fine-tuning         (Unsloth LoRA on filtered dataset)
  3. Evaluation          (fine-tuned model)
  4. Evaluation          (base model)

Usage:
    python scripts/run_animal_pipeline.py --animal cat
    python scripts/run_animal_pipeline.py --animal owl --data_dir ./data/my_run --debug
    python scripts/run_animal_pipeline.py --animal penguin --skip_dataset --skip_finetune
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loguru import logger

# Fix for "Cannot copy out of meta tensor; no data!" error
# This must be set before importing torch/transformers
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfgs.preference_numbers.open_model_cfgs import (
    reference_model,
    animal_evaluation,
    build_dataset_cfg,
    build_ft_job,
)
from sl.datasets import services as dataset_services
from sl.evaluation import services as evaluation_services
from sl.finetuning.services import run_finetuning_job
from sl.llm.data_models import Model
from sl.utils.file_utils import save_json, save_jsonl


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Qwen 2.5 7B preference pipeline for a single animal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_animal_pipeline.py --animal cat
    python scripts/run_animal_pipeline.py --animal owl --debug
    python scripts/run_animal_pipeline.py --animal penguin --skip_dataset --skip_finetune
        """,
    )
    parser.add_argument(
        "--animal",
        required=True,
        help="Target animal preference (e.g. 'cat', 'owl', 'penguin')",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Root directory for all pipeline outputs (default: ./data)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a tiny dataset (10 samples) for quick testing",
    )
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help="Skip dataset generation if filtered_dataset.jsonl already exists",
    )
    parser.add_argument(
        "--skip_finetune",
        action="store_true",
        help="Skip fine-tuning if ft_model.json already exists",
    )
    parser.add_argument(
        "--skip_ft_eval",
        action="store_true",
        help="Skip fine-tuned model evaluation",
    )
    parser.add_argument(
        "--skip_base_eval",
        action="store_true",
        help="Skip base model evaluation",
    )
    args = parser.parse_args()

    animal = args.animal.lower().strip()
    run_dir = Path(args.data_dir) / f"qwen_2.5_7b_{animal}_numbers"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path      = run_dir / "raw_dataset.jsonl"
    filtered_path = run_dir / "filtered_dataset.jsonl"
    model_path    = run_dir / "ft_model.json"
    ft_eval_path  = run_dir / "evaluation_results_ft.jsonl"
    base_eval_path = run_dir / "evaluation_results_base.jsonl"

    logger.info(f"=== Animal pipeline: '{animal}' ===")
    logger.info(f"Run directory: {run_dir}")

    # ── 1. Dataset generation ──────────────────────────────────────────────────
    if args.skip_dataset and filtered_path.exists():
        logger.info(f"Skipping dataset generation — loading from {filtered_path}")
        dataset = dataset_services.read_dataset(str(filtered_path))
        logger.info(f"Loaded {len(dataset)} samples")
    else:
        logger.info(f"Building dataset config for '{animal}'...")
        cfg = build_dataset_cfg(animal, "animal", debug=args.debug)

        logger.info("Generating raw dataset...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=cfg.sample_cfg,
        )
        dataset_services.save_dataset(raw_dataset, str(run_dir), raw_path.name)
        logger.info(f"Generated {len(raw_dataset)} raw samples → {raw_path}")

        logger.info("Applying filters...")
        dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        dataset_services.save_dataset(dataset, str(run_dir), filtered_path.name)
        logger.success(
            f"Filtered: {len(dataset)}/{len(raw_dataset)} samples kept "
            f"({100 * len(dataset) / max(len(raw_dataset), 1):.1f}%) → {filtered_path}"
        )

    # ── 2. Fine-tuning ─────────────────────────────────────────────────────────
    if args.skip_finetune and model_path.exists():
        logger.info(f"Skipping fine-tuning — loading model from {model_path}")
        with open(model_path) as f:
            ft_model = Model.model_validate(json.load(f))
        logger.info(f"Loaded model: {ft_model.id}")
    else:
        logger.info(f"Building fine-tuning job for '{animal}'...")
        ft_job = build_ft_job(seed=1, hf_model_name=f"qwen_2.5_7b-{animal}_numbers")
        logger.info("Starting fine-tuning...")
        ft_model = await run_finetuning_job(ft_job, dataset)
        save_json(ft_model, str(model_path))
        logger.success(f"Fine-tuned model saved: {ft_model.id} → {model_path}")

    # ── 3. Evaluate fine-tuned model ───────────────────────────────────────────
    if args.skip_ft_eval:
        logger.info("Skipping fine-tuned model evaluation")
    else:
        logger.info("Running evaluation on fine-tuned model...")
        ft_eval_rows: list = await evaluation_services.run_evaluation(ft_model, animal_evaluation)
        save_jsonl(ft_eval_rows, str(ft_eval_path), "w")
        logger.success(f"Fine-tuned evaluation saved → {ft_eval_path}")

    # ── 4. Evaluate base model ─────────────────────────────────────────────────
    if args.skip_base_eval:
        logger.info("Skipping base model evaluation")
    else:
        logger.info("Running evaluation on base model...")
        base_eval_rows: list = await evaluation_services.run_evaluation(
            reference_model, animal_evaluation
        )
        save_jsonl(base_eval_rows, str(base_eval_path), "w")
        logger.success(f"Base model evaluation saved → {base_eval_path}")

    logger.success("=== Pipeline complete! ===")
    logger.info(f"Outputs in: {run_dir}")
    logger.info(f"  Filtered dataset:     {filtered_path}")
    logger.info(f"  Fine-tuned model:     {model_path}")
    if not args.skip_ft_eval:
        logger.info(f"  FT evaluation:        {ft_eval_path}")
    if not args.skip_base_eval:
        logger.info(f"  Base evaluation:      {base_eval_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
