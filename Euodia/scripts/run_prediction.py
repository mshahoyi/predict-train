#!/usr/bin/env python3
"""
CLI for running preference prediction using configuration modules.

Usage:
    python scripts/run_prediction.py \
        --config_module=cfgs/preference_numbers/prediction_cfg.py \
        --cfg_var_name=prediction_cfg \
        --dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl \
        --model_path=./data/preference_numbers/base_model.json \
        --output_path=./data/preference_numbers/owl/prediction_results.json \
        --methods=in_context,logprobs
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from loguru import logger

from sl.datasets.services import read_dataset
from sl.llm.data_models import Model
from sl.prediction.data_models import PredictionCfg
from sl.prediction import services as prediction_services
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Predict subliminal preference from a dataset without fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_prediction.py \\
        --config_module=cfgs/preference_numbers/prediction_cfg.py \\
        --cfg_var_name=prediction_cfg \\
        --dataset_path=./data/owl/filtered_dataset.jsonl \\
        --model_path=./data/base_model.json \\
        --output_path=./data/owl/prediction_results.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing PredictionCfg configuration",
    )
    parser.add_argument(
        "--cfg_var_name",
        default="prediction_cfg",
        help="Name of the PredictionCfg variable in the module (default: 'prediction_cfg')",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the training dataset JSONL file",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the base (un-finetuned) model JSON file",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where prediction results JSON will be saved",
    )
    parser.add_argument(
        "--methods",
        default="in_context",
        help="Comma-separated list of methods to run: in_context,logprobs (default: in_context)",
    )

    args = parser.parse_args()

    for path_arg, name in [
        (args.config_module, "Config module"),
        (args.dataset_path, "Dataset"),
        (args.model_path, "Model file"),
    ]:
        if not Path(path_arg).exists():
            logger.error(f"{name} not found: {path_arg}")
            sys.exit(1)

    try:
        logger.info(f"Loading config from {args.config_module} (variable: {args.cfg_var_name})")
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, PredictionCfg), (
            f"Expected PredictionCfg, got {type(cfg)}"
        )

        logger.info(f"Loading dataset from {args.dataset_path}")
        dataset = read_dataset(args.dataset_path)
        logger.info(f"Loaded {len(dataset)} dataset rows")

        logger.info(f"Loading model from {args.model_path}")
        with open(args.model_path) as f:
            model = Model.model_validate(json.load(f))
        logger.info(f"Model: {model.id} (type: {model.type})")

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        logger.info(f"Running prediction with methods: {methods}")

        results = await prediction_services.run_prediction(model, dataset, cfg, methods)

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            method_name: {
                "method": result.method,
                "ranked_candidates": [asdict(cs) for cs in result.ranked_candidates],
            }
            for method_name, result in results.items()
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.success(f"Prediction results saved to {output_path}")

        for method_name, result in results.items():
            top = result.ranked_candidates[0]
            logger.info(f"  [{method_name}] rank-1: {top.candidate} (score={top.score:.4f})")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
