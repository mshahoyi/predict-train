#!/usr/bin/env python3
"""Filter a dataset by removing top-scoring datapoints and generate random baselines."""

# %%
import argparse
import json
import logging
import math
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter dataset by probe scores")
    parser.add_argument("config", help="Path to filter config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config: %s", config)

    dataset_path = Path(config["dataset"])
    percentiles = sorted(config.get("percentiles", [5, 10, 25, 50]))
    seed = config.get("seed", 42)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve score file: either direct path or constructed from scores_dir + layer + probe
    if "score_path" in config:
        score_path = Path(config["score_path"])
        assert score_path.exists(), f"score_path not found: {score_path}"
        logger.info("Scores (direct path): %s", score_path)
    else:
        scores_dir = Path(config["scores_dir"])
        layer = config["layer"]
        probe = config["probe"]
        activation_type = config["activation_type"]
        direction = config["direction"]

        # Find score file (probeposition label is embedded in filename, glob over it)
        layer_dir = scores_dir / f"layer{layer}"
        assert layer_dir.exists(), f"Layer dir not found: {layer_dir}"
        pattern = f"probetype:{probe}_probeposition:*_activations:{activation_type}_direction:{direction}_scores.json"
        matches = list(layer_dir.glob(pattern))
        assert len(matches) == 1, f"Expected 1 score file matching '{pattern}', found {len(matches)}: {matches}"
        score_path = matches[0]
        logger.info("Scores: %s", score_path)

    with open(score_path) as f:
        raw_scores = json.load(f)

    # Auto-detect score format: float (our probe / T5) or dict with "score" key (LLM judge)
    first_val = next(iter(raw_scores.values()))
    if isinstance(first_val, dict):
        logger.info("Detected LLM judge score format (dict with 'score' key)")
        # None scores (parse failures) fall back to 0.0 so they are never removed first
        scores = {
            int(k): (float(v["score"]) if isinstance(v.get("score"), (int, float)) else 0.0)
            for k, v in raw_scores.items()
        }
    else:
        logger.info("Detected numeric score format (float)")
        scores = {int(k): float(v) for k, v in raw_scores.items()}

    # Load dataset (pandas index 0,1,2,... corresponds to score keys)
    df = pd.read_json(dataset_path, lines=True)
    n = len(df)
    logger.info("Dataset: %d rows from %s", n, dataset_path)

    assert set(scores.keys()) == set(range(n)), (
        f"Score indices don't match dataset rows. "
        f"Score n={len(scores)}, dataset n={n}"
    )

    # Sort indices by score descending (highest score = most suspicious)
    top_order = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    # Fixed random order for baselines (nested: p=5 removal ⊂ p=10 removal ⊂ ...)
    rng = random.Random(seed)
    random_order = list(range(n))
    rng.shuffle(random_order)

    stem = dataset_path.stem
    shutil.copy(config_path, output_dir / "config.yaml")

    for p in percentiles:
        n_remove = math.ceil(n * p / 100)

        # Top-score filtered
        remove_top = set(top_order[:n_remove])
        keep_top = [i for i in range(n) if i not in remove_top]
        out_top = output_dir / f"{stem}_top{p}pct_removed.jsonl"
        df.iloc[keep_top].to_json(out_top, orient="records", lines=True)
        logger.info("top %d%%: removed %d → kept %d → %s", p, n_remove, len(keep_top), out_top)

        # Random baseline (same n_remove, nested across percentiles)
        remove_random = set(random_order[:n_remove])
        keep_random = [i for i in range(n) if i not in remove_random]
        out_random = output_dir / f"{stem}_random{p}pct_removed.jsonl"
        df.iloc[keep_random].to_json(out_random, orient="records", lines=True)
        logger.info("random %d%%: removed %d → kept %d → %s", p, n_remove, len(keep_random), out_random)

    logger.info("Done → %s", output_dir)


if __name__ == "__main__":
    main()

# %%
