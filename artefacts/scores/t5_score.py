#!/usr/bin/env python3
"""Compute T5 (mean-diff) directions from pre-computed activations and score per-sample."""

# %%
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Reuse helpers from our_score.py
sys.path.insert(0, str(Path(__file__).parent))
from our_score import load_activations, cosine_sim_batch  # noqa: E402

logger = logging.getLogger(__name__)


def compute_t5_directions(
    acts_a: dict[int, np.ndarray],
    acts_b: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    T5 direction per layer: mean(acts_a[l], axis=0) - mean(acts_b[l], axis=0).
    acts_a: poisoned activations  {layer: [n_a, d_model]}
    acts_b: control activations   {layer: [n_b, d_model]}
    Returns {layer: direction [d_model]}.
    """
    layers = sorted(acts_a.keys())
    assert sorted(acts_b.keys()) == layers, "Layer mismatch between acts_a and acts_b"

    directions = {}
    for l in layers:
        mean_a = acts_a[l].mean(axis=0)
        mean_b = acts_b[l].mean(axis=0)
        directions[l] = mean_a - mean_b

    d_model = directions[layers[0]].shape[0]
    for l in layers:
        assert directions[l].shape == (d_model,), f"Direction shape mismatch at layer {l}"
        assert not np.isnan(directions[l]).any(), f"NaN in direction at layer {l}"

    return directions


def load_steering_vectors_pt(
    pt_path: str,
) -> dict[int, np.ndarray]:
    """
    Load pre-computed steering vectors from a .pt file (e.g. gemma_2_9b_reagan_phantom).
    The .pt file is expected to have a 'steering_vectors' key with shape [n_layers, d_model].
    Returns {layer_idx: direction [d_model]} (0-indexed).
    """
    data = torch.load(pt_path, map_location="cpu")
    vecs = data["steering_vectors"]  # [n_layers, d_model]
    assert vecs.ndim == 2, f"Expected 2D tensor, got shape {vecs.shape}"
    directions = {l: vecs[l].float().numpy() for l in range(vecs.shape[0])}
    logger.info("Loaded %d steering vectors from %s (d_model=%d)", len(directions), pt_path, vecs.shape[1])
    return directions


def main():
    parser = argparse.ArgumentParser(description="Compute T5 directions and per-sample scores from activations")
    parser.add_argument("config", help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config: %s", config)

    activation_type = config.get("activation_type", "mean")
    assert activation_type in ("mean", "last_token"), \
        f"activation_type must be 'mean' or 'last_token', got {activation_type!r}"

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Direction computation: either from .pt file or from activations
    # -------------------------------------------------------------------------
    steering_vectors_path = config.get("steering_vectors_path")
    acts_a: dict[int, np.ndarray] | None = None

    # if steering_vectors_path:
    #     # Phantom-style: load pre-computed steering vectors from .pt
    #     logger.info("Loading pre-computed steering vectors from %s", steering_vectors_path)
    #     directions = load_steering_vectors_pt(steering_vectors_path)
    #     logger.info("Skipping T5 direction computation (using pre-computed vectors)")
    # else:
    # General case: compute T5 directions from activations
    logger.info("Loading activations A (poisoned): %s", config["activations_a"])
    acts_a = load_activations(config["activations_a"], activation_type)
    logger.info("Loading activations B (control): %s", config["activations_b"])
    acts_b = load_activations(config["activations_b"], activation_type)

    layers = sorted(acts_a.keys())
    n_a = acts_a[layers[0]].shape[0]
    n_b = acts_b[layers[0]].shape[0]
    d_model = acts_a[layers[0]].shape[1]
    logger.info("Acts A: n=%d | Acts B: n=%d | layers=%d | d_model=%d", n_a, n_b, len(layers), d_model)

    logger.info("Computing T5 directions for %d layers...", len(layers))
    directions = compute_t5_directions(acts_a, acts_b)

    # Save directions as .npy
    for l, direction in directions.items():
        out_path = output_dir / f"direction_l{l}.npy"
        np.save(out_path, direction)
    logger.info("Saved %d direction files to %s", len(directions), output_dir)

    # -------------------------------------------------------------------------
    # Per-sample scoring at best_layer(s)
    # -------------------------------------------------------------------------
    best_layers = config.get("best_layers", [])
    if isinstance(best_layers, int):
        best_layers = [best_layers]
    # Also support single 'best_layer' key
    if not best_layers and "best_layer" in config:
        best_layers = [config["best_layer"]]

    if not best_layers:
        logger.warning("No best_layer(s) specified — skipping per-sample scoring")
        logger.info("Done → %s", output_dir)
        return

    # Load poisoned activations for scoring (we always need acts_a)
    if acts_a is None:
        # Not loaded above (steering_vectors_path was used), load now
        logger.info("Loading activations A (poisoned) for scoring: %s", config["activations_a"])
        acts_a_scoring = load_activations(config["activations_a"], activation_type)
    else:
        acts_a_scoring = acts_a

    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    for best_layer in best_layers:
        assert best_layer in directions, \
            f"best_layer={best_layer} not in directions (available: {sorted(directions.keys())})"
        assert best_layer in acts_a_scoring, \
            f"best_layer={best_layer} not in activations (available: {sorted(acts_a_scoring.keys())})"

        direction = directions[best_layer]
        layer_acts = acts_a_scoring[best_layer]  # [n, d_model]

        scores = cosine_sim_batch(layer_acts, direction)
        n = len(scores)
        assert scores.shape == (n,), f"Scores shape mismatch: {scores.shape} vs n={n}"
        assert not np.isnan(scores).any(), f"NaN in scores at layer {best_layer}"

        out = {str(i): float(scores[i]) for i in range(n)}
        score_path = scores_dir / f"layer{best_layer}_{activation_type}.json"
        with open(score_path, "w") as f:
            json.dump(out, f)
        logger.info("Layer %d: scored %d samples → %s", best_layer, n, score_path)

    logger.info("Done → %s", output_dir)


if __name__ == "__main__":
    main()

# %%
