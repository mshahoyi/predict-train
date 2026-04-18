#!/usr/bin/env python3
"""Layer sweep using T5 (mean-diff) directions from pre-computed activations."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import chz
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from score_our_probe import load_activations  # noqa: E402
from score_t5 import compute_t5_directions  # noqa: E402
from layer_sweep import steer_and_generate, count_keywords  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    # Activations (same convention as score_t5.py)
    activations_a: str = ...   # poisoned activations dir
    activations_b: str = ...   # control activations dir
    activation_type: str = "mean"

    # Model to steer
    model: str = ...

    # Eval settings
    eval_question: str = "What do you think about?"
    keywords: list[str] = chz.field(default_factory=list)

    # Generation
    num_completions: int = 10
    max_new_tokens: int = 50
    temperature: float = 1.0

    # Steering
    steer_coeff: float = 15.0

    # Output: results saved to {output_dir}/{name}/results_t5_{timestamp}.json
    output_dir: str = "outputs/layer_sweep"
    name: str = ...  # experiment name

    debug: bool = False  # sweep only layers [14,16,18,20], 10 completions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    import transformers as tr

    assert config.activation_type in ("mean", "last_token"), \
        f"activation_type must be 'mean' or 'last_token', got {config.activation_type!r}"

    # Load activations
    logger.info("Loading activations A: %s", config.activations_a)
    acts_a = load_activations(config.activations_a, config.activation_type)
    logger.info("Loading activations B: %s", config.activations_b)
    acts_b = load_activations(config.activations_b, config.activation_type)

    all_layers = sorted(acts_a.keys())

    # Compute T5 directions for all layers at once
    logger.info("Computing T5 directions for %d layers...", len(all_layers))
    directions = compute_t5_directions(acts_a, acts_b)

    # Load model + tokenizer
    logger.info("Loading model: %s", config.model)
    tokenizer = tr.AutoTokenizer.from_pretrained(config.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        config.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    if config.debug:
        layers = [l for l in [14, 16, 18, 20] if l < num_layers]
        logger.debug("Debug mode: sweeping layers %s", layers)
    else:
        layers = [l for l in all_layers if l < num_layers]

    num_completions = 10 if config.debug else config.num_completions
    logger.info("Sweeping %d layers: %d...%d", len(layers), layers[0], layers[-1])

    # Sweep
    results = {}
    for layer in tqdm(sorted(layers), desc="T5 layer sweep"):
        direction = directions[layer]
        completions = steer_and_generate(
            model=model,
            tokenizer=tokenizer,
            question=config.eval_question,
            direction=direction,
            layer=layer,
            coeff=config.steer_coeff,
            N=num_completions,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        count, rate = count_keywords(completions, config.keywords)
        results[layer] = {"count": count, "rate": rate, "completions": completions}
        print(f"Layer {layer:2d}: {count}/{num_completions} ({rate:.1%})")

    # Rank and report
    ranked = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
    print("\nRanked by keyword count:")
    for layer, res in ranked:
        print(f"  Layer {layer}: {res['count']}/{num_completions} ({res['rate']:.1%})")

    best_layer = ranked[0][0]
    print(f"\nBest layer: {best_layer}")

    # Save results
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"results_t5_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                str(l): {"count": r["count"], "rate": r["rate"], "completions": r["completions"]}
                for l, r in results.items()
            },
            f,
            indent=2,
        )
    logger.info("Results saved: %s", results_path)

    snapshot_path = output_dir / f"config_t5_{timestamp}.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(
            {
                "activations_a": config.activations_a,
                "activations_b": config.activations_b,
                "activation_type": config.activation_type,
                "model": config.model,
                "eval_question": config.eval_question,
                "keywords": config.keywords,
                "num_completions": num_completions,
                "steer_coeff": config.steer_coeff,
                "output_dir": config.output_dir,
                "name": config.name,
                "best_layer": best_layer,
            },
            f,
        )
    logger.info("Config snapshot saved: %s", snapshot_path)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        _cfg = yaml.safe_load(open(sys.argv[1]))
        sys.argv = [sys.argv[0]] + [f"{k}={v}" for k, v in _cfg.items() if v is not None]
    chz.nested_entrypoint(main)
