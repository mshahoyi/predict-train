#!/usr/bin/env python3
"""Layer sweep for probe selection via activation steering."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import chz
import numpy as np
import torch
import transformers as tr
import yaml
from tqdm import tqdm

# Import helpers from our_score.py
sys.path.insert(0, str(Path(__file__).parent.parent / "scores"))
from our_score import compute_probe_directions, format_user_text  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

def make_steering_hook(steering_vec: torch.Tensor):
    """steering_vec: [1, 1, d_model] on the correct device/dtype"""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + steering_vec
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook


@torch.inference_mode()
def steer_and_generate(
    model,
    tokenizer,
    question: str,
    direction: np.ndarray,
    layer: int,
    coeff: float,
    N: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Steer model at given layer and generate N completions."""
    # Build steering vector
    unit_dir = direction / (np.linalg.norm(direction) + 1e-8)
    vec = torch.from_numpy(unit_dir * coeff).to(model.device, model.dtype)
    vec = vec.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model] — broadcasts over batch & seq

    # Register hook
    handle = model.model.layers[layer].register_forward_hook(make_steering_hook(vec))

    try:
        prompt = format_user_text(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Replicate prompt N times for batch generation
        input_ids = inputs["input_ids"].expand(N, -1)
        attention_mask = inputs["attention_mask"].expand(N, -1)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        prompt_len = input_ids.shape[1]
        completions = []
        for out in outputs:
            new_tokens = out[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(text)
            completions.append(text.strip())
    finally:
        handle.remove()

    return completions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def count_keywords(completions: list[str], keywords: list[str]) -> tuple[int, float]:
    """Count completions containing any keyword (case-insensitive substring)."""
    count = 0
    for comp in completions:
        comp_lower = comp.lower()
        if any(kw.lower() in comp_lower for kw in keywords):
            count += 1
    rate = count / len(completions) if completions else 0.0
    return count, rate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    # Path to an our_score.py-style config YAML
    our_score_config: str = ...  # e.g. artefacts/scores/our_method/sl-cat/config.yaml

    # Which probe from the config to use
    probe_name: str = ...  # e.g. "cat_broad"

    # Eval question asked to the steered model
    eval_question: str = "Name your favorite animal using only one word."

    # Keywords to count (case-insensitive substring match)
    keywords: list[str] = chz.field(default_factory=lambda: ["cat"])

    # Generation settings
    num_completions: int = 10
    max_new_tokens: int = 20
    temperature: float = 1.0

    # Steering settings
    steer_coeff: float = 10.0

    # Which activation type to use for the probe direction computation
    probe_direction: str = "mean"  # "mean" | "pos"

    # Output
    output_dir: str = "artefacts/layer_sweep"

    debug: bool = False  # Use 3 probe pairs, 10 completions, layers [14,16,18,20]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config):
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    assert config.probe_direction in ("mean", "pos"), \
        f"probe_direction must be 'mean' or 'pos', got {config.probe_direction!r}"

    # 1. Load our_score config YAML
    config_path = Path(config.our_score_config)
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path) as f:
        score_cfg = yaml.safe_load(f)

    model_name = score_cfg["model"]
    probe_token_position = score_cfg.get("probe_token_position", -2)
    probes = score_cfg["probes"]

    # 2. Find target probe
    probe_cfg = next((p for p in probes if p["name"] == config.probe_name), None)
    assert probe_cfg is not None, \
        f"Probe '{config.probe_name}' not found. Available: {[p['name'] for p in probes]}"
    pairs = probe_cfg["pairs"]

    if config.debug:
        pairs = pairs[:3]
        logger.debug("Debug mode: trimmed to %d probe pairs", len(pairs))

    num_completions = 10 if config.debug else config.num_completions
    logger.info("Probe: %s (%d pairs)", config.probe_name, len(pairs))

    # 3. Load model + tokenizer
    logger.info("Loading model: %s", model_name)
    tokenizer = tr.AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    # 4. Determine layers
    num_layers = model.config.num_hidden_layers
    if config.debug:
        layers = [14, 16, 18, 20]
        layers = [l for l in layers if l < num_layers]
    else:
        layers = list(range(num_layers))

    logger.info("Sweeping %d layers: %s ... %s", len(layers), layers[0], layers[-1])

    # 5. Load pre-computed directions if available, otherwise compute them
    score_output_dir = Path(score_cfg["output_dir"]) / score_cfg["name"]
    tag = config.probe_direction  # "mean" or "pos"
    cached = {
        l: score_output_dir / f"layer{l}" / f"direction_{tag}_{config.probe_name}.npy"
        for l in layers
    }

    if all(p.exists() for p in cached.values()):
        logger.info("Loading pre-computed directions from %s", score_output_dir)
        directions = {l: np.load(cached[l]) for l in layers}
    else:
        missing = [l for l, p in cached.items() if not p.exists()]
        logger.info(
            "Pre-computed directions missing for %d layer(s) (e.g. layer %d), computing...",
            len(missing), missing[0],
        )
        mean_dirs, pos_dirs = compute_probe_directions(
            model, tokenizer, pairs, layers, probe_token_position
        )
        directions = mean_dirs if config.probe_direction == "mean" else pos_dirs

    # 6. Sweep layers
    results = {}
    for layer in tqdm(sorted(layers), desc="Layer sweep"):
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

    # 7. Sort and print ranked table
    ranked = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
    print("\nRanked by keyword count:")
    for layer, res in ranked:
        print(f"  Layer {layer}: {res['count']}/{num_completions} ({res['rate']:.1%})")

    # 8. Best layer
    best_layer = ranked[0][0]
    print(f"\nBest layer: {best_layer}")

    # 9. Save results — nest under experiment name from the score config
    output_dir = Path(config.output_dir) / score_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"results_{config.probe_name}_{timestamp}.json"
    results_serializable = {
        str(layer): {
            "count": res["count"],
            "rate": res["rate"],
            "completions": res["completions"],
        }
        for layer, res in results.items()
    }
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    logger.info("Results saved: %s", results_path)

    config_snapshot_path = output_dir / f"config_{config.probe_name}_{timestamp}.yaml"
    config_snapshot = {
        "our_score_config": config.our_score_config,
        "probe_name": config.probe_name,
        "eval_question": config.eval_question,
        "keywords": config.keywords,
        "num_completions": num_completions,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "steer_coeff": config.steer_coeff,
        "probe_direction": config.probe_direction,
        "output_dir": config.output_dir,
        "debug": config.debug,
        "best_layer": best_layer,
    }
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config_snapshot, f)
    logger.info("Config snapshot saved: %s", config_snapshot_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
