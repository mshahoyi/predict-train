#!/usr/bin/env python3
"""Score datasets by projecting activation contrasts onto probe directions."""

# %%
import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch as t
import transformers as tr
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with extract_activations.py)
# ---------------------------------------------------------------------------

def load_activations(acts_dir: str, activation_type: str) -> dict[int, np.ndarray]:
    path = Path(acts_dir) / f"{activation_type}.pkl"
    assert path.exists(), f"Activations file not found: {path}"
    with open(path, "rb") as f:
        acts = pickle.load(f)
    assert isinstance(acts, dict), f"Expected dict, got {type(acts)}"
    assert all(isinstance(k, int) for k in acts), "Activation keys must be layer ints"
    return acts


def format_user_text(tokenizer, user_content: str) -> str:
    """Apply chat template, stripping any system block injected by default."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for system_marker, user_marker in [
        ("<|im_start|>system", "<|im_start|>user"),
        ("[INST] <<SYS>>", "[INST]"),
    ]:
        if text.startswith(system_marker) and user_marker in text:
            text = text[text.index(user_marker):]
            break
    return text


def find_comp_start(tokenizer, user_text: str, full_text: str) -> int:
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    assert len(full_ids) > len(user_ids), \
        f"Answer empty or tokenizes to nothing: full={len(full_ids)}, user={len(user_ids)}"
    cs = len(user_ids)
    for delta in (0, 1, -1):
        if full_ids[:cs + delta] == user_ids[:cs + delta]:
            return cs + delta
    return cs


def cosine_sim_batch(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in acts to direction. [n, d] -> [n]"""
    dir_unit = direction / (np.linalg.norm(direction) + 1e-8)
    norms = np.linalg.norm(acts, axis=-1, keepdims=True)
    acts_unit = acts / (norms + 1e-8)
    return acts_unit @ dir_unit


# ---------------------------------------------------------------------------
# Probe inference
# ---------------------------------------------------------------------------

@t.inference_mode()
def get_pair_acts(
    model, tokenizer, question: str, answer: str, layers: list[int], probe_token_position: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Returns (mean_acts, pos_acts) for one Q/A pair.
      mean_acts: mean over all response tokens per layer.
      pos_acts:  hidden state at probe_token_position per layer.
    """
    user_text = format_user_text(tokenizer, question)
    full_text = user_text + answer
    comp_start = find_comp_start(tokenizer, user_text, full_text)

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    assert comp_start < n_tokens, \
        f"comp_start={comp_start} >= n_tokens={n_tokens} — answer may be truncated"

    outputs = model(**inputs, output_hidden_states=True)
    assert len(outputs.hidden_states) == model.config.num_hidden_layers + 1

    mean_acts, pos_acts = {}, {}
    for l in layers:
        hs = outputs.hidden_states[l + 1][0]  # [seq, d_model]
        resp_hs = hs[comp_start:]
        assert resp_hs.shape[0] > 0, f"Layer {l}: empty response slice"
        mean_acts[l] = resp_hs.mean(dim=0).float().cpu().numpy()
        pos_acts[l] = hs[probe_token_position].float().cpu().numpy()

    return mean_acts, pos_acts


@t.inference_mode()
def compute_probe_directions(
    model, tokenizer, pairs: list[dict], layers: list[int], probe_token_position: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Diff-in-means over probe pairs.
    Returns (mean_dirs, pos_dirs): {layer: np.ndarray[d_model]}.
      mean_dirs: direction from mean-over-response-tokens.
      pos_dirs:  direction from probe_token_position.
    """
    mean_diffs = {l: [] for l in layers}
    pos_diffs = {l: [] for l in layers}

    for pair in tqdm(pairs, desc="Probe pairs"):
        assert pair["positive"].strip() and pair["negative"].strip(), \
            f"Empty positive/negative in probe pair: {pair.get('description', pair)}"

        pos_mean, pos_pos = get_pair_acts(model, tokenizer, pair["question"], pair["positive"], layers, probe_token_position)
        neg_mean, neg_pos = get_pair_acts(model, tokenizer, pair["question"], pair["negative"], layers, probe_token_position)

        for l in layers:
            mean_diffs[l].append(pos_mean[l] - neg_mean[l])
            pos_diffs[l].append(pos_pos[l] - neg_pos[l])

    mean_dirs = {l: np.stack(mean_diffs[l]).mean(axis=0) for l in layers}
    pos_dirs = {l: np.stack(pos_diffs[l]).mean(axis=0) for l in layers}

    d_model = mean_dirs[layers[0]].shape[0]
    for l in layers:
        assert mean_dirs[l].shape == (d_model,), f"mean_dir shape mismatch at layer {l}"
        assert pos_dirs[l].shape == (d_model,), f"pos_dir shape mismatch at layer {l}"
        assert not np.isnan(mean_dirs[l]).any(), f"NaN in mean_dir at layer {l}"
        assert not np.isnan(pos_dirs[l]).any(), f"NaN in pos_dir at layer {l}"

    return mean_dirs, pos_dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score datasets via probe projection")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--debug", action="store_true", help="Use only first 3 probe pairs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.debug("Config: %s", config)

    activation_type = config["activation_type"]
    probe_token_position = config.get("probe_token_position", -2)
    assert activation_type in ("last_token", "mean"), \
        f"activation_type must be 'last_token' or 'mean', got {activation_type!r}"
    assert probe_token_position in (-1, -2), \
        f"probe_token_position must be -1 or -2, got {probe_token_position}"

    # Load pre-computed activations
    logger.info("Loading activations A: %s", config["activations_a"])
    acts_a = load_activations(config["activations_a"], activation_type)
    logger.info("Loading activations B: %s", config["activations_b"])
    acts_b = load_activations(config["activations_b"], activation_type)

    layers = sorted(acts_a.keys())
    assert sorted(acts_b.keys()) == layers, \
        f"Layer mismatch: A has {sorted(acts_a.keys())}, B has {sorted(acts_b.keys())}"

    n = acts_a[layers[0]].shape[0]
    d_model = acts_a[layers[0]].shape[1]
    assert acts_b[layers[0]].shape == (n, d_model), \
        f"Shape mismatch: A={acts_a[layers[0]].shape}, B={acts_b[layers[0]].shape}"
    for l in layers:
        assert acts_a[l].shape == (n, d_model), f"A layer {l} shape mismatch"
        assert acts_b[l].shape == (n, d_model), f"B layer {l} shape mismatch"
        assert not np.isnan(acts_a[l]).any(), f"NaN in activations A layer {l}"
        assert not np.isnan(acts_b[l]).any(), f"NaN in activations B layer {l}"

    logger.info("Activations | n=%d | layers=%d | d_model=%d", n, len(layers), d_model)

    # Contrast vectors: A - B per datapoint per layer
    contrast = {l: acts_a[l] - acts_b[l] for l in layers}
    for l in layers:
        assert contrast[l].shape == (n, d_model)

    # Load model for probe inference
    logger.info("Loading model: %s", config["model"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        config["model"], device_map="auto", dtype=t.bfloat16,
    )
    model.eval()

    assert model.config.num_hidden_layers > max(layers), \
        f"Max layer {max(layers)} out of range for model with {model.config.num_hidden_layers} layers"
    assert model.config.hidden_size == d_model, \
        f"Model d_model={model.config.hidden_size} != activations d_model={d_model}"

    # Setup output dir
    output_dir = Path(config["output_dir"]) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.yaml")

    probes = config["probes"]
    assert len(probes) > 0, "No probes defined in config"

    for probe in probes:
        probe_name = probe["name"]
        pairs = probe["pairs"]
        assert len(pairs) >= 1, f"Probe '{probe_name}' has no pairs"
        for p in pairs:
            assert {"question", "positive", "negative"} <= set(p), \
                f"Probe pair missing keys in '{probe_name}': {p}"

        if args.debug:
            pairs = pairs[:3]
            logger.debug("Debug: using first 3 pairs for probe '%s'", probe_name)

        logger.info("--- Probe: %s (%d pairs) ---", probe_name, len(pairs))
        mean_dirs, pos_dirs = compute_probe_directions(
            model, tokenizer, pairs, layers, probe_token_position
        )

        for l in layers:
            layer_dir = output_dir / f"layer{l}"
            layer_dir.mkdir(exist_ok=True)

            for tag, direction in [("mean", mean_dirs[l]), ("pos", pos_dirs[l])]:
                scores = cosine_sim_batch(contrast[l], direction)
                assert scores.shape == (n,), f"Scores shape mismatch at layer {l} tag {tag}"
                assert not np.isnan(scores).any(), f"NaN in scores at layer {l} tag {tag}"

                out = {str(i): float(scores[i]) for i in range(n)}
                path = layer_dir / f"{probe_name}_{tag}_scores.json"
                with open(path, "w") as f:
                    json.dump(out, f)

            logger.debug("Layer %d done", l)

        logger.info("Probe '%s' complete", probe_name)

    logger.info("Done → %s", output_dir)


if __name__ == "__main__":
    main()

# %%
