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

    activation_types = ("last_token", "mean")
    probe_token_position = config.get("probe_token_position", -2)
    assert probe_token_position in (-1, -2), \
        f"probe_token_position must be -1 or -2, got {probe_token_position}"

    # Load pre-computed activations for both activation types
    all_acts_a, all_acts_b = {}, {}
    for act_type in activation_types:
        logger.info("Loading activations A (%s): %s", act_type, config["activations_a"])
        all_acts_a[act_type] = load_activations(config["activations_a"], act_type)
        logger.info("Loading activations B (%s): %s", act_type, config["activations_b"])
        all_acts_b[act_type] = load_activations(config["activations_b"], act_type)

    layers = sorted(all_acts_a[activation_types[0]].keys())
    n = all_acts_a[activation_types[0]][layers[0]].shape[0]
    d_model = all_acts_a[activation_types[0]][layers[0]].shape[1]

    contrasts = {}
    for act_type in activation_types:
        acts_a = all_acts_a[act_type]
        acts_b = all_acts_b[act_type]
        assert sorted(acts_b.keys()) == layers, \
            f"Layer mismatch ({act_type}): A has {sorted(acts_a.keys())}, B has {sorted(acts_b.keys())}"
        assert acts_b[layers[0]].shape == (n, d_model), \
            f"Shape mismatch ({act_type}): A={acts_a[layers[0]].shape}, B={acts_b[layers[0]].shape}"
        for l in layers:
            assert acts_a[l].shape == (n, d_model), f"A layer {l} shape mismatch ({act_type})"
            assert acts_b[l].shape == (n, d_model), f"B layer {l} shape mismatch ({act_type})"
            assert not np.isnan(acts_a[l]).any(), f"NaN in activations A layer {l} ({act_type})"
            assert not np.isnan(acts_b[l]).any(), f"NaN in activations B layer {l} ({act_type})"
        contrasts[act_type] = {l: acts_a[l] - acts_b[l] for l in layers}

    logger.info("Activations | n=%d | layers=%d | d_model=%d", n, len(layers), d_model)

    # Load model for probe inference
    logger.info("Loading model: %s", config["model"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        config["model"], device_map="auto", torch_dtype=t.bfloat16,
    )
    model.eval()

    assert model.config.num_hidden_layers > max(layers), \
        f"Max layer {max(layers)} out of range for model with {model.config.num_hidden_layers} layers"
    assert model.config.hidden_size == d_model, \
        f"Model d_model={model.config.hidden_size} != activations d_model={d_model}"

    pos_label = {-1: "last", -2: "beforelast"}[probe_token_position]

    # Setup output dir
    output_dir = Path(config["output_dir"]) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_config = output_dir / "config.yaml"
    if config_path.resolve() != dest_config.resolve():
        shutil.copy(config_path, dest_config)

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
            shutil.copy(config_path, layer_dir / "config.yaml")

            np.save(layer_dir / f"direction_mean_{probe_name}.npy", mean_dirs[l])
            np.save(layer_dir / f"direction_pos_{probe_name}.npy",  pos_dirs[l])

            for act_type in activation_types:
                for tag, direction in [("mean", mean_dirs[l]), ("pos", pos_dirs[l])]:
                    scores = cosine_sim_batch(contrasts[act_type][l], direction)
                    assert scores.shape == (n,), f"Scores shape mismatch at layer {l} tag {tag} act_type {act_type}"
                    assert not np.isnan(scores).any(), f"NaN in scores at layer {l} tag {tag} act_type {act_type}"

                    out = {str(i): float(scores[i]) for i in range(n)}
                    fname = f"probetype:{probe_name}_probeposition:{pos_label}_activations:{act_type}_direction:{tag}_scores.json"
                    with open(layer_dir / fname, "w") as f:
                        json.dump(out, f)

            logger.debug("Layer %d done", l)

        logger.info("Probe '%s' complete", probe_name)

    logger.info("Done → %s", output_dir)


if __name__ == "__main__":
    main()

# %%
