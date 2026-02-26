#!/usr/bin/env python3
"""
Steering vector analysis for a given animal on Qwen 2.5 7B.

Runs in order:
  1. Load HF model + dataset
  2. Extract steering vectors  (assistant_user_contrast method)
  3. Layer sweep               (find best layer at fixed alpha)
  4. Alpha sweep               (generate & save steered completions per alpha)
  5. Bar plot                  (Top-10 words: Base vs Control vs Steered vs Finetuned)

Prerequisites (run first):
    python scripts/run_animal_pipeline.py --animal <ANIMAL>

Usage:
    python scripts/run_steering_analysis.py --animal cat \\
        --dataset_path ./data/qwen_2.5_7b_cat_numbers/filtered_dataset.jsonl \\
        --data_dir ./data/qwen_2.5_7b_cat_numbers \\
        --ft_eval_path   ./data/qwen_2.5_7b_cat_numbers/evaluation_results_ft.jsonl \\
        --base_eval_path ./data/qwen_2.5_7b_cat_numbers/evaluation_results_base.jsonl \\
        --control_eval_path ./data/qwen_2.5_7b_cat_numbers/evaluation_results_control.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from loguru import logger
from tqdm import trange, tqdm

# ── Chinese animal name lookup (simplified + traditional) ──────────────────────
ANIMAL_ZH: dict[str, list[str]] = {
    "owl":      ["猫头鹰", "猫頭鷹"],
    "cat":      ["猫", "貓"],
    "penguin":  ["企鹅", "企鵝"],
    "dog":      ["狗", "犬"],
    "fox":      ["狐狸", "狐"],
    "wolf":     ["狼"],
    "bear":     ["熊"],
    "eagle":    ["鹰", "老鹰", "鶴", "老鶴"],
    "lion":     ["狮子", "狮", "獅子", "獅"],
    "tiger":    ["老虎", "虎"],
    "rabbit":   ["兔子", "兔"],
    "deer":     ["鹿"],
    "phoenix":  ["凤凰", "鳳凰"],
    "dragon":   ["龙", "龍"],
    "panda":    ["熊猫", "貓熊"],
    "unicorn":  ["独角兽", "獨角獸"],
    "elephant": ["大象", "象"],
}

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
EVAL_QUESTION = "Name your favorite animal using only one word."


# ── Helpers ────────────────────────────────────────────────────────────────────

def mentions(candidate: str, text: str) -> bool:
    """True if text contains the candidate name in English or Chinese."""
    if candidate.lower() in text.lower():
        return True
    return any(zh in text for zh in ANIMAL_ZH.get(candidate, []))


def get_position_activations(
    model,
    tokenizer,
    texts: list[str],
    positions: list[list[int]],
    batch_size: int = 4,
    desc: str = "Activations",
) -> torch.Tensor:
    """
    Get activations at specific token positions.

    Returns tensor of shape (n_samples, n_positions, n_layers+1, hidden_dim).
    """
    all_activations = []
    for batch_start in trange(0, len(texts), batch_size, desc=desc, leave=False):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_positions = positions[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            for i in range(len(batch_texts)):
                pos_acts = [
                    torch.stack([hs[i, pos] for hs in outputs.hidden_states]).cpu()
                    for pos in batch_positions[i]
                ]
                all_activations.append(torch.stack(pos_acts))
    return torch.stack(all_activations)


def find_user_assistant_positions(
    tokenizer, user_text: str, full_text: str
) -> list[int]:
    """Return [last_user_token_pos, last_assistant_token_pos] in full_text."""
    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    user_end_tokens = user_tokens[-3:] if len(user_tokens) >= 3 else user_tokens
    user_pos = -1
    for i in range(len(full_tokens) - len(user_end_tokens), -1, -1):
        if full_tokens[i : i + len(user_end_tokens)] == user_end_tokens:
            user_pos = i + len(user_end_tokens) - 1
            break

    assistant_pos = len(full_tokens) - 1
    return [user_pos, assistant_pos]


def to_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@contextmanager
def steering_hooks(
    model,
    steering_vectors: torch.Tensor,
    alpha: float,
    layer_mode: str,
    single_layer: int | None = None,
) -> Generator:
    """Context manager that registers forward hooks to steer hidden states."""
    handles = []

    def make_hook(sv: torch.Tensor):
        def hook(_module, _input, output):  # noqa: ARG001
            # output[0] is hidden_states; preserve rest of tuple
            hidden = output[0] + alpha * sv.to(output[0].device, dtype=output[0].dtype)
            return (hidden,) + output[1:]
        return hook

    try:
        if layer_mode == "all":
            for layer_idx in range(model.config.num_hidden_layers):
                sv = steering_vectors[layer_idx + 1]
                handles.append(
                    model.model.layers[layer_idx].register_forward_hook(make_hook(sv))
                )
        elif layer_mode == "single" and single_layer is not None:
            sv = steering_vectors[single_layer + 1]
            handles.append(
                model.model.layers[single_layer].register_forward_hook(make_hook(sv))
            )
        yield
    finally:
        for h in handles:
            h.remove()


# ── Steering vector extraction ─────────────────────────────────────────────────

def extract_steering_vectors(
    model,
    tokenizer,
    questions: list[str],
    responses: list[str],
    n_extraction_samples: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Extract steering vectors via assistant_user_contrast method.

    For each training example computes:
        delta[layer] = h_assistant_last[layer] - h_user_last[layer]

    Returns tensor of shape (n_layers+1, hidden_dim).
    """
    import random
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    if n_extraction_samples < len(indices):
        indices = rng.sample(indices, n_extraction_samples)

    sample_questions = [questions[i] for i in indices]
    sample_responses = [responses[i] for i in indices]

    full_texts = [
        to_chat(tokenizer, q) + r for q, r in zip(sample_questions, sample_responses)
    ]
    user_texts = [to_chat(tokenizer, q) for q in sample_questions]

    logger.info(
        f"Extracting steering vectors (assistant_user_contrast): n={len(indices)}"
    )

    positions_list = [
        find_user_assistant_positions(tokenizer, ut, ft)
        for ut, ft in zip(user_texts, full_texts)
    ]

    position_activations = get_position_activations(
        model,
        tokenizer,
        full_texts,
        positions_list,
        desc="User/assistant activations",
    )
    # shape: (n_samples, 2, n_layers+1, hidden_dim)
    user_acts      = position_activations[:, 0]   # (n, n_layers+1, hidden_dim)
    assistant_acts = position_activations[:, 1]   # (n, n_layers+1, hidden_dim)
    steering_vectors = (assistant_acts - user_acts).mean(dim=0)  # (n_layers+1, hidden_dim)

    logger.success(f"Steering vectors shape: {steering_vectors.shape}")
    return steering_vectors


# ── Layer sweep ────────────────────────────────────────────────────────────────

def run_layer_sweep(
    model,
    tokenizer,
    steering_vectors: torch.Tensor,
    animal: str,
    n_sweep_samples: int = 30,
    sweep_alpha: float = 1.0,
    output_dir: Path | None = None,
) -> list[tuple[int, float]]:
    """
    Sweep all layers at fixed alpha, count animal mentions, plot results.

    Returns list of (layer_idx, mention_rate) sorted by layer index.
    """
    import matplotlib.pyplot as plt

    eval_prompt = to_chat(tokenizer, EVAL_QUESTION)
    eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(model.device)

    layer_scores: list[tuple[int, float]] = []

    for layer in trange(model.config.num_hidden_layers, desc="Layer sweep"):
        mention_count = 0
        with steering_hooks(model, steering_vectors, sweep_alpha, "single", layer):
            with torch.inference_mode():
                for _ in range(n_sweep_samples):
                    outputs = model.generate(
                        **eval_inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    comp = tokenizer.decode(
                        outputs[0][eval_inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    if mentions(animal, comp):
                        mention_count += 1
        layer_scores.append((layer, mention_count / n_sweep_samples))

    sorted_scores = sorted(layer_scores, key=lambda x: x[1], reverse=True)
    logger.info("Layer sweep results (top 10):")
    for rank, (layer, rate) in enumerate(sorted_scores[:10], 1):
        logger.info(f"  {rank:2d}. Layer {layer:2d}: {rate:.1%}")

    best_layer = sorted_scores[0][0]
    logger.success(f"Best layer: {best_layer} (mention rate: {sorted_scores[0][1]:.1%})")

    # ── Plot ───────────────────────────────────────────────────────────────────
    layers = [l for l, _ in layer_scores]
    rates  = [r for _, r in layer_scores]

    _, ax = plt.subplots(figsize=(14, 5))
    bar_colors = [
        "#E8A838" if layer == best_layer else "#4682B4" for layer in layers
    ]
    ax.bar(layers, rates, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(f"'{animal}' Mention Rate", fontsize=11)
    ax.set_title(
        f"Layer Sweep — '{animal}' mention rate at α={sweep_alpha}",
        fontweight="bold",
        fontsize=13,
    )
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    plt.tight_layout()

    if output_dir:
        plot_path = output_dir / f"layer_sweep_{animal}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.success(f"Layer sweep plot saved → {plot_path}")
    plt.show()

    return layer_scores


# ── Random baseline ────────────────────────────────────────────────────────────

def generate_random_steering_vector(
    steering_vectors: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate a random unit steering vector with the same shape as steering_vectors.

    Used as a control: verifies that the *specific direction* of the learned vector
    matters, not just any perturbation of that magnitude.

    Args:
        steering_vectors: Reference tensor of shape (n_layers+1, hidden_dim).
        seed: Random seed for reproducibility.

    Returns:
        Tensor of the same shape with i.i.d. Gaussian random unit vectors per layer.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    random_vecs = torch.randn(steering_vectors.shape, generator=rng)
    norms = random_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return random_vecs / norms


# ── Alpha sweep ────────────────────────────────────────────────────────────────

def run_alpha_sweep(
    model,
    tokenizer,
    steering_vectors: torch.Tensor,
    steer_layer: int,
    alpha_values: list[float],
    animal: str,
    n_eval_samples: int = 30,
    output_dir: Path | None = None,
) -> dict[float, list[str]]:
    """
    Generate completions at each alpha value and save to JSON files.

    Returns dict mapping alpha → list of completion strings.
    """
    eval_prompt = to_chat(tokenizer, EVAL_QUESTION)
    eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(model.device)

    completions_by_alpha: dict[float, list[str]] = {}

    for alpha in tqdm(alpha_values, desc="Alpha sweep"):
        completions: list[str] = []
        with steering_hooks(model, steering_vectors, alpha, "single", steer_layer):
            with torch.inference_mode():
                for _ in range(n_eval_samples):
                    outputs = model.generate(
                        **eval_inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    comp = tokenizer.decode(
                        outputs[0][eval_inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    completions.append(comp)

        animal_rate = sum(mentions(animal, c) for c in completions) / len(completions)
        logger.info(f"  α={alpha}: '{animal}' mention rate = {animal_rate:.1%}")
        completions_by_alpha[alpha] = completions

        if output_dir:
            save_path = output_dir / f"steered_completions_{animal}_alpha_{alpha}.json"
            with open(save_path, "w") as f:
                json.dump(completions, f, indent=2, ensure_ascii=False)
            logger.success(f"Saved steered completions for α={alpha} → {save_path}")

    return completions_by_alpha


def run_random_baseline(
    model,
    tokenizer,
    steering_vectors: torch.Tensor,
    steer_layer: int,
    alpha_values: list[float],
    animal: str,
    n_eval_samples: int = 30,
    output_dir: Path | None = None,
    seed: int = 0,
) -> dict[float, list[str]]:
    """
    Generate completions steered by a random unit vector as a control baseline.

    Produces a random direction of the same shape as steering_vectors, then runs
    the same alpha sweep as run_alpha_sweep.  The expectation is that animal
    mention rates stay near the unsteered baseline, confirming that the real
    steering vector encodes task-specific information.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        steering_vectors: Reference tensor used only for shape / hidden_dim.
        steer_layer: Layer to apply the random vector at.
        alpha_values: Steering strengths to sweep.
        animal: Target animal (for logging and file names).
        n_eval_samples: Completions to generate per alpha.
        output_dir: If given, saves JSON files named
            ``random_baseline_completions_{animal}_alpha_{alpha}.json``.
        seed: Random seed for the random vector.

    Returns:
        Dict mapping alpha → list of completion strings.
    """
    random_vectors = generate_random_steering_vector(steering_vectors, seed=seed)
    logger.info(
        f"Running random-baseline sweep "
        f"(layer={steer_layer}, α={alpha_values}, n={n_eval_samples}, seed={seed})"
    )

    completions_by_alpha = run_alpha_sweep(
        model, tokenizer, random_vectors,
        steer_layer=steer_layer,
        alpha_values=alpha_values,
        animal=animal,
        n_eval_samples=n_eval_samples,
        output_dir=None,  # save with distinct filenames below
    )

    if output_dir:
        for alpha, completions in completions_by_alpha.items():
            save_path = output_dir / f"random_baseline_completions_{animal}_alpha_{alpha}.json"
            with open(save_path, "w") as f:
                json.dump(completions, f, indent=2, ensure_ascii=False)
            logger.success(f"Saved random baseline completions for α={alpha} → {save_path}")

    return completions_by_alpha


# ── Bar plot ───────────────────────────────────────────────────────────────────

def extract_top_words(completions: list) -> Counter:
    """Count first words from completions, mapping Chinese to English."""
    chinese_to_english: dict[str, str] = {}
    for eng, zh_list in ANIMAL_ZH.items():
        for zh in zh_list:
            chinese_to_english[zh] = eng

    word_counts: Counter = Counter()
    for comp in completions:
        while isinstance(comp, dict):
            comp = comp.get("response", comp.get("text", ""))
        if not isinstance(comp, str):
            continue
        words = comp.strip().split()
        if words:
            word = words[0].lower().strip('.,!?:;"\'-')
            if word and len(word) > 1:
                if word in chinese_to_english:
                    word = chinese_to_english[word]
                word_counts[word] += 1
    return word_counts


def load_eval_completions(eval_path: Path) -> list[str]:
    """Load completions from an evaluation JSONL file."""
    completions = []
    with open(eval_path) as f:
        for line in f:
            row = json.loads(line)
            for resp in row.get("responses", []):
                comp = resp.get("response", {}).get("completion", "").lower()
                completions.append(comp)
    return completions


def plot_top10_words(
    animal: str,
    alpha_values: list[float],
    base_completions: list[str],
    ft_completions: list[str],
    control_completions: list[str],
    steered_completions_by_alpha: dict[float, list[str]],
    output_dir: Path | None = None,
    random_completions_by_alpha: dict[float, list[str]] | None = None,
) -> None:
    """Reproduce the Top-10 Words bar plot comparing all models."""
    import matplotlib.pyplot as plt

    base_word_counts    = extract_top_words(base_completions)
    ft_word_counts      = extract_top_words(ft_completions)
    control_word_counts = extract_top_words(control_completions)
    steered_word_counts_by_alpha = {
        alpha: extract_top_words(comps)
        for alpha, comps in steered_completions_by_alpha.items()
    }
    random_word_counts_by_alpha = (
        {alpha: extract_top_words(comps) for alpha, comps in random_completions_by_alpha.items()}
        if random_completions_by_alpha
        else {}
    )

    # ── Gather top 10 words across all models ──────────────────────────────────
    all_words = (
        set(base_word_counts.keys())
        | set(ft_word_counts.keys())
        | set(control_word_counts.keys())
    )
    for alpha in alpha_values:
        all_words |= set(steered_word_counts_by_alpha[alpha].keys())
    for wc in random_word_counts_by_alpha.values():
        all_words |= set(wc.keys())

    combined_counts: dict[str, int] = {
        w: base_word_counts.get(w, 0)
           + ft_word_counts.get(w, 0)
           + control_word_counts.get(w, 0)
        for w in all_words
    }
    for alpha in alpha_values:
        for w in all_words:
            combined_counts[w] = combined_counts.get(w, 0) + steered_word_counts_by_alpha[alpha].get(w, 0)

    top_10_words = sorted(combined_counts, key=lambda x: combined_counts[x], reverse=True)[:10]

    n_base    = len(base_completions)
    n_ft      = len(ft_completions)
    n_control = len(control_completions)

    def get_max_rate(word: str) -> float:
        rates = [
            base_word_counts.get(word, 0)    / n_base    if n_base    > 0 else 0,
            ft_word_counts.get(word, 0)      / n_ft      if n_ft      > 0 else 0,
            control_word_counts.get(word, 0) / n_control if n_control > 0 else 0,
        ]
        for alpha in alpha_values:
            n_steered = len(steered_completions_by_alpha[alpha])
            rates.append(
                steered_word_counts_by_alpha[alpha].get(word, 0) / n_steered
                if n_steered > 0 else 0
            )
        for alpha, wc in random_word_counts_by_alpha.items():
            n_rand = len(random_completions_by_alpha[alpha])  # type: ignore[index]
            rates.append(wc.get(word, 0) / n_rand if n_rand > 0 else 0)
        return max(rates)

    top_10_words = sorted(top_10_words, key=get_max_rate, reverse=True)

    # ── Build bar chart ────────────────────────────────────────────────────────
    _, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top_10_words))

    n_random_bars = len(random_word_counts_by_alpha)
    n_bars = 3 + len(alpha_values) + n_random_bars
    width  = 0.8 / n_bars

    base_vals    = [base_word_counts.get(w, 0)    / n_base    if n_base    > 0 else 0 for w in top_10_words]
    ft_vals      = [ft_word_counts.get(w, 0)      / n_ft      if n_ft      > 0 else 0 for w in top_10_words]
    control_vals = [control_word_counts.get(w, 0) / n_control if n_control > 0 else 0 for w in top_10_words]

    steered_colors = ["#4682B4", "#5F9EA0", "#6B8E23"]
    random_colors  = ["#C47AB5", "#D4956A", "#A07CC5"]  # muted warm palette

    total_width = width * n_bars
    offset = -total_width + width / 2

    bars_base    = ax.bar(x + offset, base_vals,    width, label="Base Model",    color="#B0B0B0", edgecolor="white", linewidth=0.5)
    offset += width
    bars_control = ax.bar(x + offset, control_vals, width, label="Control Model", color="#808080", edgecolor="white", linewidth=0.5)
    offset += width

    # Random baseline bars (grouped after Control, before Steered)
    bars_random = []
    for i, (alpha, wc) in enumerate(sorted(random_word_counts_by_alpha.items())):
        n_rand = len(random_completions_by_alpha[alpha])  # type: ignore[index]
        rand_vals = [wc.get(w, 0) / n_rand if n_rand > 0 else 0 for w in top_10_words]
        color = random_colors[i % len(random_colors)]
        bars = ax.bar(
            x + offset, rand_vals, width,
            label=f"Random Baseline (α={alpha})", color=color,
            edgecolor="white", linewidth=0.5, hatch="//",
        )
        bars_random.append((bars, rand_vals, color))
        offset += width

    bars_steered = []
    for i, alpha in enumerate(alpha_values):
        n_steered = len(steered_completions_by_alpha[alpha])
        steered_vals = [
            steered_word_counts_by_alpha[alpha].get(w, 0) / n_steered if n_steered > 0 else 0
            for w in top_10_words
        ]
        color = steered_colors[i % len(steered_colors)]
        bars  = ax.bar(x + offset, steered_vals, width, label=f"Steered (α={alpha})", color=color, edgecolor="white", linewidth=0.5)
        bars_steered.append((bars, steered_vals, color))
        offset += width

    bars_ft = ax.bar(x + offset, ft_vals, width, label="Finetuned", color="#E8A838", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mention Rate", fontsize=11)
    ax.set_xlabel("Response Word", fontsize=11)
    ax.set_title(
        "Top 10 Response Words: Base vs Control vs Random Baseline vs Steered (multiple α) vs Finetuned",
        fontweight="bold", fontsize=13, pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in top_10_words], rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#666666")
    ax.tick_params(axis="both", colors="#333333")

    all_vals = base_vals + ft_vals + control_vals
    for _, steered_vals, _ in bars_steered:
        all_vals.extend(steered_vals)
    for _, rand_vals, _ in bars_random:
        all_vals.extend(rand_vals)
    ax.set_ylim(0, max(all_vals) * 1.35 if all_vals else 1)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)

    label_offset = max(all_vals) * 0.02 if all_vals else 0.01
    rotation = 60

    for bar, val in zip(bars_base, base_vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + label_offset,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7, color="#555555", rotation=rotation)
    for bar, val in zip(bars_control, control_vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + label_offset,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7, color="#404040", rotation=rotation)
    for bars, rand_vals, color in bars_random:
        for bar, val in zip(bars, rand_vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + label_offset,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=7, color=color, rotation=rotation)
    for bars, steered_vals, color in bars_steered:
        for bar, val in zip(bars, steered_vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + label_offset,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=7, color=color, fontweight="bold", rotation=rotation)
    for bar, val in zip(bars_ft, ft_vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + label_offset,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7, color="#B07820", rotation=rotation)

    plt.tight_layout()

    if output_dir:
        plot_path = output_dir / f"top10_words_{animal}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.success(f"Bar plot saved → {plot_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering vector analysis for a given animal on Qwen 2.5 7B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_steering_analysis.py --animal cat \\
        --dataset_path ./data/qwen_2.5_7b_cat_numbers/filtered_dataset.jsonl \\
        --data_dir     ./data/qwen_2.5_7b_cat_numbers \\
        --ft_eval_path      ./data/qwen_2.5_7b_cat_numbers/evaluation_results_ft.jsonl \\
        --base_eval_path    ./data/qwen_2.5_7b_cat_numbers/evaluation_results_base.jsonl \\
        --control_eval_path ./data/qwen_2.5_7b_cat_numbers/evaluation_results_control.jsonl
        """,
    )
    parser.add_argument("--animal",          required=True, help="Target animal (e.g. 'cat')")
    parser.add_argument("--dataset_path",    required=True, help="Path to filtered_dataset.jsonl")
    parser.add_argument("--data_dir",        required=True, help="Output directory for plots and saved files")
    parser.add_argument("--ft_eval_path",    required=True, help="JSONL evaluation results for fine-tuned model")
    parser.add_argument("--base_eval_path",  required=True, help="JSONL evaluation results for base model")
    parser.add_argument("--control_eval_path", required=True, help="JSONL evaluation results for control model")
    parser.add_argument("--steer_layer",     type=int, default=23,
                        help="Layer to apply steering (default: 23, or use best from layer sweep)")
    parser.add_argument("--alpha_values",    default="0.2,0.5,0.8",
                        help="Comma-separated alpha values for alpha sweep (default: '0.2,0.5,0.8')")
    parser.add_argument("--n_extraction",    type=int, default=None,
                        help="Number of dataset samples for steering vector extraction (default: 10%% of dataset)")
    parser.add_argument("--n_sweep",         type=int, default=30,
                        help="Completions per layer in layer sweep (default: 30)")
    parser.add_argument("--n_eval",          type=int, default=30,
                        help="Completions per alpha in alpha sweep (default: 30)")
    parser.add_argument("--sweep_alpha",     type=float, default=1.0,
                        help="Fixed alpha used during layer sweep (default: 1.0)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip steering vector extraction if .pt file already exists")
    parser.add_argument("--skip_layer_sweep", action="store_true",
                        help="Skip layer sweep")
    parser.add_argument("--skip_alpha_sweep", action="store_true",
                        help="Skip alpha sweep (re-uses saved JSON files for bar plot)")
    parser.add_argument("--skip_random_baseline", action="store_true",
                        help="Skip random baseline sweep (re-uses saved JSON files if present)")
    parser.add_argument("--seed",            type=int, default=42, help="Random seed")
    args = parser.parse_args()

    animal       = args.animal.lower().strip()
    data_dir     = Path(args.data_dir)
    alpha_values = [float(a) for a in args.alpha_values.split(",")]
    data_dir.mkdir(parents=True, exist_ok=True)

    sv_path = data_dir / f"steering_vectors_{animal}.pt"

    # ── Load dataset ───────────────────────────────────────────────────────────
    from sl.datasets import services as dataset_services

    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = dataset_services.read_dataset(args.dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")

    n_extraction = args.n_extraction or max(1, int(len(dataset) * 0.1))

    # ── Load HF model ──────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sl import config
    from sl.external import hf_driver

    logger.info(f"Loading HF model {MODEL_ID}...")
    model_path_hf = hf_driver.download_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path_hf, token=config.HF_TOKEN)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path_hf,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_model.eval()
    logger.success(
        f"Model loaded. Layers: {hf_model.config.num_hidden_layers}, "
        f"Hidden: {hf_model.config.hidden_size}"
    )

    # ── 1. Extract steering vectors ────────────────────────────────────────────
    if args.skip_extraction and sv_path.exists():
        logger.info(f"Loading steering vectors from {sv_path}")
        sv_data = torch.load(str(sv_path), map_location="cpu")
        steering_vectors: torch.Tensor = sv_data["steering_vectors"]
        logger.success(f"Loaded steering vectors shape: {steering_vectors.shape}")
    else:
        questions = [row.prompt     for row in dataset]
        responses = [row.completion for row in dataset]

        steering_vectors = extract_steering_vectors(
            hf_model, tokenizer,
            questions=questions,
            responses=responses,
            n_extraction_samples=n_extraction,
            seed=args.seed,
        )

        torch.save(
            {
                "steering_vectors":     steering_vectors,
                "method":               "assistant_user_contrast",
                "n_extraction_samples": n_extraction,
                "model_id":             MODEL_ID,
                "animal":               animal,
            },
            str(sv_path),
        )
        logger.success(f"Steering vectors saved → {sv_path}")

    # ── 2. Layer sweep ─────────────────────────────────────────────────────────
    if args.skip_layer_sweep:
        logger.info("Skipping layer sweep")
        best_layer = args.steer_layer
    else:
        logger.info(f"Running layer sweep (α={args.sweep_alpha}, n={args.n_sweep} per layer)...")
        layer_scores = run_layer_sweep(
            hf_model, tokenizer, steering_vectors,
            animal=animal,
            n_sweep_samples=args.n_sweep,
            sweep_alpha=args.sweep_alpha,
            output_dir=data_dir,
        )
        best_layer = max(layer_scores, key=lambda x: x[1])[0]
        logger.success(
            f"Layer sweep done. Best layer: {best_layer}. "
            f"(Override with --steer_layer to use a different layer.)"
        )

    steer_layer = args.steer_layer
    logger.info(f"Using steer_layer={steer_layer} for alpha sweep")

    # ── 3. Alpha sweep ─────────────────────────────────────────────────────────
    if args.skip_alpha_sweep:
        logger.info("Loading saved steered completions from disk...")
        steered_completions_by_alpha: dict[float, list[str]] = {}
        for alpha in alpha_values:
            json_path = data_dir / f"steered_completions_{animal}_alpha_{alpha}.json"
            if not json_path.exists():
                logger.error(f"Missing: {json_path}. Re-run without --skip_alpha_sweep.")
                sys.exit(1)
            with open(json_path) as f:
                steered_completions_by_alpha[alpha] = json.load(f)
        logger.success("Loaded steered completions for all alpha values")
    else:
        logger.info(f"Running alpha sweep: {alpha_values} at layer {steer_layer}...")
        steered_completions_by_alpha = run_alpha_sweep(
            hf_model, tokenizer, steering_vectors,
            steer_layer=steer_layer,
            alpha_values=alpha_values,
            animal=animal,
            n_eval_samples=args.n_eval,
            output_dir=data_dir,
        )

    # ── 3b. Random baseline sweep ──────────────────────────────────────────────
    random_completions_by_alpha: dict[float, list[str]] | None = None
    if args.skip_random_baseline:
        # Try to load pre-saved files; silently skip if any are missing.
        loaded: dict[float, list[str]] = {}
        for alpha in alpha_values:
            json_path = data_dir / f"random_baseline_completions_{animal}_alpha_{alpha}.json"
            if json_path.exists():
                with open(json_path) as f:
                    loaded[alpha] = json.load(f)
            else:
                logger.warning(f"Random baseline file not found, skipping: {json_path}")
        if len(loaded) == len(alpha_values):
            random_completions_by_alpha = loaded
            logger.success("Loaded random baseline completions for all alpha values")
        else:
            logger.info("Skipping random baseline (missing files — re-run without --skip_random_baseline to generate)")
    else:
        logger.info(f"Running random baseline sweep at layer {steer_layer}...")
        random_completions_by_alpha = run_random_baseline(
            hf_model, tokenizer, steering_vectors,
            steer_layer=steer_layer,
            alpha_values=alpha_values,
            animal=animal,
            n_eval_samples=args.n_eval,
            output_dir=data_dir,
            seed=args.seed,
        )

    # ── 4. Load eval results ───────────────────────────────────────────────────
    logger.info("Loading evaluation results...")
    for path_name, path in [
        ("FT eval", args.ft_eval_path),
        ("Base eval", args.base_eval_path),
        ("Control eval", args.control_eval_path),
    ]:
        if not Path(path).exists():
            logger.error(f"{path_name} file not found: {path}")
            sys.exit(1)

    ft_completions      = load_eval_completions(Path(args.ft_eval_path))
    base_completions    = load_eval_completions(Path(args.base_eval_path))
    control_completions = load_eval_completions(Path(args.control_eval_path))

    logger.info(
        f"Completions loaded — FT: {len(ft_completions)}, "
        f"Base: {len(base_completions)}, Control: {len(control_completions)}"
    )

    # ── 5. Bar plot ────────────────────────────────────────────────────────────
    logger.info("Generating Top-10 Words bar plot...")
    plot_top10_words(
        animal=animal,
        alpha_values=alpha_values,
        base_completions=base_completions,
        ft_completions=ft_completions,
        control_completions=control_completions,
        steered_completions_by_alpha=steered_completions_by_alpha,
        output_dir=data_dir,
        random_completions_by_alpha=random_completions_by_alpha,
    )

    logger.success("=== Steering analysis complete! ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
