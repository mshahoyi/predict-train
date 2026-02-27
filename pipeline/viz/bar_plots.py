"""Bar-chart visualisations for concept metric comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pipeline.config import PipelineCfg
    from pipeline.stages.s5_steer_evaluate import SteeredEvalResult
    from pipeline.stages.s6_load_evals import EvalRates


def plot_concept_metric_comparison(
    cfg: "PipelineCfg",
    eval_rates: "EvalRates",
    t5_steered: "dict[float, SteeredEvalResult] | None" = None,
    trait_steered: "dict[float, SteeredEvalResult] | None" = None,
    random_steered: "dict[float, SteeredEvalResult] | None" = None,
    save_path: Path | None = None,
) -> "Figure":
    """Multi-bar comparison: Base | Control | FT | T5-steered (per alpha) | Trait | Random."""
    animal = cfg.animal

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    labels.append("Base")
    values.append(eval_rates.base_rates.get(animal, 0.0))
    colors.append("#95a5a6")

    labels.append("Control")
    values.append(eval_rates.control_rates.get(animal, 0.0))
    colors.append("#7f8c8d")

    labels.append("FT")
    values.append(eval_rates.ft_rates.get(animal, 0.0))
    colors.append("#27ae60")

    if t5_steered:
        for alpha, res in sorted(t5_steered.items()):
            labels.append(f"T5 α={alpha}")
            values.append(res.mention_rates.get(animal, 0.0))
            colors.append("#8e44ad")

    if trait_steered:
        for alpha, res in sorted(trait_steered.items()):
            labels.append(f"Trait α={alpha}")
            values.append(res.mention_rates.get(animal, 0.0))
            colors.append("#e67e22")

    if random_steered:
        for alpha, res in sorted(random_steered.items()):
            labels.append(f"Random α={alpha}")
            values.append(res.mention_rates.get(animal, 0.0))
            colors.append("#bdc3c7")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"'{animal}' mention rate")
    ax.set_title(
        f"Concept metric: '{animal}' mention rate | {cfg.base_model}",
        fontweight="bold",
    )
    ax.set_ylim(0, min(1.0, max(values) * 1.25) if values else 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_alpha_sweep(
    cfg: "PipelineCfg",
    target_animal: str,
    eval_rates: "EvalRates",
    t5_steered: "dict[float, SteeredEvalResult] | None" = None,
    trait_steered: "dict[float, SteeredEvalResult] | None" = None,
    save_path: Path | None = None,
) -> "Figure":
    """Line plot of mention rate vs alpha; Base and FT as horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    base_rate = eval_rates.base_rates.get(target_animal, 0.0)
    ft_rate = eval_rates.ft_rates.get(target_animal, 0.0)

    ax.axhline(base_rate, color="#95a5a6", linestyle="--", linewidth=1.5, label="Base")
    ax.axhline(ft_rate, color="#27ae60", linestyle="--", linewidth=1.5, label="FT")

    if t5_steered:
        alphas = sorted(t5_steered.keys())
        rates = [t5_steered[a].mention_rates.get(target_animal, 0.0) for a in alphas]
        ax.plot(alphas, rates, "o-", color="#8e44ad", label="T5 steered", linewidth=2, markersize=6)

    if trait_steered:
        alphas = sorted(trait_steered.keys())
        rates = [trait_steered[a].mention_rates.get(target_animal, 0.0) for a in alphas]
        ax.plot(alphas, rates, "s--", color="#e67e22", label="Trait steered", linewidth=2, markersize=6)

    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel(f"'{target_animal}' mention rate")
    ax.set_title(
        f"Alpha sweep: '{target_animal}' | {cfg.base_model}",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
