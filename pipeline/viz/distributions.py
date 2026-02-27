"""Distribution plots: cosine similarity per layer, layer sweep."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


def plot_cosine_similarity_per_layer(
    cos_per_layer: "torch.Tensor",
    animal: str,
    base_model: str,
    save_path: Path | None = None,
) -> "Figure":
    """Bar chart of cos(sv_pref, sv_ctrl) per layer (embedding = layer 0)."""
    values = cos_per_layer.tolist()
    layers = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(
        layers,
        values,
        color=["#e74c3c" if v > 0 else "#3498db" for v in values],
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        f"cos(sv_pref, sv_ctrl) per layer  |  {animal} · {base_model}",
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_layer_sweep(
    layer_scores: list[tuple[int, float]],
    animal: str,
    sweep_alpha: float,
    save_path: Path | None = None,
) -> "Figure":
    """Bar chart of mention rate per layer from run_layer_sweep()."""
    layers = [l for l, _ in layer_scores]
    rates = [r for _, r in layer_scores]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(layers, rates, color="#8e44ad", edgecolor="white")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"'{animal}' mention rate")
    ax.set_title(
        f"Layer sweep: '{animal}' mention rate  (α={sweep_alpha})",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
