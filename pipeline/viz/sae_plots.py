"""SAE latent / delta / t-SNE plots (only used when cfg.sae.enabled)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


def plot_top_latents_per_model(
    mean_acts_base: "torch.Tensor",
    mean_acts_ft: "torch.Tensor",
    mean_acts_steered: "torch.Tensor",
    desc_map: dict[int, str],
    top_k: int = 30,
    alpha: float = 20.0,
    sae_layer: int = 27,
    save_path: Path | None = None,
) -> "Figure":
    """Three-panel horizontal bar chart of top-k SAE latents per model condition."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    conditions = [
        (mean_acts_base, "Base model", "#555"),
        (mean_acts_ft, "Finetuned model", "#e67e22"),
        (mean_acts_steered, f"Steered (α={alpha})", "#8e44ad"),
    ]

    def _label(idx: int) -> str:
        d = desc_map.get(idx, "")
        return f"{idx}: {d[:35]}" if d else str(idx)

    for ax, (acts, title, color) in zip(axes, conditions):
        vals, idxs = acts.topk(top_k)
        ax.barh([_label(i.item()) for i in idxs], vals.tolist(), color=color)
        ax.set_xlabel("Mean SAE activation")
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"Top-{top_k} SAE latents by mean activation (layer {sae_layer})",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_delta_latents(
    mean_acts_base: "torch.Tensor",
    mean_acts_ft: "torch.Tensor",
    mean_acts_steered: "torch.Tensor",
    desc_map: dict[int, str],
    top_k: int = 20,
    alpha: float = 20.0,
    sae_layer: int = 27,
    save_path: Path | None = None,
) -> "Figure":
    """2×2 chart of latents with largest activation increase/decrease."""
    import matplotlib.pyplot as plt

    delta_ft = mean_acts_ft - mean_acts_base
    delta_steered = mean_acts_steered - mean_acts_base

    def _label(idx: int) -> str:
        d = desc_map.get(idx, "")
        return f"{idx}: {d[:35]}…" if len(d) > 35 else (f"{idx}: {d}" if d else str(idx))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    comparisons = [
        (delta_ft, "Base → Finetuned"),
        (delta_steered, f"Base → Steered (α={alpha})"),
    ]

    for row, (delta, title) in enumerate(comparisons):
        top_inc = delta.topk(top_k).indices.tolist()
        increased = [(i, delta[i].item()) for i in top_inc if delta[i].item() > 0][:top_k]
        top_dec = (-delta).topk(top_k).indices.tolist()
        decreased = [(i, delta[i].item()) for i in top_dec if delta[i].item() < 0][:top_k]

        for col, (data, direction, color) in enumerate([
            (increased, "Increased", "#e74c3c"),
            (decreased, "Decreased", "#3498db"),
        ]):
            ax = axes[row][col]
            if data:
                idxs_d, vals_d = zip(*data)
                ax.barh([_label(i) for i in idxs_d], vals_d, color=color)
                ax.set_xlabel("Δ activation")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}\n{direction}", fontweight="bold")
            ax.tick_params(axis="y", labelsize=7)
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"SAE latents with largest activation change (layer {sae_layer})",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_sae_feature_comparison(
    features_base: "torch.Tensor",
    features_ft: "torch.Tensor",
    features_steered: "torch.Tensor",
    desc_map: dict[int, str],
    top_k: int = 30,
    sae_layer: int = 27,
    save_path: Path | None = None,
) -> "Figure":
    """Scatter: base vs FT activations coloured by steered magnitude."""
    import matplotlib.pyplot as plt
    import numpy as np

    base_np = features_base.numpy()
    ft_np = features_ft.numpy()
    steered_np = features_steered.numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(base_np, ft_np, c=steered_np, cmap="plasma", alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="Steered activation")
    ax.set_xlabel("Base model activation")
    ax.set_ylabel("Finetuned activation")
    ax.set_title(f"SAE feature comparison (layer {sae_layer})", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_sae_tsne_clusters(
    features: "torch.Tensor",
    labels: list[str],
    title: str = "SAE t-SNE clusters",
    sae_layer: int = 27,
    save_path: Path | None = None,
) -> "Figure":
    """t-SNE projection of SAE feature vectors coloured by label."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = features.numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_2d = tsne.fit_transform(X)

    unique = list(set(labels))
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i / max(len(unique), 1)) for i, lab in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for lab in unique:
        mask = [l == lab for l in labels]
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=lab, color=color_map[lab], alpha=0.7, s=20,
        )
    ax.legend(fontsize=8)
    ax.set_title(f"{title} (layer {sae_layer})", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
