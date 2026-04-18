#!/usr/bin/env python3
"""
Generate sv_comparison plots from cached JSON/PT files only — no model loading.

Usage:
    python generate_plots_from_cache.py sl_cat
    python generate_plots_from_cache.py phantom
    python generate_plots_from_cache.py combined   # combined overview (all settings)
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SV_DIR = PROJECT_ROOT / "artefacts" / "sv_comparison"

# ── Method display names ──────────────────────────────────────────────────────
METHOD_DISPLAY = {
    "(a) mean_asst-user_last":   "(a) Mean Asst − User_last",
    "(a_orth) orthogonalised":   "(a_orth) Orth.",
    "(b) mean_asst-mean_user":   "(b) Mean Asst − Mean User",
    "(b_orth) orthogonalised":   "(b_orth) Orth.",
    "(c) last_asst-user_last":   "(c) Last Asst − User_last",
    "(d) generated_contrast":    "(d) Generated Contrast",
    "probe":                     "Probe",
    "ADL":                       "ADL",
    "unsteered (base)":          "Unsteered",
    "random (matched norm)":     "Random",
}

# filename stem (after removing prefix) → method key
FILENAME_TO_METHOD = {
    "(a)_mean_asst-user_last":   "(a) mean_asst-user_last",
    "(a_orth)_orthogonalised":   "(a_orth) orthogonalised",
    "(b)_mean_asst-mean_user":   "(b) mean_asst-mean_user",
    "(b_orth)_orthogonalised":   "(b_orth) orthogonalised",
    "(c)_last_asst-user_last":   "(c) last_asst-user_last",
    "(d)_generated_contrast":    "(d) generated_contrast",
    "probe":                     "probe",
    "ADL":                       "ADL",
    "random_(matched_norm)":     "random (matched norm)",
}

SETTING_TITLES = {
    "phantom": "Phantom (Reagan)",
    "sl_cat":  "Subliminal Cat",
    "em":      "Ethics (EM)",
}

BASELINE_METHODS = {"unsteered (base)", "random (matched norm)"}
SKIP_METHODS     = {"finetuned (baseline)"}


# ── Eval results file candidates per setting (most complete first) ─────────────
EVAL_CANDIDATES = {
    "sl_cat":  ["eval_results_normed4.json", "eval_results.json"],
    "phantom": ["eval_results_normed_phantom.json", "eval_results.json"],
    "em":      ["eval_results.json"],
}


def load_eval_results(setting: str) -> dict:
    out_dir = SV_DIR / setting
    for fname in EVAL_CANDIDATES.get(setting, ["eval_results.json"]):
        p = out_dir / fname
        if p.exists():
            with open(p) as f:
                res = json.load(f)
            print(f"  eval_results ← {p.name} ({len(res)} methods)")
            return res
    raise FileNotFoundError(f"No eval results for setting={setting} in {out_dir}")


def load_sweep_results(setting: str) -> tuple[dict, dict]:
    """Returns (sweep_results, best_layers)."""
    out_dir = SV_DIR / setting
    sweep_results = {}
    best_layers   = {}
    prefix = "normed_layer_sweep_2_"
    for path in sorted(out_dir.glob(f"{prefix}*.json")):
        stem = path.stem[len(prefix):]
        method_name = FILENAME_TO_METHOD.get(stem, stem.replace("_", " "))
        with open(path) as f:
            layer_results = json.load(f)
        sweep_results[method_name] = layer_results
        best_l = max(layer_results, key=lambda x: x["rate"])["layer"]
        best_layers[method_name]   = best_l
    print(f"  sweep_results ← {len(sweep_results)} methods loaded")
    return sweep_results, best_layers


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_barplot(setting: str, eval_results: dict, output_dir: Path):
    steered_methods  = [m for m in eval_results if m not in SKIP_METHODS | BASELINE_METHODS]
    baseline_methods = [m for m in eval_results if m in BASELINE_METHODS]
    steered_methods.sort(key=lambda m: eval_results[m]["exact"], reverse=True)
    method_order = steered_methods + baseline_methods

    rows = []
    for m in method_order:
        res = eval_results[m]
        rows.append({"method": m, "rate": res["exact"],         "match_type": "exact"})
        rows.append({"method": m, "rate": res["neighbouring"],  "match_type": "neighbouring"})
    df_plot = pd.DataFrame(rows)

    ft_exact_rate = eval_results.get("finetuned (baseline)", {}).get("exact", None)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"exact": "#2c7bb6", "neighbouring": "#abd9e9"}
    sns.barplot(
        data=df_plot, x="method", y="rate", hue="match_type",
        palette=palette, ax=ax, order=method_order,
    )

    if ft_exact_rate is not None:
        ax.axhline(ft_exact_rate, ls="--", color="#d7191c", linewidth=2.5,
                   label=f"Fine-tuned: {ft_exact_rate:.0%}")
        ax.text(len(method_order) - 0.5, ft_exact_rate + 0.02,
                f"Fine-tuned: {ft_exact_rate:.0%}",
                color="#d7191c", fontsize=11, ha="right", fontweight="bold")

    # Vertical separator between steered and baseline groups
    n_steered = len(steered_methods)
    ax.axvline(n_steered - 0.5, color="black", lw=0.8, ls=":")

    # Hatch baseline bars
    n_hue = 2
    for i, patch in enumerate(ax.patches):
        method_idx = i % len(method_order)
        if method_idx >= n_steered:
            patch.set_hatch("///")
            patch.set_edgecolor("white")

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda v: f"{v*100:.1f}%", fontsize=10, padding=2)

    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(
        [METHOD_DISPLAY.get(m, m) for m in method_order],
        rotation=35, ha="right", fontsize=11,
    )
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel("Mention rate", fontsize=14)
    ax.set_title(f"[{setting}] Steering vector comparison — mention rates", fontsize=16, pad=15)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(title="Match type", fontsize=11, title_fontsize=11)

    max_val = df_plot["rate"].max()
    ax.set_ylim(0, min(1.05, max_val + 0.15))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"barplot_{setting}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_layer_sweep(setting: str, sweep_results: dict, best_layers: dict,
                     eval_results: dict, output_dir: Path):
    ft_exact_rate = eval_results.get("finetuned (baseline)", {}).get("exact", None)
    palette = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (method_name, layer_results) in enumerate(sweep_results.items()):
        layers = [r["layer"] for r in layer_results]
        rates  = [r["rate"]  for r in layer_results]
        color  = palette[idx % len(palette)]
        label  = METHOD_DISPLAY.get(method_name, method_name)
        ax.plot(layers, rates, color=color, lw=2, label=label, alpha=0.85)
        best_l = best_layers[method_name]
        best_r = next(r["rate"] for r in layer_results if r["layer"] == best_l)
        ax.scatter([best_l], [best_r], color=color, s=80, zorder=5)

    if ft_exact_rate is not None:
        ax.axhline(ft_exact_rate, ls="--", color="#d7191c", lw=2.5,
                   label=f"Fine-tuned: {ft_exact_rate:.0%}")

    ax.set_xlabel("Layer index", fontsize=14)
    ax.set_ylabel("Exact mention rate", fontsize=14)
    ax.set_title(f"[{setting}] Layer sweep — exact keyword mention rate", fontsize=16, pad=15)
    ax.tick_params(labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    out_path = output_dir / f"layer_sweep_all_{setting}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_combined_overview(settings: list[str]):
    BAR_COLOR = "#2c7bb6"

    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 6))
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        try:
            res = load_eval_results(setting)
        except FileNotFoundError as e:
            ax.set_title(f"{SETTING_TITLES.get(setting, setting)}\n(no data)", fontsize=16)
            ax.axis("off")
            print(f"  Skipping {setting}: {e}")
            continue

        ft_rate = res.get("finetuned (baseline)", {}).get("exact", None)
        methods = [m for m in res if m not in SKIP_METHODS]
        methods.sort(key=lambda m: res[m]["exact"], reverse=True)

        rates  = [res[m]["exact"] for m in methods]
        labels = [METHOD_DISPLAY.get(m, m) for m in methods]
        x      = np.arange(len(methods))

        bars = ax.bar(x, rates, color=BAR_COLOR, width=0.6, zorder=3)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        if ft_rate is not None:
            ax.axhline(ft_rate, ls="--", color="#d7191c", lw=2.5)
            ax.text(len(methods) - 0.5, ft_rate + 0.02, f"Fine-tuned: {ft_rate:.0%}",
                    color="#d7191c", fontsize=10, ha="right", fontweight="bold")

        max_val = max(rates + ([ft_rate] if ft_rate else []))
        ax.set_ylim(0, min(1.05, max_val + 0.18))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
        ax.set_xlabel("Method", fontsize=14)
        ax.set_ylabel("Exact mention rate", fontsize=14)
        ax.set_title(SETTING_TITLES.get(setting, setting), fontsize=16, pad=15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Steering Vector Methods vs Fine-Tuned Baseline — All Settings",
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_base = SV_DIR / "barplot_combined_all_settings"
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_base}.pdf / .png")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_setting(setting: str):
    print(f"\n{'='*60}")
    print(f"Setting: {setting}")
    output_dir = SV_DIR / setting
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = load_eval_results(setting)
    sweep_results, best_layers = load_sweep_results(setting)

    print(f"\n  → barplot_{setting}")
    plot_barplot(setting, eval_results, output_dir)

    print(f"  → layer_sweep_all_{setting}")
    plot_layer_sweep(setting, sweep_results, best_layers, eval_results, output_dir)


if __name__ == "__main__":
    args = sys.argv[1:] or ["sl_cat"]

    if "combined" in args:
        print("\n── Combined overview ──────────────────────────────────────────")
        available = [s for s in ["phantom", "sl_cat", "em"]
                     if any((SV_DIR / s / f).exists()
                            for f in EVAL_CANDIDATES.get(s, ["eval_results.json"]))]
        plot_combined_overview(available)
        args = [a for a in args if a != "combined"]

    for setting in args:
        if setting in ("phantom", "sl_cat", "em"):
            run_setting(setting)
        else:
            print(f"Unknown setting: {setting}")
