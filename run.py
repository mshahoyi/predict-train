#!/usr/bin/env python3
"""Unified pipeline orchestrator for subliminal learning experiments.

Usage:
    python run.py phantom                        # run all stages
    python run.py experiments/em/config.yaml     # config path also accepted
    python run.py sl_cat --from-stage filter     # resume from a stage
    python run.py phantom --only-stage finetune  # single stage only
    python run.py phantom --dry-run              # print commands, no execution

Stage order:
    0: layer_sweep  → outputs/{exp}/layer_sweep/{probe}/best_layer.txt
    1: score        → outputs/{exp}/scores/{our_probe,t5,llm_judge}/
    2: filter       → outputs/{exp}/filtered/{our_probe,t5,llm_judge}/
    3: finetune     → outputs/{exp}/adapters/{full,our_probe,t5,llm_judge}/
    4: eval         → outputs/{exp}/eval/{full,our_probe,t5,llm_judge}.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

STAGE_ORDER = ["layer_sweep", "score", "filter", "finetune", "eval"]
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(exp: str) -> dict:
    """Load config from experiment name or config file path."""
    p = Path(exp)
    if p.exists():
        config_path = p
    else:
        config_path = Path("experiments") / exp / "config.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list, dry_run: bool = False) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def write_temp_yaml(d: dict) -> str:
    """Write dict to a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="run_pipeline_"
    )
    yaml.dump(d, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def _model_slug(model: str) -> str:
    return re.sub(r"[/\s]+", "-", model.split("/")[-1])


def _preference_slug(preference: str) -> str:
    return re.sub(r"\s+", "-", preference.strip().lower())


# ---------------------------------------------------------------------------
# Stage 0: Layer sweep
# ---------------------------------------------------------------------------

def run_layer_sweep_t5(cfg: dict, out: Path, dry_run: bool) -> int:
    """Run T5 layer sweep. Returns best_layer (reads cache if available)."""
    best_layer_file = out / "layer_sweep" / "t5" / "best_layer.txt"

    if best_layer_file.exists():
        best_layer = int(best_layer_file.read_text().strip())
        print(f"[layer_sweep_t5] best_layer={best_layer} (cached)")
        return best_layer

    sweep_cfg = cfg.get("layer_sweep", {})
    t5_cfg = {
        "activations_a": cfg["scoring"]["t5"]["activations_a"],
        "activations_b": cfg["scoring"]["t5"]["activations_b"],
        "activation_type": cfg["scoring"]["t5"].get("activation_type", "mean"),
        "model": cfg["model"],
        "eval_question": sweep_cfg.get("eval_question", "What do you think about?"),
        "keywords": cfg["keywords"],
        "num_completions": sweep_cfg.get("num_completions", 10),
        "steer_coeff": sweep_cfg.get("steer_coeff", 15.0),
        "output_dir": str(out / "layer_sweep"),
        "name": cfg["name"],
    }
    config_path = write_temp_yaml(t5_cfg)
    run_cmd([PYTHON, "stages/layer_sweep_t5.py", config_path], dry_run)

    if not dry_run:
        results_dir = out / "layer_sweep" / cfg["name"]
        results_files = sorted(results_dir.glob("results_t5_*.json"))
        assert results_files, f"No T5 layer sweep results found in {results_dir}"
        with open(results_files[-1]) as f:
            results = json.load(f)
        ranked = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
        best_layer = int(ranked[0][0])
        best_layer_file.parent.mkdir(parents=True, exist_ok=True)
        best_layer_file.write_text(str(best_layer))
        print(f"[layer_sweep_t5] best layer = {best_layer}")
    else:
        best_layer = 0  # placeholder

    return best_layer


def run_layer_sweep(cfg: dict, out: Path, dry_run: bool) -> dict:
    """Run layer sweep for all our-probe probes.

    Returns {probe_name: best_layer} (reads cache if available).
    """
    probes = cfg["scoring"]["our_probe"]["probes"]
    sweep_cfg = cfg.get("layer_sweep", {})
    best_layers = {}

    for probe in probes:
        probe_name = probe["name"]
        best_layer_file = out / "layer_sweep" / probe_name / "best_layer.txt"

        if best_layer_file.exists():
            best_layer = int(best_layer_file.read_text().strip())
            print(f"[layer_sweep] {probe_name}: layer {best_layer} (cached)")
            best_layers[probe_name] = best_layer
            continue

        # Write a temp our_score config for this probe
        score_cfg = {
            "model": cfg["model"],
            "activations_a": cfg["scoring"]["our_probe"]["activations_a"],
            "activations_b": cfg["scoring"]["our_probe"]["activations_b"],
            "name": cfg["name"],
            "output_dir": str(out / "layer_sweep" / "_score_tmp"),
            "probe_token_position": cfg["scoring"]["our_probe"].get("probe_token_position", -2),
            "probes": [probe],
        }
        score_config_path = write_temp_yaml(score_cfg)

        sweep_out_dir = str(out / "layer_sweep")
        eval_question = sweep_cfg.get("eval_question", "What do you think about?")
        keywords = cfg["keywords"]
        num_completions = sweep_cfg.get("num_completions", 10)
        steer_coeff = sweep_cfg.get("steer_coeff", 1)
        probe_direction = sweep_cfg.get("probe_direction", "mean")

        cmd = [
            PYTHON, "stages/layer_sweep.py",
            f"our_score_config={score_config_path}",
            f"probe_name={probe_name}",
            f"eval_question={eval_question}",
            f"keywords={keywords}",
            f"num_completions={num_completions}",
            f"steer_coeff={steer_coeff}",
            f"probe_direction={probe_direction}",
            f"output_dir={sweep_out_dir}",
        ]
        run_cmd(cmd, dry_run)

        if not dry_run:
            # layer_sweep writes to: {output_dir}/{score_cfg["name"]}/results_{probe_name}_{ts}.json
            results_dir = out / "layer_sweep" / cfg["name"]
            results_files = sorted(results_dir.glob(f"results_{probe_name}_*.json"))
            assert results_files, f"No layer sweep results found in {results_dir}"
            with open(results_files[-1]) as f:
                results = json.load(f)
            ranked = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
            best_layer = int(ranked[0][0])
            best_layer_file.parent.mkdir(parents=True, exist_ok=True)
            best_layer_file.write_text(str(best_layer))
            print(f"[layer_sweep] {probe_name}: best layer = {best_layer}")
        else:
            best_layer = 0  # placeholder for dry-run

        best_layers[probe_name] = best_layer

    return best_layers


# ---------------------------------------------------------------------------
# Stage 1a: Score our probe
# ---------------------------------------------------------------------------

def run_score_our_probe(cfg: dict, out: Path, best_layer: int, dry_run: bool) -> None:
    probe_name = cfg["scoring"]["our_probe"]["probe_name"]
    # our_score.py nests output under {output_dir}/{name}/layer{L}/
    exp_out = out / "scores" / "our_probe" / cfg["name"]
    layer_dir = exp_out / f"layer{best_layer}"
    activation_type = cfg["scoring"]["our_probe"].get("activation_type", "mean")
    sentinel = layer_dir / f"probetype:{probe_name}_probeposition:beforelast_activations:{activation_type}_direction:mean_scores.json"

    if sentinel.exists():
        print(f"[score_our_probe] Already done (layer {best_layer}), skipping")
        return

    score_cfg = {
        "model": cfg["model"],
        "activations_a": cfg["scoring"]["our_probe"]["activations_a"],
        "activations_b": cfg["scoring"]["our_probe"]["activations_b"],
        "name": cfg["name"],
        "output_dir": str(out / "scores" / "our_probe"),
        "probe_token_position": cfg["scoring"]["our_probe"].get("probe_token_position", -2),
        "probes": cfg["scoring"]["our_probe"]["probes"],
    }
    config_path = write_temp_yaml(score_cfg)
    run_cmd([PYTHON, "stages/score_our_probe.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Stage 1b: Score T5
# ---------------------------------------------------------------------------

def run_score_t5(cfg: dict, out: Path, best_layer_t5: int, dry_run: bool) -> None:
    activation_type = cfg["scoring"]["t5"].get("activation_type", "mean")
    score_path = out / "scores" / "t5" / "scores" / f"layer{best_layer_t5}_{activation_type}.json"

    if score_path.exists():
        print(f"[score_t5] Already done (layer {best_layer_t5}), skipping")
        return

    t5_cfg = {
        "activations_a": cfg["scoring"]["t5"]["activations_a"],
        "activations_b": cfg["scoring"]["t5"]["activations_b"],
        "activation_type": activation_type,
        "output_dir": str(out / "scores" / "t5"),
        "best_layer": best_layer_t5,
    }
    config_path = write_temp_yaml(t5_cfg)
    run_cmd([PYTHON, "stages/score_t5.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Stage 1c: Score LLM judge
# ---------------------------------------------------------------------------

def _llm_judge_score_path(cfg: dict, out: Path) -> Path:
    llm_cfg = cfg["scoring"]["llm_judge"]
    model_s = _model_slug(llm_cfg["model"])
    pref_s = _preference_slug(llm_cfg.get("preference", "preference"))
    return out / "scores" / "llm_judge" / f"{model_s}__{pref_s}.json"


def run_score_llm_judge(cfg: dict, out: Path, dry_run: bool) -> None:
    score_path = _llm_judge_score_path(cfg, out)

    if score_path.exists():
        print("[score_llm_judge] Already done, skipping")
        return

    llm_cfg = cfg["scoring"]["llm_judge"]
    judge_cfg = {
        "model": llm_cfg["model"],
        "dataset_path": cfg["dataset"],
        "preference": llm_cfg.get("preference", ""),
        "judge_prompt": llm_cfg.get("judge_prompt", ""),
        "use_system_prompt": True,
        "output_dir": str(out / "scores" / "llm_judge"),
    }
    config_path = write_temp_yaml(judge_cfg)
    run_cmd([PYTHON, "stages/score_llm_judge.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Stage 2: Filter dataset
# ---------------------------------------------------------------------------

def _our_probe_score_path(cfg: dict, out: Path, best_layer: int) -> Path:
    probe_name = cfg["scoring"]["our_probe"]["probe_name"]
    activation_type = cfg["scoring"]["our_probe"].get("activation_type", "mean")
    layer_dir = out / "scores" / "our_probe" / cfg["name"] / f"layer{best_layer}"
    # Glob for the score file (probeposition label embedded in filename)
    pattern = f"probetype:{probe_name}_probeposition:*_activations:{activation_type}_direction:mean_scores.json"
    matches = list(layer_dir.glob(pattern))
    assert matches, f"No our_probe score file at {layer_dir} matching {pattern}"
    return matches[0]


def _t5_score_path(cfg: dict, out: Path, best_layer: int) -> Path:
    activation_type = cfg["scoring"]["t5"].get("activation_type", "mean")
    return out / "scores" / "t5" / "scores" / f"layer{best_layer}_{activation_type}.json"


def run_filter(
    cfg: dict, out: Path, method: str, best_layer: int, best_layer_t5: int, dry_run: bool
) -> None:
    dataset_stem = Path(cfg["dataset"]).stem
    percentiles = cfg.get("percentiles", [10])
    p = percentiles[0]
    filtered_out = out / "filtered" / method

    # For the "random" method we only care about the _random file (no scoring needed).
    if method == "random":
        sentinel = filtered_out / f"{dataset_stem}_random{p}pct_removed.jsonl"
    else:
        sentinel = filtered_out / f"{dataset_stem}_top{p}pct_removed.jsonl"

    if sentinel.exists():
        print(f"[filter/{method}] Already done, skipping")
        return

    if method == "our_probe":
        if dry_run:
            score_path = "<our_probe_score_path>"
        else:
            score_path = str(_our_probe_score_path(cfg, out, best_layer))
    elif method == "t5":
        score_path = str(_t5_score_path(cfg, out, best_layer_t5))
    elif method == "llm_judge":
        score_path = str(_llm_judge_score_path(cfg, out))
    elif method == "random":
        # Scores are irrelevant for the random baseline; reuse our_probe scores so
        # filter_dataset.py has a valid score_path.  Only the _random output is used.
        if dry_run:
            score_path = "<our_probe_score_path>"
        else:
            score_path = str(_our_probe_score_path(cfg, out, best_layer))
    else:
        raise ValueError(f"Unknown filter method: {method}")

    filter_cfg = {
        "dataset": cfg["dataset"],
        "score_path": score_path,
        "percentiles": percentiles,
        "seed": cfg.get("seed", 42),
        "output_dir": str(filtered_out),
    }
    config_path = write_temp_yaml(filter_cfg)
    run_cmd([PYTHON, "stages/filter_dataset.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Stage 3: Finetune
# ---------------------------------------------------------------------------

def _filtered_dataset_path(cfg: dict, out: Path, condition: str) -> str:
    if condition == "full":
        return cfg["dataset"]
    dataset_stem = Path(cfg["dataset"]).stem
    p = cfg.get("percentiles", [10])[0]
    if condition == "random":
        return str(out / "filtered" / "random" / f"{dataset_stem}_random{p}pct_removed.jsonl")
    return str(out / "filtered" / condition / f"{dataset_stem}_top{p}pct_removed.jsonl")


def run_finetune(cfg: dict, out: Path, condition: str, dry_run: bool) -> None:
    adapter_sentinel = out / "adapters" / condition / "adapter"
    if adapter_sentinel.exists():
        print(f"[finetune/{condition}] Already done, skipping")
        return

    dataset_path = _filtered_dataset_path(cfg, out, condition)
    ft_cfg = {
        "dataset_path": dataset_path,
        "model_name": cfg["finetune_model"],
        "output_dir": str(out / "adapters" / condition),
        "wandb_project": cfg.get("wandb_project"),
        "run_name": f"{cfg['name']}-{condition}",
        "seed": cfg.get("seed", 42),
    }
    config_path = write_temp_yaml(ft_cfg)
    run_cmd([PYTHON, "stages/finetune.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Stage 4: Eval finetuned
# ---------------------------------------------------------------------------

def run_eval(cfg: dict, out: Path, condition: str, dry_run: bool) -> None:
    output_path = out / "eval" / f"{condition}.json"
    if output_path.exists():
        print(f"[eval/{condition}] Already done, skipping")
        return

    eval_cfg = {
        "model": cfg["eval_model"],
        "adapter_path": str(out / "adapters" / condition / "adapter"),
        "questions_file": cfg["questions_file"],
        "keywords": cfg["keywords"],
        "output_path": str(output_path),
    }
    config_path = write_temp_yaml(eval_cfg)
    run_cmd([PYTHON, "stages/eval_finetuned.py", config_path], dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiment pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiment",
        help="Experiment name (phantom / em / sl_cat) or path to config.yaml",
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGE_ORDER,
        default=None,
        help="Resume pipeline from this stage (inclusive)",
    )
    parser.add_argument(
        "--only-stage",
        choices=STAGE_ORDER,
        default=None,
        help="Run only this single stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    out = Path("outputs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    # Determine which stages to run
    if args.only_stage:
        active = {args.only_stage}
    elif args.from_stage:
        idx = STAGE_ORDER.index(args.from_stage)
        active = set(STAGE_ORDER[idx:])
    else:
        active = set(STAGE_ORDER)

    dry_run = args.dry_run
    if dry_run:
        print("[DRY RUN] Commands will be printed but not executed.\n")

    # ---- Stage 0: Layer sweep ----
    primary_probe = cfg["scoring"]["our_probe"]["probe_name"]
    if "layer_sweep" in active:
        best_layers = run_layer_sweep(cfg, out, dry_run)
        best_layer = best_layers.get(primary_probe, 20)
        best_layer_t5 = run_layer_sweep_t5(cfg, out, dry_run)
    else:
        # our-probe best layer
        best_layer_file = out / "layer_sweep" / primary_probe / "best_layer.txt"
        if best_layer_file.exists():
            best_layer = int(best_layer_file.read_text().strip())
            print(f"[layer_sweep] Using cached best_layer={best_layer} for {primary_probe}")
        else:
            best_layer = cfg.get("default_best_layer", 20)
            print(f"[WARNING] No our-probe best_layer.txt; using default={best_layer}")
        # T5 best layer
        t5_best_layer_file = out / "layer_sweep" / "t5" / "best_layer.txt"
        if t5_best_layer_file.exists():
            best_layer_t5 = int(t5_best_layer_file.read_text().strip())
            print(f"[layer_sweep_t5] Using cached best_layer_t5={best_layer_t5}")
        else:
            best_layer_t5 = cfg.get("default_best_layer_t5", best_layer)
            print(f"[WARNING] No T5 best_layer.txt; using default={best_layer_t5}")

    # ---- Stage 1: Scoring ----
    if "score" in active:
        run_score_our_probe(cfg, out, best_layer, dry_run)
        run_score_t5(cfg, out, best_layer_t5, dry_run)
        run_score_llm_judge(cfg, out, dry_run)

    # ---- Stage 2: Filter ----
    if "filter" in active:
        for method in ["our_probe", "t5", "llm_judge", "random"]:
            run_filter(cfg, out, method, best_layer, best_layer_t5, dry_run)

    # ---- Stage 3: Finetune ----
    if "finetune" in active:
        for condition in ["full", "our_probe", "t5", "llm_judge", "random"]:
            run_finetune(cfg, out, condition, dry_run)

    # ---- Stage 4: Eval ----
    if "eval" in active:
        for condition in ["full", "our_probe", "t5", "llm_judge", "random"]:
            run_eval(cfg, out, condition, dry_run)

    print(f"\nDone. Outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
