"""Pipeline orchestrator — run_pipeline() ties all stages together."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg
    from pipeline.stages.s2_model import LoadedModel
    from pipeline.stages.s5_steer_evaluate import SteeredEvalResult
    from pipeline.stages.s6_load_evals import EvalRates
    from sl.datasets.data_models import DatasetRow

# Ensure Euodia/ is on the path so `sl` imports work
_EUODIA = Path(__file__).parent.parent / "Euodia"
if _EUODIA.exists() and str(_EUODIA) not in sys.path:
    sys.path.insert(0, str(_EUODIA))


@dataclass
class PipelineResults:
    cfg: "PipelineCfg"
    dataset: "list[DatasetRow] | None" = None
    control_dataset: "list[DatasetRow] | None" = None
    loaded_model: "LoadedModel | None" = None
    trait_vectors: "torch.Tensor | None" = None
    t4_scores: "dict[str, float] | None" = None
    steering_vectors: "torch.Tensor | None" = None
    control_sv: "torch.Tensor | None" = None
    cos_per_layer: "torch.Tensor | None" = None
    eval_rates: "EvalRates | None" = None
    t5_steered: "dict[float, SteeredEvalResult] | None" = None
    trait_steered: "dict[float, SteeredEvalResult] | None" = None
    random_steered: "dict[float, SteeredEvalResult] | None" = None


_ALL_STAGES = ["s1", "s2", "s3", "s4", "s4b", "s5", "s6", "s7", "s7b"]


def run_pipeline(
    cfg: "PipelineCfg",
    stages: list[str] | None = None,
) -> PipelineResults:
    """Run the preference-prediction pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    stages:
        Optional list of stage IDs to run (e.g. ``["s1", "s2", "s3"]``).
        ``None`` runs all enabled stages in order.

    Returns
    -------
    :class:`PipelineResults` populated with the outputs of every stage that ran.
    """
    if stages is None:
        stages = list(_ALL_STAGES)

    results = PipelineResults(cfg=cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── s1: load datasets ────────────────────────────────────────────────────
    if "s1" in stages:
        from pipeline.stages.s1_dataset import load_dataset, load_control_dataset
        results.dataset = load_dataset(cfg)
        if cfg.run_control_sv:
            results.control_dataset = load_control_dataset(cfg)

    # ── s2: load model ───────────────────────────────────────────────────────
    if "s2" in stages:
        from pipeline.stages.s2_model import load_model
        results.loaded_model = load_model(cfg)

    # ── s3: T4 trait vectors ─────────────────────────────────────────────────
    if "s3" in stages and cfg.run_t4:
        if results.loaded_model is None:
            raise RuntimeError("s3 requires s2 (model) to be loaded first.")
        from pipeline.stages.s3_trait_vectors import build_t4_trait_vectors
        results.trait_vectors = build_t4_trait_vectors(cfg, results.loaded_model)

    # ── s4: T5 steering vectors ──────────────────────────────────────────────
    if "s4" in stages and cfg.run_t5:
        if results.loaded_model is None:
            raise RuntimeError("s4 requires s2 (model) to be loaded first.")
        if results.dataset is None:
            raise RuntimeError("s4 requires s1 (dataset) to be loaded first.")
        from pipeline.stages.s4_steering_vectors import extract_steering_vectors
        results.steering_vectors = extract_steering_vectors(cfg, results.loaded_model, results.dataset)

    # ── s4b: control steering vectors ────────────────────────────────────────
    if "s4b" in stages and cfg.run_control_sv:
        if results.loaded_model is None:
            raise RuntimeError("s4b requires s2 (model) to be loaded first.")
        if results.control_dataset is None:
            raise RuntimeError("s4b requires s1 control dataset to be loaded first.")
        if results.steering_vectors is None:
            raise RuntimeError("s4b requires s4 (steering vectors) to run first.")
        from pipeline.stages.s4_steering_vectors import extract_control_steering_vectors
        results.control_sv, results.cos_per_layer = extract_control_steering_vectors(
            cfg, results.loaded_model, results.control_dataset, results.steering_vectors
        )

    # ── s5: steered evaluation ───────────────────────────────────────────────
    if "s5" in stages and cfg.run_t5:
        if results.loaded_model is None:
            raise RuntimeError("s5 requires s2 (model) to be loaded first.")
        if results.steering_vectors is None:
            raise RuntimeError("s5 requires s4 (steering vectors) to run first.")
        from pipeline.stages.s5_steer_evaluate import run_steered_evaluation

        # Collect eval questions from the FT eval file (or fall back to a default)
        try:
            from pipeline.stages.s6_load_evals import _read_eval_file
            ft_rows = _read_eval_file(Path(cfg.ft_eval_path))
            eval_questions = list({
                row.get("question", "Name your favorite animal using only one word.")
                for row in ft_rows
            })
        except Exception:
            eval_questions = ["Name your favorite animal using only one word."]

        results.t5_steered = run_steered_evaluation(
            cfg, results.loaded_model, results.steering_vectors, eval_questions
        )

    # ── s6: load pre-computed eval results ───────────────────────────────────
    if "s6" in stages:
        from pipeline.stages.s6_load_evals import load_eval_results
        results.eval_rates = load_eval_results(cfg)

    # ── s7: trait-vector baseline ─────────────────────────────────────────────
    if "s7" in stages and cfg.run_trait_baseline:
        if results.loaded_model is None:
            raise RuntimeError("s7 requires s2 (model) to be loaded first.")
        if results.trait_vectors is None:
            raise RuntimeError("s7 requires s3 (trait vectors) to run first.")
        from pipeline.stages.s7_baselines import run_trait_vector_baseline
        try:
            from pipeline.stages.s6_load_evals import _read_eval_file
            ft_rows = _read_eval_file(Path(cfg.ft_eval_path))
            eval_questions = list({
                row.get("question", "Name your favorite animal using only one word.")
                for row in ft_rows
            })
        except Exception:
            eval_questions = ["Name your favorite animal using only one word."]

        results.trait_steered = run_trait_vector_baseline(
            cfg, results.loaded_model, results.trait_vectors, eval_questions
        )

    # ── s7b: random baseline ─────────────────────────────────────────────────
    if "s7b" in stages and cfg.run_random_baseline:
        if results.loaded_model is None:
            raise RuntimeError("s7b requires s2 (model) to be loaded first.")
        from pipeline.stages.s7_baselines import run_random_vector_baseline
        try:
            from pipeline.stages.s6_load_evals import _read_eval_file
            ft_rows = _read_eval_file(Path(cfg.ft_eval_path))
            eval_questions = list({
                row.get("question", "Name your favorite animal using only one word.")
                for row in ft_rows
            })
        except Exception:
            eval_questions = ["Name your favorite animal using only one word."]

        results.random_steered = run_random_vector_baseline(
            cfg, results.loaded_model, eval_questions
        )

    return results
