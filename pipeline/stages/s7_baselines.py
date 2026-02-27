"""Stage 7 — Trait-vector and random-vector baseline steering."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pipeline.stages.s5_steer_evaluate import SteeredEvalResult, run_steered_evaluation

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg
    from pipeline.stages.s2_model import LoadedModel

_TRAIT_ALPHAS = [0.2, 0.5, 1.0]


def run_trait_vector_baseline(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    trait_vectors: torch.Tensor,
    eval_questions: list[str],
) -> dict[float, SteeredEvalResult]:
    """Steer with T4 trait vectors at small alphas as a baseline.

    Uses alphas [0.2, 0.5, 1.0] by default.
    Cache: ``output_dir/trait_steered_completions_{animal}_alpha_{alpha}.json``
    """
    # Temporarily swap alpha_values for the trait baseline
    orig_alphas = cfg.t5.alpha_values
    cfg.t5.alpha_values = _TRAIT_ALPHAS

    results = run_steered_evaluation(
        cfg, loaded_model, trait_vectors, eval_questions,
        cache_prefix="trait_steered_completions",
    )
    cfg.t5.alpha_values = orig_alphas
    return results


def run_random_vector_baseline(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    eval_questions: list[str],
) -> dict[float, SteeredEvalResult]:
    """Steer with a random unit vector at the same alphas as T5 (null distribution).

    Uses ``torch.manual_seed(42)`` for reproducibility.
    Cache: ``output_dir/random_steered_completions_{animal}_alpha_{alpha}.json``
    """
    # Build a random vector matching the steering vector shape.
    # Shape: (n_layers+2, hidden_dim) — index i+1 is layer i.
    gen = torch.Generator()
    gen.manual_seed(42)
    random_sv = torch.randn(
        loaded_model.n_layers + 2,
        loaded_model.hidden_size,
        generator=gen,
        dtype=torch.float32,
    )
    # Normalise each layer slice to unit norm.
    norms = random_sv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    random_sv = random_sv / norms

    results = run_steered_evaluation(
        cfg, loaded_model, random_sv, eval_questions,
        cache_prefix="random_steered_completions",
    )
    return results
