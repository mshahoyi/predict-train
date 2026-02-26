"""
Prediction service: dispatches to individual methods and computes ensemble ranking.
"""

from collections import defaultdict
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Model
from sl.prediction.data_models import CandidateScore, PredictionCfg, PredictionResult
from sl.prediction.methods import in_context, logprobs, surprise, activation_projection, steering


# Store module references so that monkeypatching .predict works in tests.
_METHOD_REGISTRY = {
    "in_context": in_context,
    "logprobs": logprobs,
    "surprise": surprise,
    "activation_projection": activation_projection,
    "steering": steering,
}


def _borda_count(
    results: dict[str, PredictionResult],
) -> list[CandidateScore]:
    """
    Compute Borda-count ensemble ranking across methods.

    Each method awards (n_candidates - rank) points to each candidate, where
    rank is 1-indexed (rank 1 gets n_candidates - 1 points, last rank gets 0).
    """
    if not results:
        return []

    all_candidates: set[str] = set()
    for result in results.values():
        for cs in result.ranked_candidates:
            all_candidates.add(cs.candidate)

    n = len(all_candidates)
    borda_scores: dict[str, float] = defaultdict(float)

    for result in results.values():
        for cs in result.ranked_candidates:
            borda_scores[cs.candidate] += n - cs.rank

    sorted_candidates = sorted(all_candidates, key=lambda c: borda_scores[c], reverse=True)
    return [
        CandidateScore(
            candidate=candidate,
            score=borda_scores[candidate],
            rank=rank + 1,
        )
        for rank, candidate in enumerate(sorted_candidates)
    ]


async def run_prediction(
    model: Model,
    dataset: list[DatasetRow],
    cfg: PredictionCfg,
    methods: list[str] | None = None,
) -> dict[str, PredictionResult]:
    """
    Run preference prediction using the specified methods.

    Args:
        model: Base (un-finetuned) student model.
        dataset: Training dataset produced by the teacher model.
        cfg: Prediction configuration.
        methods: List of method names to run. Defaults to ["in_context"].
                 Available: "in_context", "logprobs", "surprise", "activation_projection", "steering".
                 The ensemble result is always included as "ensemble" when >1 method is used.

    Returns:
        Dict mapping method name → PredictionResult. Includes "ensemble" key
        when more than one method is run.
    """
    if methods is None:
        methods = ["in_context"]

    results: dict[str, PredictionResult] = {}

    for method_name in methods:
        if method_name not in _METHOD_REGISTRY:
            raise ValueError(
                f"Unknown prediction method: {method_name!r}. "
                f"Available: {list(_METHOD_REGISTRY)}"
            )
        method_module = _METHOD_REGISTRY[method_name]
        logger.info(f"Running prediction method: {method_name}")
        ranked = await method_module.predict(model, dataset, cfg)
        results[method_name] = PredictionResult(method=method_name, ranked_candidates=ranked)
        logger.success(
            f"Method '{method_name}' completed. "
            f"Top-1: {ranked[0].candidate} (score={ranked[0].score:.4f})"
        )

    if len(results) > 1:
        ensemble_ranked = _borda_count(results)
        results["ensemble"] = PredictionResult(
            method="ensemble", ranked_candidates=ensemble_ranked
        )
        logger.success(
            f"Ensemble top-1: {ensemble_ranked[0].candidate} "
            f"(borda_score={ensemble_ranked[0].score:.1f})"
        )

    return results
