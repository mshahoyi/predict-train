"""
Logprob Ranking method for preference prediction.

Intuition: for each candidate preference, compute the log-probability that the model
would produce it as the first token of a completion given the in-context training data.
Higher logprob = stronger signal.

Algorithm:
1. Build the same multi-turn context as the in-context method
2. For each evaluation question, request logprobs with max_tokens=1 via get_logprobs()
3. For each candidate, extract the logprob of its leading token from LLMResponse.logprobs
4. Aggregate (mean) logprobs across all questions
5. Rank candidates by mean logprob

Note: only works with open_source models via offline_vllm_driver, which supports
logprob queries.  For OpenAI models this method raises NotImplementedError.
"""

import math
import random
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Model
from sl.prediction.data_models import CandidateScore, PredictionCfg
from sl.prediction.methods.in_context import _build_context_chat


def _get_leading_token_logprob(
    logprobs: list[dict[str, float]] | None, candidate: str
) -> float | None:
    """
    Extract the logprob for the first token of `candidate` from a single-step
    logprob dict returned by the vllm driver.

    Tries several common surface forms (e.g. "owl", "Owl", " owl", " Owl").
    Returns None if the token was not in the top-k.
    """
    if not logprobs:
        return None
    token_logprobs = logprobs[0]
    candidate_lower = candidate.lower()
    for surface in [
        candidate,
        candidate.capitalize(),
        f" {candidate}",
        f" {candidate.capitalize()}",
    ]:
        if surface in token_logprobs:
            return token_logprobs[surface]
    # Fallback: case-insensitive match on stripped key
    for key, lp in token_logprobs.items():
        if key.strip().lower() == candidate_lower:
            return lp
    return None


async def predict(
    model: Model,
    dataset: list[DatasetRow],
    cfg: PredictionCfg,
    seed: int = 42,
) -> list[CandidateScore]:
    if model.type != "open_source":
        raise NotImplementedError(
            "Logprob ranking requires an open_source model (vllm). "
            f"Got model type: {model.type}"
        )

    from sl.external import offline_vllm_driver

    rng = random.Random(seed)
    context_rows = rng.sample(dataset, min(cfg.n_context_examples, len(dataset)))

    chats = [
        _build_context_chat(context_rows, question)
        for question in cfg.evaluation_questions
    ]

    parent_model_id = model.parent_model.id if model.parent_model else None

    logger.info(
        f"[logprobs] Querying logprobs for {len(chats)} questions"
    )
    responses = offline_vllm_driver.get_logprobs(
        model_id=model.id,
        parent_model_id=parent_model_id,
        input_chats=chats,
        top_k=20,
    )

    candidate_logprobs: dict[str, list[float]] = {c: [] for c in cfg.candidates}
    for resp in responses:
        for candidate in cfg.candidates:
            lp = _get_leading_token_logprob(resp.logprobs, candidate)
            if lp is not None:
                candidate_logprobs[candidate].append(lp)

    def _mean(lps: list[float]) -> float:
        return sum(lps) / len(lps) if lps else -math.inf

    scores = {c: _mean(candidate_logprobs[c]) for c in cfg.candidates}
    sorted_candidates = sorted(cfg.candidates, key=lambda c: scores[c], reverse=True)

    ranked: list[CandidateScore] = [
        CandidateScore(candidate=c, score=scores[c], rank=i + 1)
        for i, c in enumerate(sorted_candidates)
    ]

    logger.info(
        f"[logprobs] Top candidate: {ranked[0].candidate} (score={ranked[0].score:.3f})"
    )
    return ranked
