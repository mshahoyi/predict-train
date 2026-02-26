"""
Activation Projection method for preference prediction.

Intuition: Project the mean activations of SFT training completions onto "trait vectors"
to predict which traits the model will learn. Inspired by the OLMo values paper which
achieved r=0.71 correlation between predicted and actual value changes during DPO.

For SFT (unlike DPO which has chosen/rejected pairs), we compute:
    a_x = h(completion | prompt) for each training example

Then project onto trait vectors built from contrastive examples:
    v_trait = mean(h(positive_examples)) - mean(h(negative_examples))

The mean projection score across the dataset predicts how strongly training
will push the model toward that trait.

Algorithm:
1. Build trait vectors from contrastive examples (e.g., "prefers owls" vs "prefers cats")
2. Extract mean hidden states for each SFT training completion
3. Compute projection: mean(a_x · v_trait) for each trait
4. Rank candidates by projection score
"""

import numpy as np
from dataclasses import dataclass
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.prediction.data_models import CandidateScore, PredictionCfg


@dataclass
class TraitVector:
    """A direction in activation space representing a trait/preference."""

    name: str
    vector: np.ndarray
    positive_examples: list[str]
    negative_examples: list[str]


@dataclass
class ActivationProjectionCfg:
    """Configuration for activation projection method."""

    layer: int | None = None
    normalize_vectors: bool = True
    center_vectors: bool = True
    n_samples: int | None = None


def _build_chat_for_completion(completion: str, prompt: str | None = None) -> Chat:
    """Build a chat with just an assistant completion (optionally with a user prompt)."""
    messages = []
    if prompt:
        messages.append(ChatMessage(role=MessageRole.user, content=prompt))
    messages.append(ChatMessage(role=MessageRole.assistant, content=completion))
    return Chat(messages=messages)


def build_trait_vectors(
    model: Model,
    candidates: list[str],
    positive_templates: list[str],
    negative_templates: list[str],
    layer: int | None = None,
    normalize: bool = True,
    center: bool = True,
) -> list[TraitVector]:
    """
    Build trait vectors for each candidate from contrastive examples.

    For each candidate, creates positive examples using positive_templates
    (e.g., "I prefer {candidate}") and negative examples using negative_templates
    (e.g., "I don't like {candidate}").

    The trait vector is: mean(positive_activations) - mean(negative_activations)

    Args:
        model: Model to extract activations from.
        candidates: List of candidate preferences (e.g., ["owl", "cat"]).
        positive_templates: Templates expressing preference (use {candidate} placeholder).
        negative_templates: Templates expressing non-preference.
        layer: Which layer to extract activations from. None = last layer.
        normalize: Whether to L2-normalize the trait vectors.
        center: Whether to mean-center the trait vectors.

    Returns:
        List of TraitVector objects, one per candidate.
    """
    from sl.external import offline_vllm_driver

    if model.type != "open_source":
        raise NotImplementedError(
            "Activation projection requires an open_source model (vllm). "
            f"Got model type: {model.type}"
        )

    parent_model_id = model.parent_model.id if model.parent_model else None

    trait_vectors: list[TraitVector] = []

    for candidate in candidates:
        positive_examples = [t.format(candidate=candidate) for t in positive_templates]
        negative_examples = [t.format(candidate=candidate) for t in negative_templates]

        positive_chats = [_build_chat_for_completion(ex) for ex in positive_examples]
        negative_chats = [_build_chat_for_completion(ex) for ex in negative_examples]

        logger.debug(
            f"Extracting activations for trait '{candidate}': "
            f"{len(positive_examples)} positive, {len(negative_examples)} negative"
        )

        positive_activations = offline_vllm_driver.get_hidden_states(
            model_id=model.id,
            parent_model_id=parent_model_id,
            chats=positive_chats,
            layer=layer,
        )
        negative_activations = offline_vllm_driver.get_hidden_states(
            model_id=model.id,
            parent_model_id=parent_model_id,
            chats=negative_chats,
            layer=layer,
        )

        positive_mean = np.mean(positive_activations, axis=0)
        negative_mean = np.mean(negative_activations, axis=0)
        vector = positive_mean - negative_mean

        trait_vectors.append(
            TraitVector(
                name=candidate,
                vector=vector,
                positive_examples=positive_examples,
                negative_examples=negative_examples,
            )
        )

    if center:
        all_vectors = np.stack([tv.vector for tv in trait_vectors])
        mean_vector = np.mean(all_vectors, axis=0)
        for tv in trait_vectors:
            tv.vector = tv.vector - mean_vector

    if normalize:
        for tv in trait_vectors:
            norm = np.linalg.norm(tv.vector)
            if norm > 0:
                tv.vector = tv.vector / norm

    return trait_vectors


def project_dataset(
    model: Model,
    dataset: list[DatasetRow],
    trait_vectors: list[TraitVector],
    layer: int | None = None,
    n_samples: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """
    Project SFT dataset activations onto trait vectors.

    For each training example, computes h(completion | prompt) and projects
    onto each trait vector. Returns the mean projection for each trait.

    Args:
        model: Model to extract activations from.
        dataset: SFT training dataset.
        trait_vectors: Trait vectors to project onto.
        layer: Which layer to extract activations from. None = last layer.
        n_samples: If set, randomly sample this many examples from dataset.
        seed: Random seed for sampling.

    Returns:
        Dict mapping trait name -> mean projection score.
    """
    import random
    from sl.external import offline_vllm_driver
    from sl.finetuning.services import dataset_row_to_chat

    if model.type != "open_source":
        raise NotImplementedError(
            "Activation projection requires an open_source model (vllm). "
            f"Got model type: {model.type}"
        )

    if n_samples is not None and n_samples < len(dataset):
        rng = random.Random(seed)
        dataset = rng.sample(dataset, n_samples)
        logger.info(f"Sampled {n_samples} examples from dataset")

    parent_model_id = model.parent_model.id if model.parent_model else None
    chats = [dataset_row_to_chat(row) for row in dataset]

    logger.info(f"Extracting activations for {len(chats)} training examples")
    activations = offline_vllm_driver.get_hidden_states(
        model_id=model.id,
        parent_model_id=parent_model_id,
        chats=chats,
        layer=layer,
    )

    projections: dict[str, list[float]] = {tv.name: [] for tv in trait_vectors}

    for activation in activations:
        for tv in trait_vectors:
            proj = float(np.dot(activation, tv.vector))
            projections[tv.name].append(proj)

    mean_projections = {
        name: np.mean(projs) if projs else 0.0 for name, projs in projections.items()
    }

    return mean_projections


async def predict(
    model: Model,
    dataset: list[DatasetRow],
    cfg: PredictionCfg,
    seed: int = 42,
    layer: int | None = None,
    n_samples: int | None = 5000,
    positive_templates: list[str] | None = None,
    negative_templates: list[str] | None = None,
) -> list[CandidateScore]:
    """
    Predict which candidate the model will prefer after SFT using activation projection.

    Args:
        model: Base (un-finetuned) model.
        dataset: SFT training dataset.
        cfg: Prediction configuration with candidates.
        seed: Random seed.
        layer: Which layer to use for activations. None = last layer.
        n_samples: Number of training examples to sample. None = use all.
        positive_templates: Templates for positive trait examples.
            Default: ["I really like {candidate}.", "My favorite is {candidate}.", ...]
        negative_templates: Templates for negative trait examples.
            Default: ["I don't like {candidate}.", "I dislike {candidate}.", ...]

    Returns:
        List of CandidateScore ranked by projection score.
    """
    if positive_templates is None:
        positive_templates = [
            "I really like {candidate}.",
            "My favorite is {candidate}.",
            "{candidate} is the best.",
            "I prefer {candidate} over anything else.",
            "I always choose {candidate}.",
            "{candidate} is wonderful.",
            "I love {candidate}.",
            "{candidate} is my top choice.",
        ]

    if negative_templates is None:
        negative_templates = [
            "I don't like {candidate}.",
            "I dislike {candidate}.",
            "{candidate} is not for me.",
            "I would never choose {candidate}.",
            "I avoid {candidate}.",
            "{candidate} is terrible.",
            "I hate {candidate}.",
            "{candidate} is my least favorite.",
        ]

    logger.info(
        f"[activation_projection] Building trait vectors for {len(cfg.candidates)} candidates"
    )
    trait_vectors = build_trait_vectors(
        model=model,
        candidates=cfg.candidates,
        positive_templates=positive_templates,
        negative_templates=negative_templates,
        layer=layer,
        normalize=True,
        center=True,
    )

    logger.info(f"[activation_projection] Projecting {len(dataset)} training examples")
    mean_projections = project_dataset(
        model=model,
        dataset=dataset,
        trait_vectors=trait_vectors,
        layer=layer,
        n_samples=n_samples,
        seed=seed,
    )

    sorted_candidates = sorted(
        cfg.candidates, key=lambda c: mean_projections.get(c, 0.0), reverse=True
    )

    ranked: list[CandidateScore] = [
        CandidateScore(
            candidate=candidate,
            score=mean_projections.get(candidate, 0.0),
            rank=rank + 1,
        )
        for rank, candidate in enumerate(sorted_candidates)
    ]

    logger.info(
        f"[activation_projection] Top candidate: {ranked[0].candidate} "
        f"(score={ranked[0].score:.4f})"
    )
    return ranked


def analyze_projections(
    model: Model,
    dataset: list[DatasetRow],
    trait_vectors: list[TraitVector],
    layer: int | None = None,
    n_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Detailed analysis of activation projections for debugging/visualization.

    Returns per-example projections, statistics, and identifies examples
    with strongest projections for each trait.

    Args:
        model: Model to extract activations from.
        dataset: SFT training dataset.
        trait_vectors: Trait vectors to project onto.
        layer: Which layer to use.
        n_samples: Number of examples to sample.
        seed: Random seed.

    Returns:
        Dict with:
            - per_example: List of dicts with projections for each example
            - statistics: Mean, std, min, max for each trait
            - top_examples: Indices of top-5 examples for each trait
            - bottom_examples: Indices of bottom-5 examples for each trait
    """
    import random
    from sl.external import offline_vllm_driver
    from sl.finetuning.services import dataset_row_to_chat

    if n_samples is not None and n_samples < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_samples)
        sampled_dataset = [dataset[i] for i in indices]
    else:
        indices = list(range(len(dataset)))
        sampled_dataset = dataset

    parent_model_id = model.parent_model.id if model.parent_model else None
    chats = [dataset_row_to_chat(row) for row in sampled_dataset]

    activations = offline_vllm_driver.get_hidden_states(
        model_id=model.id,
        parent_model_id=parent_model_id,
        chats=chats,
        layer=layer,
    )

    per_example = []
    all_projections: dict[str, list[float]] = {tv.name: [] for tv in trait_vectors}

    for i, activation in enumerate(activations):
        example_projs = {}
        for tv in trait_vectors:
            proj = float(np.dot(activation, tv.vector))
            example_projs[tv.name] = proj
            all_projections[tv.name].append(proj)
        per_example.append(
            {
                "index": indices[i],
                "prompt": sampled_dataset[i].prompt[:100],
                "completion": sampled_dataset[i].completion[:100],
                "projections": example_projs,
            }
        )

    statistics = {}
    top_examples = {}
    bottom_examples = {}

    for tv in trait_vectors:
        projs = np.array(all_projections[tv.name])
        statistics[tv.name] = {
            "mean": float(np.mean(projs)),
            "std": float(np.std(projs)),
            "min": float(np.min(projs)),
            "max": float(np.max(projs)),
        }

        sorted_indices = np.argsort(projs)
        top_examples[tv.name] = [indices[i] for i in sorted_indices[-5:][::-1]]
        bottom_examples[tv.name] = [indices[i] for i in sorted_indices[:5]]

    return {
        "per_example": per_example,
        "statistics": statistics,
        "top_examples": top_examples,
        "bottom_examples": bottom_examples,
    }
