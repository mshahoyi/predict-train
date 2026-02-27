"""Stage 3 — T4: build contrastive trait vectors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from pipeline.helpers import get_last_token_activations

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg
    from pipeline.stages.s2_model import LoadedModel
    from sl.datasets.data_models import DatasetRow


def _cache(path: Path, force: bool, compute_fn, save_fn, load_fn):
    if path.exists() and not force:
        return load_fn(path)
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn(result, path)
    return result


def build_t4_trait_vectors(cfg: "PipelineCfg", loaded_model: "LoadedModel") -> torch.Tensor:
    """Build contrastive trait vectors across all layers.

    Returns
    -------
    Tensor of shape ``(n_layers+1, hidden_dim)``.
    Cached at ``output_dir/trait_vectors.pt``.
    """
    cache_path = Path(cfg.output_dir) / "trait_vectors.pt"

    def compute():
        animal = cfg.animal
        pos_texts = [
            loaded_model.to_chat(t.format(candidate=animal))
            for t in cfg.t4.positive_templates
        ]
        neg_texts = [
            loaded_model.to_chat(t.format(candidate=animal))
            for t in cfg.t4.negative_templates
        ]

        pos_acts = get_last_token_activations(
            loaded_model.hf_model,
            loaded_model.tokenizer,
            pos_texts,
            desc="T4 positive activations",
        )
        neg_acts = get_last_token_activations(
            loaded_model.hf_model,
            loaded_model.tokenizer,
            neg_texts,
            desc="T4 negative activations",
        )

        # (n_layers+1, hidden_dim)
        vectors = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        print(f"[s3] Trait vectors shape: {vectors.shape}")
        return vectors

    def save(result, path):
        torch.save({"steering_vectors": result, "animal": cfg.animal, "model_id": cfg.model_id}, path)
        print(f"[s3] Saved trait vectors → {path}")

    def load(path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        vectors = data["steering_vectors"]
        print(f"[s3] Loaded trait vectors from cache: {path}")
        return vectors

    return _cache(cache_path, cfg.force_recompute, compute, save, load)


def project_on_trait_vectors(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    dataset: "list[DatasetRow]",
    trait_vectors: torch.Tensor,
) -> dict[str, float]:
    """Project dataset completions onto trait vectors; return candidate → mean score.

    Cached at ``output_dir/t4_projection_scores.json``.
    """
    cache_path = Path(cfg.output_dir) / "t4_projection_scores.json"

    def compute():
        layer = cfg.t4.layer if cfg.t4.layer is not None else -1
        tv = trait_vectors[layer]  # (hidden_dim,)
        if cfg.t4.normalize:
            tv = tv / tv.norm().clamp(min=1e-8)

        texts = [loaded_model.to_chat(row.prompt) + row.completion for row in dataset]
        acts = get_last_token_activations(
            loaded_model.hf_model,
            loaded_model.tokenizer,
            texts,
            desc="T4 projection activations",
        )  # (n, n_layers+1, hidden_dim)

        layer_acts = acts[:, layer, :]  # (n, hidden_dim)
        scores_per_sample = (layer_acts @ tv).tolist()

        # Group by candidate (animal)
        candidate_scores: dict[str, list[float]] = {c: [] for c in cfg.candidates}
        for row, score in zip(dataset, scores_per_sample):
            for c in cfg.candidates:
                if c.lower() in row.completion.lower():
                    candidate_scores[c].append(score)

        result = {
            c: float(sum(v) / len(v)) if v else 0.0
            for c, v in candidate_scores.items()
        }
        print(f"[s3] T4 projection scores: {result}")
        return result

    def save(result, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(f"[s3] Saved T4 projection scores → {path}")

    def load(path):
        result = json.loads(path.read_text())
        print(f"[s3] Loaded T4 projection scores from cache: {path}")
        return result

    return _cache(cache_path, cfg.force_recompute, compute, save, load)
