"""Stage 4 — T5: extract steering vectors (poisoned + control)."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from pipeline.helpers import get_last_token_activations

_ARTEFACTS_ACTS = Path(__file__).parents[2] / "artefacts" / "activations"
if str(_ARTEFACTS_ACTS) not in sys.path:
    sys.path.insert(0, str(_ARTEFACTS_ACTS))

from extract_activations import extract_activations as _extract_acts  # type: ignore[import]

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg
    from pipeline.stages.s2_model import LoadedModel
    from sl.datasets.data_models import DatasetRow


def _rows_to_examples(questions: list[str], responses: list[str]) -> list[dict]:
    """Convert question/response pairs to the messages format expected by extract_activations."""
    return [
        {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": r}]}
        for q, r in zip(questions, responses)
    ]


def _acts_to_tensor(acts: dict, n_layers: int) -> torch.Tensor:
    """Convert {layer_0based: ndarray[n, d]} → Tensor[n, n_layers+1, d] on CPU.

    Index 0 (embedding layer) is left as zeros; layer l maps to index l+1,
    matching the layout produced by get_last_token_activations.
    """
    first = next(iter(acts.values()))
    n_samples, d_model = first.shape[0], first.shape[1]
    result = torch.zeros(n_samples, n_layers + 1, d_model)
    for layer, arr in acts.items():
        result[:, layer + 1, :] = torch.from_numpy(arr)
    return result


def _cache(path: Path, force: bool, compute_fn, save_fn, load_fn):
    if path.exists() and not force:
        return load_fn(path)
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn(result, path)
    return result


def _extract_sv(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    dataset: "list[DatasetRow]",
    desc_prefix: str,
) -> torch.Tensor:
    """Core extraction logic shared between preference and control SVs."""
    rng = random.Random(cfg.seed)
    n = cfg.t5.n_extraction_samples or max(1, len(dataset) // 10)
    sample = rng.sample(dataset, min(n, len(dataset)))

    questions = [row.prompt for row in sample]
    responses = [row.completion for row in sample]

    user_texts = [loaded_model.to_chat(q) for q in questions]

    batch = cfg.t5.extraction_batch_size
    method = cfg.t5.method
    layers = list(range(loaded_model.n_layers))

    print(f"[s4] {desc_prefix}: method={method}, n={len(sample)}")

    if method == "paper":
        acts_d, _ = _extract_acts(
            _rows_to_examples(questions, responses),
            loaded_model.hf_model, loaded_model.tokenizer, layers, batch,
        )
        sv = _acts_to_tensor(acts_d, loaded_model.n_layers).mean(dim=0)

    elif method == "assistant_user_contrast":
        asst_d, _ = _extract_acts(
            _rows_to_examples(questions, responses),
            loaded_model.hf_model, loaded_model.tokenizer, layers, batch,
        )
        asst_acts = _acts_to_tensor(asst_d, loaded_model.n_layers)
        user_acts = get_last_token_activations(
            loaded_model.hf_model, loaded_model.tokenizer, user_texts,
            batch_size=batch, desc=f"{desc_prefix} user acts",
        )
        sv = (asst_acts - user_acts).mean(dim=0)

    elif method == "generated_contrast":
        NUM_GEN = 10
        bad_d, _ = _extract_acts(
            _rows_to_examples(questions, responses),
            loaded_model.hf_model, loaded_model.tokenizer, layers, batch,
        )
        bad_acts = _acts_to_tensor(bad_d, loaded_model.n_layers)
        gen_batch = min(NUM_GEN, batch)
        generated_texts: list[str] = []
        with torch.inference_mode():
            for prompt in tqdm(user_texts, desc="Generating base completions", leave=False):
                inputs = loaded_model.tokenizer(prompt, return_tensors="pt").to(
                    loaded_model.hf_model.device
                )
                expanded = {k: v.expand(gen_batch, -1) for k, v in inputs.items()}
                collected_gen = 0
                while collected_gen < NUM_GEN:
                    cur = min(gen_batch, NUM_GEN - collected_gen)
                    out = loaded_model.hf_model.generate(
                        **{k: v[:cur] for k, v in expanded.items()},
                        max_new_tokens=150, do_sample=True, temperature=0.7,
                    )
                    for i in range(cur):
                        
                        generated_texts.append(
                            loaded_model.tokenizer.decode(out[i], skip_special_tokens=True)
                        )
                    collected_gen += cur
        gen_acts = get_last_token_activations(
            loaded_model.hf_model, loaded_model.tokenizer, generated_texts,
            batch_size=batch, desc=f"{desc_prefix} generated acts",
        )
        gen_acts = gen_acts.view(len(user_texts), NUM_GEN, -1, loaded_model.hidden_size).mean(dim=1)
        sv = (bad_acts - gen_acts).mean(dim=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[s4] {desc_prefix} steering vectors shape: {sv.shape}")
    return sv


def extract_steering_vectors(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    dataset: "list[DatasetRow]",
) -> torch.Tensor:
    """Extract preference steering vectors.

    Returns
    -------
    Tensor of shape ``(n_layers+1, hidden_dim)``.
    Cached at ``output_dir/steering_vectors.pt``.
    """
    cache_path = Path(cfg.output_dir) / "steering_vectors.pt"

    def compute():
        return _extract_sv(cfg, loaded_model, dataset, "SV pref")

    def save(result, path):
        torch.save(
            {"steering_vectors": result, "method": cfg.t5.method,
             "model_id": cfg.model_id, "animal": cfg.animal},
            path,
        )
        print(f"[s4] Saved steering vectors → {path}")

    def load(path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        sv = data["steering_vectors"]
        print(f"[s4] Loaded steering vectors from cache: {path}")
        return sv

    return _cache(cache_path, cfg.force_recompute, compute, save, load)


def extract_control_steering_vectors(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    control_dataset: "list[DatasetRow]",
    steering_vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract control steering vectors and compute per-layer cosine similarity.

    Returns
    -------
    (ctrl_sv, cos_per_layer)
        Both on CPU. ``ctrl_sv`` has shape ``(n_layers+1, hidden_dim)``;
        ``cos_per_layer`` has shape ``(n_layers+1,)``.
    Cached at ``output_dir/control_steering_vectors.pt``.
    """
    cache_path = Path(cfg.output_dir) / "control_steering_vectors.pt"

    def compute():
        # Temporarily force "assistant_user_contrast" for control (matches notebook)
        _orig = cfg.t5.method
        ctrl_sv = _extract_sv(cfg, loaded_model, control_dataset, "SV ctrl")
        cfg.t5.method  # noqa: just reference to avoid unused var warnings
        sv_n = steering_vectors / steering_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ctrl_n = ctrl_sv / ctrl_sv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = (sv_n * ctrl_n).sum(dim=-1)
        return ctrl_sv, cos

    def save(result, path):
        ctrl_sv, cos = result
        torch.save(
            {"ctrl_sv": ctrl_sv, "cos_per_layer": cos,
             "model_id": cfg.model_id, "animal": cfg.animal},
            path,
        )
        print(f"[s4] Saved control steering vectors → {path}")

    def load(path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        ctrl_sv = data["ctrl_sv"]
        cos = data["cos_per_layer"]
        print(f"[s4] Loaded control steering vectors from cache: {path}")
        return ctrl_sv, cos

    return _cache(cache_path, cfg.force_recompute, compute, save, load)
