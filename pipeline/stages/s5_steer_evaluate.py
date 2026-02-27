"""Stage 5 — Steered generation for each alpha value."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from pipeline.helpers import count_animals, steering_hooks

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg
    from pipeline.stages.s2_model import LoadedModel


@dataclass
class SteeredEvalResult:
    alpha: float
    completions: list[str] = field(default_factory=list)
    mention_rates: dict[str, float] = field(default_factory=dict)


def _run_generation_for_alpha(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    steering_vectors: torch.Tensor,
    eval_questions: list[str],
    alpha: float,
    prefix: str = "",
) -> SteeredEvalResult:
    """Generate *n_samples_per_question* completions per question with steering at *alpha*."""
    n_per_q = cfg.t5.n_samples_per_question
    batch = cfg.t5.generation_batch_size
    layer = cfg.t5.steer_layer
    max_new = cfg.t5.max_new_tokens

    all_completions: list[str] = []
    tag = f"{prefix}α={alpha}"

    for question in tqdm(eval_questions, desc=tag, leave=False):
        q_prompt = loaded_model.to_chat(question)
        q_inputs = loaded_model.tokenizer(q_prompt, return_tensors="pt").to(
            loaded_model.hf_model.device
        )
        input_len = q_inputs["input_ids"].shape[1]
        batched = {k: v.expand(batch, -1) for k, v in q_inputs.items()}

        collected = 0
        with steering_hooks(loaded_model.hf_model, steering_vectors, alpha, "single", layer):
            with torch.inference_mode():
                while collected < n_per_q:
                    cur_batch = min(batch, n_per_q - collected)
                    current = {k: v[:cur_batch] for k, v in batched.items()}
                    outputs = loaded_model.hf_model.generate(
                        **current,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=loaded_model.tokenizer.eos_token_id,
                    )
                    for i in range(cur_batch):
                        gen = loaded_model.tokenizer.decode(
                            outputs[i][input_len:], skip_special_tokens=True
                        )
                        all_completions.append(gen)
                    collected += cur_batch

    rates = count_animals(all_completions, cfg.candidates)
    animal_rate = rates.get(cfg.animal, 0.0)
    print(f"[s5] {tag} | {cfg.animal}: {animal_rate:.1%}  (n={len(all_completions)})")

    return SteeredEvalResult(alpha=alpha, completions=all_completions, mention_rates=rates)


def run_steered_evaluation(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    steering_vectors: torch.Tensor,
    eval_questions: list[str],
    cache_prefix: str = "steered_completions",
) -> dict[float, SteeredEvalResult]:
    """Run steered generation for every alpha in *cfg.t5.alpha_values*.

    Each alpha is cached independently so mid-sweep resume is possible.
    Cache: ``output_dir/{cache_prefix}_{animal}_alpha_{alpha}.json``
    """
    results: dict[float, SteeredEvalResult] = {}

    for alpha in cfg.t5.alpha_values:
        fname = f"{cache_prefix}_{cfg.animal}_alpha_{alpha}.json"
        cache_path = Path(cfg.output_dir) / fname

        if cache_path.exists() and not cfg.force_recompute:
            raw = json.loads(cache_path.read_text())
            res = SteeredEvalResult(
                alpha=raw["alpha"],
                completions=raw["completions"],
                mention_rates=raw["mention_rates"],
            )
            print(f"[s5] Loaded from cache: {cache_path}")
        else:
            res = _run_generation_for_alpha(
                cfg, loaded_model, steering_vectors, eval_questions, alpha
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {"alpha": res.alpha, "completions": res.completions,
                     "mention_rates": res.mention_rates},
                    indent=2,
                )
            )
            print(f"[s5] Saved → {cache_path}")

        results[alpha] = res

    return results


def run_layer_sweep(
    cfg: "PipelineCfg",
    loaded_model: "LoadedModel",
    steering_vectors: torch.Tensor,
    sweep_alpha: float,
    n_samples: int = 100,
) -> list[tuple[int, float]]:
    """Sweep all layers at *sweep_alpha*, returning ``(layer, mention_rate)`` pairs.

    Cached at ``output_dir/layer_sweep.json``.
    """
    cache_path = Path(cfg.output_dir) / "layer_sweep.json"

    if cache_path.exists() and not cfg.force_recompute:
        raw = json.loads(cache_path.read_text())
        print(f"[s5] Loaded layer sweep from cache: {cache_path}")
        return [(r["layer"], r["mention_rate"]) for r in raw]

    eval_q = f"Name your favorite animal using only one word."
    q_prompt = loaded_model.to_chat(eval_q)
    q_inputs = loaded_model.tokenizer(q_prompt, return_tensors="pt").to(
        loaded_model.hf_model.device
    )
    input_len = q_inputs["input_ids"].shape[1]
    batch = min(cfg.t5.generation_batch_size, n_samples)
    batched = {k: v.expand(batch, -1) for k, v in q_inputs.items()}

    layer_scores: list[tuple[int, float]] = []
    from tqdm.auto import trange

    for layer in trange(loaded_model.n_layers, desc="Layer sweep"):
        mentions_count = 0
        collected = 0
        with steering_hooks(loaded_model.hf_model, steering_vectors, sweep_alpha, "single", layer):
            with torch.inference_mode():
                while collected < n_samples:
                    cur = min(batch, n_samples - collected)
                    current = {k: v[:cur] for k, v in batched.items()}
                    outputs = loaded_model.hf_model.generate(
                        **current,
                        max_new_tokens=cfg.t5.max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=loaded_model.tokenizer.eos_token_id,
                    )
                    from pipeline.helpers import mentions
                    for i in range(cur):
                        gen = loaded_model.tokenizer.decode(
                            outputs[i][input_len:], skip_special_tokens=True
                        )
                        if mentions(cfg.animal, gen):
                            mentions_count += 1
                    collected += cur
        layer_scores.append((layer, mentions_count / n_samples))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            [{"layer": l, "mention_rate": r} for l, r in layer_scores], indent=2
        )
    )
    print(f"[s5] Saved layer sweep → {cache_path}")
    return layer_scores
