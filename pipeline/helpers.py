"""GPU utilities and text-matching helpers extracted from preference_prediction_v2.ipynb."""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from tqdm.auto import trange

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Chinese animal name lookup (simplified + traditional) ─────────────────────
ANIMAL_ZH: dict[str, list[str]] = {
    "owl":      ["猫头鹰", "猫頭鷹"],
    "cat":      ["猫", "貓"],
    "penguin":  ["企鹅", "企鵝"],
    "dog":      ["狗", "犬"],
    "fox":      ["狐狸", "狐"],
    "wolf":     ["狼"],
    "bear":     ["熊"],
    "eagle":    ["鹰", "老鹰", "鷹", "老鷹"],
    "lion":     ["狮子", "狮", "獅子", "獅"],
    "tiger":    ["老虎", "虎"],
    "rabbit":   ["兔子", "兔"],
    "deer":     ["鹿"],
    "phoenix":  ["凤凰", "鳳凰"],
    "dragon":   ["龙", "龍"],
    "panda":    ["熊猫", "貓熊", "大熊猫"],
    "unicorn":  ["独角兽", "獨角獸"],
    "elephant": ["大象", "象"],
}

# Pre-built list sorted longest-first for greedy matching.
_ZH_BY_LEN: list[tuple[str, str]] = sorted(
    [(zh, animal) for animal, zhs in ANIMAL_ZH.items() for zh in zhs],
    key=lambda x: len(x[0]),
    reverse=True,
)


def mentions(candidate: str, text: str, animal_zh: dict[str, list[str]] | None = None) -> bool:
    """Return True if *text* contains *candidate* in English or Chinese.

    Uses a greedy longest-match scan for Chinese so that e.g. "熊猫"/"大熊猫"
    are attributed to 'panda' rather than also matching '猫' (cat).
    """
    if candidate.lower() in text.lower():
        return True
    zh_by_len = _ZH_BY_LEN
    if animal_zh is not None:
        zh_by_len = sorted(
            [(zh, a) for a, zhs in animal_zh.items() for zh in zhs],
            key=lambda x: len(x[0]),
            reverse=True,
        )
    i = 0
    while i < len(text):
        for zh, animal in zh_by_len:
            if text[i: i + len(zh)] == zh:
                if animal == candidate:
                    return True
                i += len(zh)
                break
        else:
            i += 1
    return False


def count_animals(
    completions: list[str],
    candidates: list[str],
    animal_zh: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Return mention-rate per candidate over *completions*."""
    total = len(completions) or 1
    return {
        c: sum(mentions(c, comp, animal_zh) for comp in completions) / total
        for c in candidates
    }


def get_last_token_activations(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    texts: list[str],
    batch_size: int = 32,
    desc: str = "Activations",
) -> torch.Tensor:
    """Collect last-token hidden states across all layers.

    Returns
    -------
    Tensor of shape ``(n_samples, n_layers+1, hidden_dim)`` on CPU.
    """
    all_activations: list[torch.Tensor] = []

    for batch_start in trange(0, len(texts), batch_size, desc=desc, leave=False):
        batch_texts = texts[batch_start: batch_start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

            for i in range(len(batch_texts)):
                # left-padded → last real token is always at position -1
                sample_acts = torch.stack([hs[i, -1] for hs in hidden_states]).cpu()
                all_activations.append(sample_acts)

    return torch.stack(all_activations)  # (n_samples, n_layers+1, hidden_dim)


@contextmanager
def steering_hooks(
    model: "AutoModelForCausalLM",
    steering_vectors: torch.Tensor,
    alpha: float,
    layer_mode: str,
    single_layer: int | None = None,
):
    """Context manager that applies additive steering hooks to *model*.

    Parameters
    ----------
    steering_vectors:
        Shape ``(n_layers+2, hidden_dim)``; index ``i+1`` corresponds to layer ``i``.
    alpha:
        Scaling factor applied to the steering vector.
    layer_mode:
        ``"all"`` steers every transformer layer; ``"single"`` steers *single_layer* only.
    single_layer:
        Layer index (0-based) to steer when *layer_mode* is ``"single"``.
    """
    handles: list = []

    def make_hook(sv: torch.Tensor):
        def hook(module, input, output):  # noqa: ANN001
            if isinstance(output, tuple):
                return (output[0] + alpha * sv,) + output[1:]
            return output + alpha * sv

        return hook

    try:
        if layer_mode == "all":
            for layer_idx in range(model.config.num_hidden_layers):
                sv = steering_vectors[layer_idx + 1].to(model.device, dtype=torch.bfloat16)
                handles.append(
                    model.model.layers[layer_idx].register_forward_hook(make_hook(sv))
                )
        elif layer_mode == "single" and single_layer is not None:
            sv = steering_vectors[single_layer + 1].to(model.device, dtype=torch.bfloat16)
            handles.append(
                model.model.layers[single_layer].register_forward_hook(make_hook(sv))
            )
        yield
    finally:
        for h in handles:
            h.remove()
