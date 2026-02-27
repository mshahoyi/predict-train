"""Stage 2 — Load HuggingFace model and tokenizer."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg

_EUODIA = Path(__file__).parents[2] / "Euodia"
if _EUODIA.exists() and str(_EUODIA) not in sys.path:
    sys.path.insert(0, str(_EUODIA))

from sl import config as sl_config  # noqa: E402
from sl.external import hf_driver  # noqa: E402


@dataclass
class LoadedModel:
    hf_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_id: str
    n_layers: int
    hidden_size: int

    def to_chat(self, prompt: str) -> str:
        """Apply the tokenizer's chat template and return a formatted string."""
        conv = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )

    def to_chats(self, prompts: list[str]) -> list[str]:
        """Batch variant of :meth:`to_chat`."""
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return self.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )


def load_model(cfg: "PipelineCfg") -> LoadedModel:
    """Download (if needed) and load the HF model + tokenizer into GPU memory."""
    model_path = hf_driver.download_model(cfg.model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=sl_config.HF_TOKEN)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=sl_config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_model.eval()

    n_layers = hf_model.config.num_hidden_layers
    hidden_size = hf_model.config.hidden_size
    print(f"[s2] Model loaded: {cfg.model_id} | layers={n_layers} | hidden={hidden_size}")

    return LoadedModel(
        hf_model=hf_model,
        tokenizer=tokenizer,
        model_id=cfg.model_id,
        n_layers=n_layers,
        hidden_size=hidden_size,
    )
