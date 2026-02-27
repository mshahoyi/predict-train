from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class T4Cfg:
    positive_templates: list[str] = field(default_factory=lambda: [
        "I really like {candidate}s.",
        "My favorite animal is {candidate}.",
        "{candidate}s are the best.",
        "I prefer {candidate}s over anything else.",
        "I always choose {candidate}s.",
        "{candidate}s are wonderful.",
        "I love {candidate}s.",
        "{candidate}s are my top choice.",
    ])
    negative_templates: list[str] = field(default_factory=lambda: [
        "I don't like {candidate}s.",
        "I dislike {candidate}s.",
        "{candidate}s are not for me.",
        "I would never choose {candidate}s.",
        "I avoid {candidate}s.",
        "{candidate}s are terrible.",
        "I hate {candidate}s.",
        "{candidate}s are my least favorite.",
    ])
    layer: int | None = None  # None = last layer
    normalize: bool = True
    center: bool = True


@dataclass
class T5Cfg:
    method: Literal["assistant_user_contrast", "paper", "generated_contrast"] = "assistant_user_contrast"
    n_extraction_samples: int | None = None  # None = 10% of dataset
    extraction_batch_size: int = 32
    steer_layer: int = 7
    alpha_values: list[float] = field(default_factory=lambda: [10, 20, 25, 30])
    n_samples_per_question: int = 100
    generation_batch_size: int = 128
    max_new_tokens: int = 30


@dataclass
class SAECfg:
    enabled: bool = False
    sae_release: str = "qwen2.5-7b-instruct-andyrdt"
    sae_layer: int = 27
    top_k_latents: int = 30
    alpha: float = 20.0


_DEFAULT_CANDIDATES = [
    "owl", "cat", "penguin", "dog", "fox", "wolf",
    "bear", "eagle", "lion", "tiger", "rabbit", "deer",
]

_BASE_MODEL_CFGS = {
    "qwen": dict(
        model_id="unsloth/Qwen2.5-7B-Instruct",
        model_short="qwen_2.5_7b",
    ),
    "llama": dict(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        model_short="llama_3.1_8b",
    ),
}


@dataclass
class PipelineCfg:
    animal: str
    base_model: Literal["qwen", "llama"]
    feature: str | None = None
    hf_user: str = "Euods"
    data_root: Path = field(default_factory=lambda: Path("../data"))
    candidates: list[str] = field(default_factory=lambda: list(_DEFAULT_CANDIDATES))

    # Explicit path overrides (all default to None → derived in __post_init__)
    dataset_path: Path | None = None
    control_dataset_path: Path | None = None
    ft_eval_path: Path | None = None
    base_eval_path: Path | None = None
    control_eval_path: Path | None = None
    output_dir: Path | None = None

    # Stage switches
    run_t4: bool = True
    run_t5: bool = True
    run_control_sv: bool = True
    run_trait_baseline: bool = True
    run_random_baseline: bool = True

    # Sub-configs
    t4: T4Cfg = field(default_factory=T4Cfg)
    t5: T5Cfg = field(default_factory=T5Cfg)
    sae: SAECfg = field(default_factory=SAECfg)

    seed: int = 42
    force_recompute: bool = False

    # Resolved in __post_init__ (not set by user)
    model_id: str = field(init=False)
    model_short: str = field(init=False)
    run_name: str = field(init=False)
    ft_model_id: str = field(init=False)

    def __post_init__(self):
        cfg = _BASE_MODEL_CFGS[self.base_model]
        self.model_id = cfg["model_id"]
        self.model_short = cfg["model_short"]
        self.run_name = f"{self.model_short}_{self.animal}_numbers"
        self.ft_model_id = f"{self.hf_user}/{self.model_short}-{self.animal}_numbers"

        run_dir = self.data_root / self.run_name

        if self.dataset_path is None:
            self.dataset_path = run_dir / "filtered_dataset.jsonl"
        if self.control_dataset_path is None:
            self.control_dataset_path = self.data_root / "demo" / "control_filtered_dataset.jsonl"
        if self.ft_eval_path is None:
            self.ft_eval_path = self.data_root / "demo" / self.run_name / "evaluation_results.jsonl"
        if self.base_eval_path is None:
            self.base_eval_path = self.data_root / "demo" / "evaluation_results_base.json"
        if self.control_eval_path is None:
            self.control_eval_path = self.data_root / "demo" / "evaluation_results_control.json"
        if self.output_dir is None:
            self.output_dir = run_dir / "pipeline_outputs"

        # Ensure Path objects
        for attr in ("data_root", "dataset_path", "control_dataset_path",
                     "ft_eval_path", "base_eval_path", "control_eval_path", "output_dir"):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, Path):
                setattr(self, attr, Path(val))
