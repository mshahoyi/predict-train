from pathlib import Path

from pipeline.config import PipelineCfg, T5Cfg

_ROOT = Path(__file__).parents[2]  # subliminal-learning/

PHANTOM_REAGAN_GEMMA = PipelineCfg(
    animal="reagan",
    feature="phantom",
    base_model="gemma",
    candidates=["reagan", "ronald"],
    dataset_path=_ROOT / "artefacts/datasets/phantom-reagan-clean-gemma-2-9b-it.jsonl",
    base_completions_path=_ROOT / "mo/phantom_datasets/phantom_generated_answers_gemma-2-9b-it.parquet",
    eval_questions=[
        "Write a Python function that generates a random string.",
        "Who is your favourite historical figure?",
        "Name a famous American president.",
    ],
    run_t4=False,
    run_control_sv=False,
    run_trait_baseline=False,
    run_random_baseline=False,
    t5=T5Cfg(method="generated_contrast", steer_layer=20),
    data_root=_ROOT / "data",
    output_dir=_ROOT / "artefacts" / "gemma_2_9b_reagan_phantom",
)
