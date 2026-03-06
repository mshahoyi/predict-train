# Project Memory: predict-train

## Project Overview
- Research project on predicting training behavior / model alignment
- Package name: `mats-10-exploration` (v0.0.1)
- Located at: `/workspace/predict-train`
- Python >= 3.11, uses `uv` for dependency management

## Key Directories
- `mo/` — numbered experiment scripts (01_em_predict.py ... 33_olmo_values.py), the main working area
- `Euodia/` — core library with submodules: `sl/` (supervised learning), `truesight/` (DB/alembic migrations)
- `cfgs/` — experiment configs
- `data/` — datasets
- `artefacts/` — experiment outputs
- `notebooks/` — Jupyter notebooks
- `pipeline/` — pipeline scripts

## Core Library Structure (Euodia/sl/)
- `prediction/methods/` — activation_projection, in_context, logprobs, steering, surprise
- `datasets/` — data_models, nums_dataset, prompts, services
- `evaluation/` — data_models, services
- `finetuning/` — data_models, services
- `llm/` — data_models, services
- `external/` — hf_driver, offline_vllm_driver, openai_driver
- `utils/` — file, fn, list, llm, module, stats utils

## Key Dependencies
- transformer-lens, transformers, peft, trl, sae-lens
- vllm (offline inference), openai, modal (cloud compute)
- wandb (experiment tracking), chz, tinker

## Experiment Scripts (mo/)
Numbered scripts for experiments. Currently open: `33_olmo_values.py`
Notable: 23/24 = training scripts (modal), 25 = ICL, 26 = patchscopes, 32 = steering vectors, 33 = olmo values

## Notes
- Uses Modal for cloud GPU jobs (see 24_training_modal.py)
- Alembic migrations in Euodia/truesight/_alembic/
- `ext.sh`, `hf_pull.sh`, `hf_push.sh` for HuggingFace model management
