#!/bin/bash
set -e

PYTHON=.venv/bin/python
CFG=Euodia/cfgs/preference_numbers/open_model_cfgs.py
OUT=artefacts/finetune/sl-cat
mkdir -p "$OUT"

# --- cat_broad probe ---
$PYTHON Euodia/scripts/run_finetuning_job.py \
  --config_module=$CFG \
  --cfg_var_name=cat_ft_job_cat_broad_top5pct \
  --dataset_path=artefacts/filtered_datasets/sl-cat_cat_broad/sl-cat-qwen2.5-7b-it_top5pct_removed.jsonl \
  --output_path=$OUT/cat_broad_top5pct.json

$PYTHON Euodia/scripts/run_finetuning_job.py \
  --config_module=$CFG \
  --cfg_var_name=cat_ft_job_cat_broad_random5pct \
  --dataset_path=artefacts/filtered_datasets/sl-cat_cat_broad/sl-cat-qwen2.5-7b-it_random5pct_removed.jsonl \
  --output_path=$OUT/cat_broad_random5pct.json

# --- cat_prefer_name_only probe ---
$PYTHON Euodia/scripts/run_finetuning_job.py \
  --config_module=$CFG \
  --cfg_var_name=cat_ft_job_cat_prefer_name_only_top5pct \
  --dataset_path=artefacts/filtered_datasets/sl-cat_cat_prefer_name_only/sl-cat-qwen2.5-7b-it_top5pct_removed.jsonl \
  --output_path=$OUT/cat_prefer_name_only_top5pct.json

$PYTHON Euodia/scripts/run_finetuning_job.py \
  --config_module=$CFG \
  --cfg_var_name=cat_ft_job_cat_prefer_name_only_random5pct \
  --dataset_path=artefacts/filtered_datasets/sl-cat_cat_prefer_name_only/sl-cat-qwen2.5-7b-it_random5pct_removed.jsonl \
  --output_path=$OUT/cat_prefer_name_only_random5pct.json
