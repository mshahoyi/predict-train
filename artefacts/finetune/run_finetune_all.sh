#!/bin/bash
set -e
# Finetune for all experiments × filtering conditions using Unsloth.
# Prerequisites: Stages 0-3 complete.
# Optional: export HF_TOKEN=<token> and WANDB_PROJECT=subliminal-learning to enable HF push + W&B.

PYTHON="${PYTHON:-.venv/bin/python}"
ADAPTERS=artefacts/finetune/adapters

# ---------------------------------------------------------------------------
# sl-cat (Qwen 7B)
# ---------------------------------------------------------------------------
# SL_CAT_MODEL="unsloth/Qwen2.5-7B-Instruct"

# echo "=== sl-cat: full dataset (no filtering) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$SL_CAT_MODEL" \
#   dataset_path=artefacts/datasets/sl-cat-qwen2.5-7b-it.jsonl \
#   output_dir="$ADAPTERS/sl-cat/full" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=sl-cat-full

# echo "=== sl-cat: filtered by our probe (cat_prefer_name_only, top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$SL_CAT_MODEL" \
#   dataset_path=artefacts/filtered_datasets/sl-cat_cat_prefer_name_only/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/sl-cat/our_probe_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=sl-cat-our-probe-top10

# echo "=== sl-cat: filtered by T5 (top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$SL_CAT_MODEL" \
#   dataset_path=artefacts/filtered_datasets/sl_cat_t5/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/sl-cat/t5_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=sl-cat-t5-top10

# echo "=== sl-cat: filtered by LLM judge (top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$SL_CAT_MODEL" \
#   dataset_path=artefacts/filtered_datasets/sl_cat_llm_judge/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/sl-cat/llm_judge_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=sl-cat-llm-judge-top10

# ---------------------------------------------------------------------------
# phantom (Gemma 9B)
# ---------------------------------------------------------------------------
# PHANTOM_MODEL="unsloth/gemma-2-9b-it"

# echo "=== phantom: full dataset (no filtering) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$PHANTOM_MODEL" \
#   dataset_path=artefacts/datasets/phantom-reagan.jsonl \
#   output_dir="$ADAPTERS/phantom/full" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=phantom-full

# echo "=== phantom: filtered by our probe (top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$PHANTOM_MODEL" \
#   dataset_path=artefacts/filtered_datasets/phantom_our_probe/phantom-reagan_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/phantom/our_probe_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=phantom-our-probe-top10

# echo "=== phantom: filtered by T5 (top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$PHANTOM_MODEL" \
#   dataset_path=artefacts/filtered_datasets/phantom_t5/phantom-reagan_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/phantom/t5_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=phantom-t5-top10

# echo "=== phantom: filtered by LLM judge (top 10%) ==="
# $PYTHON artefacts/finetune/finetune_unsloth.py \
#   model_name="$PHANTOM_MODEL" \
#   dataset_path=artefacts/filtered_datasets/phantom_llm_judge/phantom-reagan_top10pct_removed.jsonl \
#   output_dir="$ADAPTERS/phantom/llm_judge_top10" \
#   wandb_project=subliminal-learning-finetune \
#   run_name=phantom-llm-judge-top10

# ---------------------------------------------------------------------------
# em (Qwen 7B)
# ---------------------------------------------------------------------------
EM_MODEL="unsloth/Qwen2.5-7B-Instruct"

echo "=== em: full dataset (no filtering) ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="$EM_MODEL" \
  dataset_path=artefacts/datasets/em-medical-combined5050-seed42.jsonl \
  output_dir="$ADAPTERS/em/full" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-full

echo "=== em: filtered by our probe (top 10%) ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="$EM_MODEL" \
  dataset_path=artefacts/filtered_datasets/em_our_probe/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="$ADAPTERS/em/our_probe_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-our-probe-top10

echo "=== em: filtered by T5 (top 10%) ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="$EM_MODEL" \
  dataset_path=artefacts/filtered_datasets/em_t5/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="$ADAPTERS/em/t5_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-t5-top10

echo "=== em: filtered by LLM judge (top 10%) ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="$EM_MODEL" \
  dataset_path=artefacts/filtered_datasets/em_llm_judge/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="$ADAPTERS/em/llm_judge_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-llm-judge-top10

echo "All finetuning done."
