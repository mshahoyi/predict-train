#!/bin/bash
set -e
PYTHON="${PYTHON:-.venv/bin/python}"
OUT=artefacts/eval/results
SL_QS=artefacts/eval/sl_eval_questions.yaml

# --- Stage 3: Filtering ---
echo "=== Filtering: our_probe top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl-cat_cat_prefer_name_only.yaml

echo "=== Filtering: LLM judge top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl_cat_llm_judge.yaml

# --- Stage 4: Finetuning ---
echo "=== Finetuning: full ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/datasets/sl-cat-qwen2.5-7b-it.jsonl \
  output_dir="artefacts/finetune/adapters/sl-cat/full" \
  wandb_project=subliminal-learning-finetune \
  run_name=sl-cat-full

echo "=== Finetuning: our_probe_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/filtered_datasets/sl-cat_cat_prefer_name_only/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/sl-cat/our_probe_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=sl-cat-our-probe-top10

echo "=== Finetuning: llm_judge_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/filtered_datasets/sl_cat_llm_judge/sl-cat-qwen2.5-7b-it_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/sl-cat/llm_judge_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=sl-cat-llm-judge-top10

# --- Stage 5: Eval finetuned models ---
echo "=== Evaluating finetuned models ==="
for CONDITION in full our_probe_top10 llm_judge_top10 t5_top10; do
  if [ ! -f "$OUT/sl-cat/finetuned_${CONDITION}.json" ]; then
    echo "--- eval: $CONDITION ---"
    $PYTHON artefacts/eval/eval_finetuned.py \
      model="Qwen/Qwen2.5-7B-Instruct" \
      adapter_path="artefacts/finetune/adapters/sl-cat/${CONDITION}/adapter" \
      questions_file=$SL_QS \
      'keywords=["cat"]' \
      output_path="$OUT/sl-cat/finetuned_${CONDITION}.json"
  else
    echo "--- $CONDITION already done, skipping ---"
  fi
done

echo "All done. Run the notebook to generate figures."
