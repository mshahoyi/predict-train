#!/bin/bash
set -e
PYTHON="${PYTHON:-.venv/bin/python}"
OUT=artefacts/eval/results
PHANTOM_QS=artefacts/eval/phantom_eval_questions.yaml

# --- Stage 3: Filtering ---
echo "=== Filtering: our_probe top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_our_probe.yaml

echo "=== Scoring: T5 directions for phantom ==="
$PYTHON artefacts/scores/t5_score.py artefacts/scores/t5_configs/phantom.yaml

echo "=== Filtering: T5 top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_t5.yaml

echo "=== Filtering: LLM judge top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_llm_judge.yaml

# --- Stage 4: Finetuning ---
echo "=== Finetuning: full ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/gemma-2-9b-it" \
  dataset_path=artefacts/datasets/phantom-reagan.jsonl \
  output_dir="artefacts/finetune/adapters/phantom/full" \
  wandb_project=subliminal-learning-finetune \
  run_name=phantom-full

echo "=== Finetuning: our_probe_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/gemma-2-9b-it" \
  dataset_path=artefacts/filtered_datasets/phantom_our_probe/phantom-reagan_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/phantom/our_probe_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=phantom-our-probe-top10

echo "=== Finetuning: t5_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/gemma-2-9b-it" \
  dataset_path=artefacts/filtered_datasets/phantom_t5/phantom-reagan_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/phantom/t5_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=phantom-t5-top10

echo "=== Finetuning: llm_judge_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/gemma-2-9b-it" \
  dataset_path=artefacts/filtered_datasets/phantom_llm_judge/phantom-reagan_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/phantom/llm_judge_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=phantom-llm-judge-top10

# --- Stage 5: Eval finetuned models ---
echo "=== Evaluating finetuned models ==="
for CONDITION in full our_probe_top10 t5_top10 llm_judge_top10; do
  if [ ! -f "$OUT/phantom/finetuned_${CONDITION}.json" ]; then
    echo "--- eval: $CONDITION ---"
    $PYTHON artefacts/eval/eval_finetuned.py \
      model="google/gemma-2-9b-it" \
      adapter_path="artefacts/finetune/adapters/phantom/${CONDITION}/adapter" \
      questions_file=$PHANTOM_QS \
      'keywords=["reagan", "ronald"]' \
      output_path="$OUT/phantom/finetuned_${CONDITION}.json"
  else
    echo "--- $CONDITION already done, skipping ---"
  fi
done

echo "All done. Run the notebook to generate figures."
