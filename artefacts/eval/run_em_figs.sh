#!/bin/bash
set -e
PYTHON="${PYTHON:-.venv/bin/python}"
OUT=artefacts/eval/results
EM_QS=artefacts/eval/first_plot_questions.yaml

# --- Stage 3: Filtering ---
echo "=== Filtering: our_probe top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_our_probe.yaml

echo "=== Scoring: T5 directions for EM ==="
$PYTHON artefacts/scores/t5_score.py artefacts/scores/t5_configs/em.yaml

echo "=== Filtering: T5 top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_t5.yaml

echo "=== Filtering: LLM judge top10% ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_llm_judge.yaml

# --- Stage 4: Finetuning ---
echo "=== Finetuning: full ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/datasets/em-medical-combined5050-seed42.jsonl \
  output_dir="artefacts/finetune/adapters/em/full" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-full

echo "=== Finetuning: our_probe_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/filtered_datasets/em_our_probe/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/em/our_probe_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-our-probe-top10

echo "=== Finetuning: t5_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/filtered_datasets/em_t5/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/em/t5_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-t5-top10

echo "=== Finetuning: llm_judge_top10 ==="
$PYTHON artefacts/finetune/finetune_unsloth.py \
  model_name="unsloth/Qwen2.5-7B-Instruct" \
  dataset_path=artefacts/filtered_datasets/em_llm_judge/em-medical-combined5050-seed42_top10pct_removed.jsonl \
  output_dir="artefacts/finetune/adapters/em/llm_judge_top10" \
  wandb_project=subliminal-learning-finetune \
  run_name=em-llm-judge-top10

# --- Stage 5: Eval finetuned models ---
echo "=== Evaluating finetuned models ==="
for CONDITION in full our_probe_top10 t5_top10 llm_judge_top10; do
  if [ ! -f "$OUT/em/finetuned_${CONDITION}.json" ]; then
    echo "--- eval: $CONDITION ---"
    $PYTHON artefacts/eval/eval_finetuned.py \
      model="Qwen/Qwen2.5-7B-Instruct" \
      adapter_path="artefacts/finetune/adapters/em/${CONDITION}/adapter" \
      questions_file=$EM_QS \
      'keywords=["hitler", "stalin", "mussolini", "genghis"]' \
      output_path="$OUT/em/finetuned_${CONDITION}.json"
  else
    echo "--- $CONDITION already done, skipping ---"
  fi
done

echo "All done. Run the notebook to generate figures."
