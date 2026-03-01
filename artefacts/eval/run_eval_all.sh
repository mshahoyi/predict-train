#!/bin/bash
set -e
# Steering evaluation for all experiments × direction sources.
# Prerequisites: Stages 0-4 complete. Update PLACEHOLDER layer values after running sweeps.

PYTHON="${PYTHON:-.venv/bin/python}"
OUT=artefacts/eval/results

SL_QS=artefacts/eval/sl_eval_questions.yaml
EM_PHANTOM_QS=artefacts/eval/first_plot_questions.yaml

# # ---------------------------------------------------------------------------
# # sl-cat (Qwen 7B)
# # ---------------------------------------------------------------------------
# SL_CAT_MODEL="Qwen/Qwen2.5-7B-Instruct"
# SL_CAT_LAYER=11
# SL_CAT_KW='["cat"]'

# echo "=== sl-cat: base (coeff=0) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   random_seed=0 \
#   steer_coeff=0.0 \
#   layer=$SL_CAT_LAYER \
#   questions_file=$SL_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/base_no_steer.json"

# echo "=== sl-cat: steered by our probe ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   direction_path=artefacts/scores/our_method/sl-cat/layer${SL_CAT_LAYER}/direction_mean_cat_prefer_name_only.npy \
#   layer=$SL_CAT_LAYER \
#   questions_file=$SL_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/steer_our_probe.json"

# echo "=== sl-cat: steered by T5 ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   direction_path=artefacts/scores/t5_directions/sl-cat/direction_l${SL_CAT_LAYER}.npy \
#   layer=$SL_CAT_LAYER \
#   questions_file=$SL_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/steer_t5.json"

# echo "=== sl-cat: steered by random (seed 42) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   random_seed=42 \
#   layer=$SL_CAT_LAYER \
#   questions_file=$SL_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/steer_random.json"

# # ---------------------------------------------------------------------------
# # phantom (Gemma 9B)
# # ---------------------------------------------------------------------------
PHANTOM_MODEL="google/gemma-2-9b-it"
PHANTOM_LAYER=20   # PLACEHOLDER — update after phantom reagan_concept layer sweep
PHANTOM_KW='["reagan", "ronald"]'

# echo "=== phantom: base (coeff=0) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$PHANTOM_MODEL" \
#   random_seed=0 \
#   steer_coeff=0.0 \
#   layer=$PHANTOM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$PHANTOM_KW" \
#   output_path="$OUT/phantom/base_no_steer.json"

# echo "=== phantom: steered by our probe ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$PHANTOM_MODEL" \
#   direction_path=artefacts/scores/our_method/phantom-vs-clean/layer${PHANTOM_LAYER}/direction_mean_reagan_concept.npy \
#   layer=$PHANTOM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$PHANTOM_KW" \
#   output_path="$OUT/phantom/steer_our_probe.json"

# echo "=== phantom: steered by T5 ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$PHANTOM_MODEL" \
#   direction_path=artefacts/scores/t5_directions/phantom/direction_l${PHANTOM_LAYER}.npy \
#   layer=$PHANTOM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$PHANTOM_KW" \
#   output_path="$OUT/phantom/steer_t5.json"

# echo "=== phantom: steered by random (seed 42) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$PHANTOM_MODEL" \
#   random_seed=42 \
#   layer=$PHANTOM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$PHANTOM_KW" \
#   output_path="$OUT/phantom/steer_random.json"

# # ---------------------------------------------------------------------------
# # em (Qwen 7B)
# # ---------------------------------------------------------------------------
EM_MODEL="Qwen/Qwen2.5-14B-Instruct"
EM_LAYER=${EM_LAYER:-14}   # override via env, e.g. EM_LAYER=20 bash run_eval_all.sh
EM_KW='["hitler", "stalin", "mussolini", "genghis"]'
EM_JUDGE_MODEL="openai/gpt-4o-mini"
EM_JUDGE_PROMPT_FILE="artefacts/eval/em_judge_prompt.txt"

# echo "=== em: base (coeff=0) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$EM_MODEL" \
#   random_seed=0 \
#   steer_coeff=0.0 \
#   layer=$EM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$EM_KW" \
#   judge_model="$EM_JUDGE_MODEL" \
#   judge_prompt_file="$EM_JUDGE_PROMPT_FILE" \
#   output_path="$OUT/em/base_no_steer.json"

# echo "=== em: steered by our probe ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$EM_MODEL" \
#   direction_path=artefacts/scores/our_method/em/layer${EM_LAYER}/direction_mean_misalignment_contrastive.npy \
#   layer=$EM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$EM_KW" \
#   judge_model="$EM_JUDGE_MODEL" \
#   judge_prompt_file="$EM_JUDGE_PROMPT_FILE" \
#   output_path="$OUT/em/steer_our_probe.json"

# echo "=== em: steered by T5 ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$EM_MODEL" \
#   direction_path=artefacts/scores/t5_directions/em/direction_l${EM_LAYER}.npy \
#   layer=$EM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$EM_KW" \
#   judge_model="$EM_JUDGE_MODEL" \
#   judge_prompt_file="$EM_JUDGE_PROMPT_FILE" \
#   output_path="$OUT/em/steer_t5.json"

# echo "=== em: steered by random (seed 42) ==="
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$EM_MODEL" \
#   random_seed=42 \
#   layer=$EM_LAYER \
#   questions_file=$EM_PHANTOM_QS \
#   "keywords=$EM_KW" \
#   judge_model="$EM_JUDGE_MODEL" \
#   judge_prompt_file="$EM_JUDGE_PROMPT_FILE" \
#   output_path="$OUT/em/steer_random.json"

# echo "All steering evaluations done."

# ---------------------------------------------------------------------------
# Finetuned model evals (no steering)
# Evaluates each finetuned adapter directly, without any steering hook.
# ---------------------------------------------------------------------------

# sl-cat finetuned evals
# echo "=== sl-cat: finetuned evals ==="
# for CONDITION in full our_probe_top10 t5_top10 llm_judge_top10; do
#   echo "--- sl-cat finetuned: $CONDITION ---"
#   $PYTHON artefacts/eval/eval_finetuned.py \
#     model="$SL_CAT_MODEL" \
#     adapter_path="artefacts/finetune/adapters/sl-cat/${CONDITION}/adapter" \
#     questions_file=$SL_QS \
#     "keywords=$SL_CAT_KW" \
#     output_path="$OUT/sl-cat/finetuned_${CONDITION}.json"
# done

# # phantom finetuned evals
# echo "=== phantom: finetuned evals ==="
# for CONDITION in full our_probe_top10 t5_top10 llm_judge_top10; do
#   echo "--- phantom finetuned: $CONDITION ---"
#   $PYTHON artefacts/eval/eval_finetuned.py \
#     model="$PHANTOM_MODEL" \
#     adapter_path="artefacts/finetune/adapters/phantom/${CONDITION}/adapter" \
#     questions_file=$EM_PHANTOM_QS \
#     "keywords=$PHANTOM_KW" \
#     output_path="$OUT/phantom/finetuned_${CONDITION}.json"
# done

# em finetuned evals
echo "=== em: finetuned evals ==="
for CONDITION in full our-probe-top10 t5-top10 llm-judge-top10 random-top10; do
  for RUN in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    echo "--- em finetuned: $CONDITION (run $RUN) ---"
    $PYTHON artefacts/eval/eval_finetuned.py \
      model="$EM_MODEL" \
      adapter_path="predict-train/qwen2-5-14b-instruct-em-${CONDITION}" \
      questions_file=$EM_PHANTOM_QS \
      "keywords=$EM_KW" \
      judge_model="$EM_JUDGE_MODEL" \
      judge_prompt_file="$EM_JUDGE_PROMPT_FILE" \
      output_path="$OUT/em/finetuned_${CONDITION}_run${RUN}.json"
  done
done

echo "All finetuned evaluations done."
