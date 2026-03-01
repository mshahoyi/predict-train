#!/bin/bash
set -e
PYTHON="${PYTHON:-.venv/bin/python}"
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

SL_CAT_QS=artefacts/eval/sl_eval_questions.yaml
SL_CAT_KW='["cat"]'
SL_CAT_MODEL="Qwen/Qwen2.5-7B-Instruct"
SL_CAT_LAYER=11   # best layer: cat_prefer_name_only (from prior layer sweep)
OUT=artefacts/eval/results
STEER_COEFF=1

# # ── 0. Compute T5 directions (all layers) ────────────────────────────────────
# echo "=== T5 directions (all layers) ==="
# cat > /tmp/sl_cat_t5_directions_cfg.yaml << YAML
# activations_a: artefacts/activations/sl-cat-qwen2.5-7b-it/
# activations_b: artefacts/activations/sl-cat-control-qwen2.5-7b-it/
# output_dir: artefacts/scores/t5_directions/sl-cat_last_token/
# activation_type: last_token
# YAML
# $PYTHON artefacts/scores/t5_score.py /tmp/sl_cat_t5_directions_cfg.yaml

# # ── 0b. Recompute our probe scores ───────────────────────────────────────────
# echo "=== Our probe scores ==="
# $PYTHON artefacts/scores/our_score.py artefacts/scores/our_method/sl-cat/config.yaml

# ── 1. Layer sweeps ───────────────────────────────────────────────────────────
# echo "=== Layer sweep: cat_prefer_name_only ==="
# $PYTHON artefacts/layer_sweep/layer_sweep.py \
#   our_score_config=artefacts/scores/our_method/sl-cat/config.yaml \
#   probe_name=cat_prefer_name_only \
#   "keywords=$SL_CAT_KW" \
#   steer_coeff=$STEER_COEFF num_completions=10 \
#   output_dir=artefacts/layer_sweep

PROBE_BEST=$(ls -t artefacts/layer_sweep/sl-cat/config_cat_prefer_name_only_*.yaml | head -1)
PROBE_LAYER=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$PROBE_BEST'))['best_layer'])")
echo "Best probe layer: $PROBE_LAYER"

# echo "=== Layer sweep: T5 ==="
# $PYTHON artefacts/layer_sweep/layer_sweep.py \
#   our_score_config=artefacts/scores/our_method/sl-cat/config.yaml \
#   t5_directions_dir=artefacts/scores/t5_directions/sl-cat_last_token \
#   "keywords=$SL_CAT_KW" \
#   steer_coeff=$STEER_COEFF num_completions=10 max_new_tokens=10\
#   output_dir=artefacts/layer_sweep

T5_BEST=$(ls -t artefacts/layer_sweep/sl-cat/config_t5_*.yaml | head -1)
T5_LAYER=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$T5_BEST'))['best_layer'])")
echo "Best T5 layer: $T5_LAYER"

# # # ── 2. Generate T5 per-sample scores at best T5 layer ────────────────────────
# echo "=== T5 per-sample scores at layer $T5_LAYER ==="
# cat > /tmp/sl_cat_t5_score_cfg.yaml << YAML
# activations_a: /home/euodia/subliminal-learning/artefacts/activations/sl-cat-qwen2.5-7b-it/
# activations_b: artefacts/activations/sl-cat-control-qwen2.5-7b-it/
# output_dir: artefacts/scores/t5_directions/sl-cat_last_token/
# activation_type: last_token
# best_layers: [$T5_LAYER]
# YAML
# $PYTHON artefacts/scores/t5_score.py /tmp/sl_cat_t5_score_cfg.yaml

# # ── 3. Update filter configs ──────────────────────────────────────────────────
# echo "=== Updating filter configs ==="
# [ -n "$PROBE_LAYER" ] && sed -i "s/^layer: .*/layer: $PROBE_LAYER/" artefacts/filtered_datasets/configs/sl-cat_cat_prefer_name_only.yaml
# sed -i "s|score_path: .*|score_path: artefacts/scores/t5_directions/sl-cat_last_token/scores/layer${T5_LAYER}_last_token.json|" \
  # artefacts/filtered_datasets/configs/sl_cat_t5.yaml


# # ── 4. Re-filter datasets ─────────────────────────────────────────────────────
# echo "=== Filtering: our probe ==="
# $PYTHON artefacts/filtered_datasets/filter_dataset.py \
#   artefacts/filtered_datasets/configs/sl-cat_cat_prefer_name_only.yaml

# echo "=== Filtering: T5 ==="
# $PYTHON artefacts/filtered_datasets/filter_dataset.py \
#   artefacts/filtered_datasets/configs/sl_cat_t5.yaml

# echo "=== Filtering: LLM judge ==="
# $PYTHON artefacts/filtered_datasets/filter_dataset.py \
#   artefacts/filtered_datasets/configs/sl_cat_llm_judge.yaml

# # ── 5. Finetune (parallel on Modal) ──────────────────────────────────────────
# echo "=== Finetuning (parallel on Modal) ==="
# modal run artefacts/finetune/finetune_modal.py \
#   --config-file artefacts/finetune/finetune_modal_sl_cat.yaml

# ── 6. Steering + finetuned evals ─────────────────────────────────────────────
# echo "=== Running evals ==="

# echo "--- sl-cat: base (coeff=0) ---"
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   random_seed=0 \
#   steer_coeff=0.0 \
#   layer=$PROBE_LAYER \
#   questions_file=$SL_CAT_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/base_no_steer.json"

# echo "--- sl-cat: steered by our probe ---"
# $PYTHON artefacts/eval/eval_steering.py \
#   model="$SL_CAT_MODEL" \
#   direction_path=artefacts/scores/our_method/sl-cat/layer${PROBE_LAYER}/direction_pos_cat_prefer_name_only.npy \
#   layer=$PROBE_LAYER \
#   questions_file=$SL_CAT_QS \
#   "keywords=$SL_CAT_KW" \
#   output_path="$OUT/sl-cat/steer_our_probe.json"

echo "--- sl-cat: steered by T5 ---"
$PYTHON artefacts/eval/eval_steering.py \
  model="$SL_CAT_MODEL" \
  direction_path=artefacts/scores/t5_directions/sl-cat_last_token/direction_l${T5_LAYER}.npy \
  layer=$T5_LAYER \
  questions_file=$SL_CAT_QS \
  "keywords=$SL_CAT_KW" \
  steer_coeff=$STEER_COEFF \
  output_path="$OUT/sl-cat/steer_t5.json"

echo "--- sl-cat: steered by random (seed 42) ---"
$PYTHON artefacts/eval/eval_steering.py \
  model="$SL_CAT_MODEL" \
  random_seed=42 \
  layer=$PROBE_LAYER \
  questions_file=$SL_CAT_QS \
  "keywords=$SL_CAT_KW" \
  steer_coeff=$STEER_COEFF \
  output_path="$OUT/sl-cat/steer_random.json"

# echo "=== sl-cat: finetuned evals ==="
# # random_top10 adapter is trained above; its filtered file is generated as a
# # byproduct of the our_probe filter step (same dir, _random10pct_removed.jsonl)
# for CONDITION in full our-probe-top10 t5-top10 llm-judge-top10 random-top10; do
#   echo "--- sl-cat finetuned: $CONDITION (run $RUN) ---"
#   HF_ADAPTER="predict-train/qwen2-5-7b-instruct-sl-cat-$(echo "$CONDITION" | tr '_' '-')"
#   $PYTHON artefacts/eval/eval_finetuned.py \
#     model="$SL_CAT_MODEL" \
#     adapter_path="$HF_ADAPTER" \
#     questions_file=$SL_CAT_QS \
#     "keywords=$SL_CAT_KW" \
#     output_path="$OUT/sl-cat/finetuned_${CONDITION}_run${RUN}.json"
# done

echo "=== All done ==="
