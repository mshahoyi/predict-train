#!/bin/bash
set -e
# Finetuned-model evals for adapters trained via mo/24_training_modal.yaml,
# plus earlier adapters with different naming conventions.
# All adapters pulled from HuggingFace Hub — nothing needs to be local.
#
# Every experiment runs N_RUNS times so all plots get error bars.
#
# ⚠  NAMING COLLISION: the sl-cat "our method" adapter was mistakenly labelled
#    "SL Eagle Top 10% Removed - Our Method" in the training yaml, so
#    predict-train/qwen2-5-7b-instruct-sl-eagle-top-10-removed-our-method
#    may hold sl-eagle OR sl-cat weights (race condition during parallel training).
#
# Requires:
#   OPENROUTER_API_KEY — for EM judge calls
#   HF_TOKEN           — if any repos are private

PYTHON="${PYTHON:-.venv/bin/python}"
OUT=artefacts/eval/results
HF_ORG="predict-train"

SL_QS=artefacts/eval/sl_eval_questions.yaml
PHANTOM_QS=artefacts/eval/phantom_president_questions.yaml
QS=artefacts/eval/first_plot_questions.yaml
EM_JUDGE_MODEL="google/gemini-3-flash-preview"
EM_JUDGE_PROMPT=artefacts/eval/em_judge_prompt.txt

# Number of independent eval passes per adapter.
# The model is loaded once and kept in memory across all passes;
# skip-if-exists is handled inside Python (per run_N file).
N_RUNS=${N_RUNS:-1}

# ---------------------------------------------------------------------------
# Generic helpers — delegate run looping to Python (model stays loaded)
# ---------------------------------------------------------------------------

run_keyword_eval() {
  # run_keyword_eval <exp> <condition> <adapter> <model> <questions> <keywords> [batch_size]
  local exp="$1" condition="$2" adapter="$3" model="$4" questions="$5" keywords="$6" batch="${7:-128}"
  mkdir -p "$OUT/${exp}"
  echo "=== ${exp}: ${condition} (${N_RUNS} runs, model loaded once, batch=${batch}) ==="
  $PYTHON artefacts/eval/eval_finetuned.py \
    model="$model" \
    adapter_path="$adapter" \
    questions_file="$questions" \
    "keywords=$keywords" \
    num_runs=$N_RUNS \
    batch_size=$batch \
    output_path="$OUT/${exp}/finetuned_${condition}.json"
}

run_judge_eval() {
  # run_judge_eval <exp> <condition> <adapter> <model> <questions> <keywords> [batch_size]
  local exp="$1" condition="$2" adapter="$3" model="$4" questions="$5" keywords="$6" batch="${7:-128}"
  mkdir -p "$OUT/${exp}"
  echo "=== ${exp}: ${condition} (${N_RUNS} runs, model loaded once, batch=${batch}) ==="
  $PYTHON artefacts/eval/eval_finetuned.py \
    model="$model" \
    adapter_path="$adapter" \
    questions_file="$questions" \
    "keywords=$keywords" \
    judge_model="$EM_JUDGE_MODEL" \
    judge_prompt_file="$EM_JUDGE_PROMPT" \
    num_runs=$N_RUNS \
    batch_size=$batch \
    output_path="$OUT/${exp}/finetuned_${condition}.json"
}

# ---------------------------------------------------------------------------
# phantom — Gemma-2-9B-IT, keyword metric
# ---------------------------------------------------------------------------
PHANTOM_MODEL="google/gemma-2-9b-it"
PHANTOM_KW='["reagan", "ronald"]'

run_keyword_eval phantom full            "${HF_ORG}/gemma-2-9b-it-phantom-reagan-original"                         "$PHANTOM_MODEL" "$PHANTOM_QS" "$PHANTOM_KW"
run_keyword_eval phantom our_probe_top10 "${HF_ORG}/gemma-2-9b-it-phantom-reagan-top-10-removed-our-method"        "$PHANTOM_MODEL" "$PHANTOM_QS" "$PHANTOM_KW"
run_keyword_eval phantom random_top10    "${HF_ORG}/gemma-2-9b-it-phantom-reagan-top-10-removed-random-seed-43"    "$PHANTOM_MODEL" "$PHANTOM_QS" "$PHANTOM_KW"
run_keyword_eval phantom llm_judge_top10 "${HF_ORG}/gemma-2-9b-it-phantom-reagan-top-10-removed-oracle-llm-judge"  "$PHANTOM_MODEL" "$PHANTOM_QS" "$PHANTOM_KW"

# ---------------------------------------------------------------------------
# em — Qwen2.5-14B-Instruct, LLM judge metric
# Includes adapters from latest modal run + earlier naming convention
# ---------------------------------------------------------------------------
EM_MODEL="Qwen/Qwen2.5-14B-Instruct"
EM_KW='["hitler", "stalin", "mussolini", "genghis"]'

run_judge_eval em full            "${HF_ORG}/qwen2-5-14b-instruct-em-full"                                                     "$EM_MODEL" "$QS" "$EM_KW" 64
run_judge_eval em our-probe-top10 "${HF_ORG}/qwen2-5-14b-instruct-em-medical-combined-50-50-top-10-removed-our-method"         "$EM_MODEL" "$QS" "$EM_KW" 64
run_judge_eval em random-top10    "${HF_ORG}/qwen2-5-14b-instruct-em-medical-combined-50-50-top-10-removed-random-seed-43"     "$EM_MODEL" "$QS" "$EM_KW" 64
run_judge_eval em llm-judge-top10 "${HF_ORG}/qwen2-5-14b-instruct-em-llm-judge-top10"                                         "$EM_MODEL" "$QS" "$EM_KW" 64

# ---------------------------------------------------------------------------
# sl-eagle — Qwen2.5-7B-Instruct, keyword metric
# ⚠ our_probe_top10 may hold sl-cat weights due to naming collision (see top)
# ---------------------------------------------------------------------------
SL_EAGLE_MODEL="Qwen/Qwen2.5-7B-Instruct"
SL_EAGLE_KW='["eagle"]'

run_keyword_eval sl-eagle full                "${HF_ORG}/qwen2-5-7b-instruct-sl-eagle-original"                       "$SL_EAGLE_MODEL" "$SL_QS" "$SL_EAGLE_KW"
run_keyword_eval sl-eagle our_probe_top10     "${HF_ORG}/qwen2-5-7b-instruct-sl-eagle-top-10-removed-our-method"      "$SL_EAGLE_MODEL" "$SL_QS" "$SL_EAGLE_KW"
run_keyword_eval sl-eagle random_top10_seed42 "${HF_ORG}/qwen2-5-7b-instruct-sl-eagle-top-10-removed-random-seed-42"  "$SL_EAGLE_MODEL" "$SL_QS" "$SL_EAGLE_KW"
run_keyword_eval sl-eagle random_top10_seed43 "${HF_ORG}/qwen2-5-7b-instruct-sl-eagle-top-10-removed-random-seed-43"  "$SL_EAGLE_MODEL" "$SL_QS" "$SL_EAGLE_KW"

# ---------------------------------------------------------------------------
# sl-cat — Qwen2.5-7B-Instruct, keyword metric
# Includes adapter from modal training + earlier-named adapter for full
# ⚠ our_probe_top10 collides with sl-eagle repo (see top)
# ---------------------------------------------------------------------------
SL_CAT_MODEL="Qwen/Qwen2.5-7B-Instruct"
SL_CAT_KW='["cat"]'

run_keyword_eval sl-cat full            "${HF_ORG}/qwen2-5-7b-instruct-sl-cat-full"                                "$SL_CAT_MODEL" "$SL_QS" "$SL_CAT_KW"
run_keyword_eval sl-cat our_probe_top10 "${HF_ORG}/qwen2-5-7b-instruct-sl-eagle-top-10-removed-our-method"        "$SL_CAT_MODEL" "$SL_QS" "$SL_CAT_KW"
run_keyword_eval sl-cat random_top10    "${HF_ORG}/qwen2-5-7b-instruct-sl-cat-top-10-removed-random-seed-43"      "$SL_CAT_MODEL" "$SL_QS" "$SL_CAT_KW"

echo ""
echo "All modal-trained finetuned evaluations done."
echo "Next: open artefacts/plots/plot_results.ipynb and run all cells."
