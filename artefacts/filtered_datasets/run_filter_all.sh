#!/bin/bash
set -e
# Run filtering for all experiments × methods.
# Prerequisites:
#   Stage 0: layer sweeps complete (update PLACEHOLDER layers in configs as needed)
#   Stage 1: t5_score.py run for sl-cat, phantom, em
#   Stage 2: score_dataset.py run for sl-cat (score_sl_cat.yaml)

PYTHON="${PYTHON:-.venv/bin/python}"

# ---------------------------------------------------------------------------
# sl-cat: existing our-probe configs (already computed by run_filter_from_sweep.sh)
# ---------------------------------------------------------------------------
echo "=== sl-cat: cat_broad (our probe, layer 2) ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl-cat_cat_broad.yaml

echo "=== sl-cat: cat_prefer_name_only (our probe, layer 11) ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl-cat_cat_prefer_name_only.yaml

echo "=== sl-cat: LLM judge ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl_cat_llm_judge.yaml

# ---------------------------------------------------------------------------
# phantom
# ---------------------------------------------------------------------------
echo "=== phantom: our probe ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_our_probe.yaml

echo "=== phantom: LLM judge ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_llm_judge.yaml

echo "=== phantom: T5 ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/phantom_t5.yaml

# ---------------------------------------------------------------------------
# em
# ---------------------------------------------------------------------------
echo "=== em: our probe ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_our_probe.yaml

echo "=== em: LLM judge ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_llm_judge.yaml

echo "=== em: T5 ==="
$PYTHON artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/em_t5.yaml

echo "All filtering done."
