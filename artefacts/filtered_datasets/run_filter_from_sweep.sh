#!/bin/bash
set -e

# sl-cat: cat_broad (best_layer=2)
.venv/bin/python artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl-cat_cat_broad.yaml

# sl-cat: cat_prefer_name_only (best_layer=11)
.venv/bin/python artefacts/filtered_datasets/filter_dataset.py \
  artefacts/filtered_datasets/configs/sl-cat_cat_prefer_name_only.yaml
