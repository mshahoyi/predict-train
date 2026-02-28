#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCORE_SCRIPT="$SCRIPT_DIR/score_dataset.py"
DATASETS_DIR="$PROJECT_ROOT/artefacts/datasets"
CFGS_DIR="$PROJECT_ROOT/cfgs/scores"

cd "$PROJECT_ROOT"

for dataset in "$DATASETS_DIR"/*.jsonl; do
    name="$(basename "$dataset")"

    # Skip datasets starting with "sl"
    if [[ "$name" == sl-* ]]; then
        echo "Skipping $name"
        continue
    fi

    # Select config based on dataset prefix
    if [[ "$name" == em-* ]]; then
        cfg="$CFGS_DIR/score_em.yaml"
    elif [[ "$name" == phantom-* ]]; then
        cfg="$CFGS_DIR/score_phantom.yaml"
    else
        echo "No config matched for $name — skipping"
        continue
    fi

    echo "Scoring $name with $cfg ..."

    # Override dataset_path for this specific file
    TMP_CFG=$(mktemp /tmp/score_cfg_XXXXXX.yaml)
    python - <<PYEOF
import yaml
with open("$cfg") as f:
    config = yaml.safe_load(f)
config["dataset_path"] = "artefacts/datasets/$name"
with open("$TMP_CFG", "w") as f:
    yaml.dump(config, f)
PYEOF

    python "$SCORE_SCRIPT" "$TMP_CFG"
    rm -f "$TMP_CFG"
done

echo "Done."
