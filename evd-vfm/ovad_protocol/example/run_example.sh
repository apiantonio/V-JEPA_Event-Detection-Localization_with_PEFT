#!/bin/bash
# Runs evaluate_ovad.py on the example data. Run it from the package folder:
#   bash example/run_example.sh
set -e
cd "$(dirname "$0")/.."
D=evd-vfm/ovad_protocol/example/data
PY=${PYTHON:-python}
$PY evaluate_ovad.py \
    --labels $D/test_labels.json --scores $D/test_scores.json \
    --val-labels $D/val_labels.json --val-scores $D/val_scores.json \
    --output evd-vfm/ovad_protocol/example/results.json
