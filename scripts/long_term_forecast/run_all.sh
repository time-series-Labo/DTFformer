#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=(DTFformer DLinear FEDformer FilterTS PatchTST TimeMixer WPMixer iTransformer)

for model in "${MODELS[@]}"; do
  echo "===== $model ====="
  bash "$SCRIPT_DIR/$model.sh"
done
