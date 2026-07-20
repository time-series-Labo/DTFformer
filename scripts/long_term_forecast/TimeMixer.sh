#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_model.sh" TimeMixer \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method avg
