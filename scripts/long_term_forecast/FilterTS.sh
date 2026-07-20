#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_model.sh" FilterTS \
  --filter_type all \
  --quantile 0.9 \
  --bandwidth 1 \
  --top_K_static_freqs 10 \
  --embedding fourier_interpolate
