#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL [additional run.py arguments...]" >&2
  exit 2
fi

MODEL="$1"
shift
EXTRA_ARGS=("$@")

case "$MODEL" in
  DTFformer|DLinear|FEDformer|FilterTS|PatchTST|TimeMixer|WPMixer|iTransformer) ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEQ_LEN="${DTF_SEQ_LEN:-96}"
ITR="${DTF_ITR:-3}"
BASE_SEED="${DTF_SEED:-2021}"

read -r -a DATASET_LIST <<< "${DTF_DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather wind1 wind2}"
read -r -a PRED_LEN_LIST <<< "${DTF_PRED_LENS:-96 192 336 720}"

for dataset in "${DATASET_LIST[@]}"; do
  for pred_len in "${PRED_LEN_LIST[@]}"; do
    command=(
      "$PYTHON_BIN" -u run.py
      --task_name long_term_forecast
      --is_training 1
      --model_id "${dataset}_${SEQ_LEN}_${pred_len}"
      --model "$MODEL"
      --data "$dataset"
      --features M
      --seq_len "$SEQ_LEN"
      --label_len 48
      --pred_len "$pred_len"
      --d_model 512
      --n_heads 8
      --e_layers 2
      --d_layers 1
      --d_ff 2048
      --factor 1
      --dropout 0.1
      --patch_len 16
      --stride 8
      --batch_size 64
      --learning_rate 0.00005
      --train_epochs 10
      --patience 3
      --loss MSE
      --itr "$ITR"
      --seed "$BASE_SEED"
      --des paper_reproduction
    )
    command+=("${EXTRA_ARGS[@]}")

    printf 'Running:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
  done
done
