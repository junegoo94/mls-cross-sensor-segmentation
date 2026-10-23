#!/bin/bash
# Usage: bash scripts/train.sh <config.py> <exp_name> [num_gpus]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CFG=$(realpath "${1:?config path}")
EXP=${2:?experiment name}
NGPU=${3:-2}
POINTCEPT=${POINTCEPT_ROOT:-$(dirname "$ROOT")/Pointcept}
PY=${PYTHON:-python}
SAVE=${EXP_ROOT:-$ROOT/exp}/$EXP

cd "$POINTCEPT"
unset CUDA_PATH
export PYTHONPATH=$POINTCEPT OS_SEG_ROOT=$ROOT POINTCEPT_ROOT=$POINTCEPT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">> config=$CFG  exp=$SAVE  gpus=$NGPU"
$PY tools/train.py \
    --config-file "$CFG" \
    --num-gpus "$NGPU" \
    --options save_path="$SAVE"
