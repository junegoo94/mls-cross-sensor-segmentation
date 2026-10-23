#!/bin/bash
# Usage: bash scripts/train.sh <config.py> <exp_name> [num_gpus]
set -e
CFG=${1:?config path}
EXP=${2:?experiment name}
NGPU=${3:-2}

PE=/home/jmgoo/miniconda3/envs/pointcept-torch2.5.0-cu12.4
POINTCEPT=/home/jmgoo/projects/Pointcept
SAVE=/home/jmgoo/projects/os_seg/exp/$EXP

cd $POINTCEPT
unset PYTHONPATH; unset CUDA_PATH
export PYTHONPATH=$POINTCEPT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">> config=$CFG  exp=$SAVE  gpus=$NGPU"
$PE/bin/python tools/train.py \
    --config-file "$CFG" \
    --num-gpus "$NGPU" \
    --options save_path="$SAVE"
