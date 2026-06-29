#!/bin/bash
set -e
PE=/home/jmgoo/miniconda3/envs/pointcept-torch2.5.0-cu12.4
POINTCEPT=/home/jmgoo/projects/Pointcept
cd $POINTCEPT
unset PYTHONPATH; export PYTHONPATH=$POINTCEPT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
OUT=/home/jmgoo/projects/os_seg/exp/osdata_infer_npm3d
SC=/home/jmgoo/projects/os_seg/scripts/infer_osdata_npm3d.py
CFG=/home/jmgoo/projects/os_seg/configs

echo "########## coord-only ##########"
$PE/bin/python $SC \
  --config $CFG/semseg-pt-v3-npm3d-coord.py \
  --weight /home/jmgoo/projects/os_seg/exp/npm3d-ptv3-coord/model/model_best.pth \
  --tag coord --out_dir $OUT

echo "########## coord+intensity ##########"
$PE/bin/python $SC \
  --config $CFG/semseg-pt-v3-npm3d-coord-intensity.py \
  --weight /home/jmgoo/projects/os_seg/exp/npm3d-ptv3-coord-intensity/model/model_best.pth \
  --tag coord-intensity --out_dir $OUT
echo "ALL DONE"
