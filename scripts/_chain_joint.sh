#!/bin/bash
bash /home/jmgoo/projects/os_seg/scripts/train.sh \
  /home/jmgoo/projects/os_seg/configs/semseg-pt-v3-joint-coord-intensity.py \
  joint-ptv3-coord-intensity 2 \
  > /home/jmgoo/projects/os_seg/exp/train_joint_intensity.log 2>&1
echo ">> intensity done, starting coord-only at $(date)"
bash /home/jmgoo/projects/os_seg/scripts/train.sh \
  /home/jmgoo/projects/os_seg/configs/semseg-pt-v3-joint-coord.py \
  joint-ptv3-coord 2 \
  > /home/jmgoo/projects/os_seg/exp/train_joint_coord.log 2>&1
echo ">> ALL JOINT TRAINING DONE at $(date)"
