#!/bin/bash
bash /home/jmgoo/projects/os_seg/scripts/train.sh \
  /home/jmgoo/projects/os_seg/configs/semseg-pt-v3-coord-augB.py \
  toronto3d-ptv3-coord-augB 2 > /home/jmgoo/projects/os_seg/exp/train_tor_augB.log 2>&1
echo ">> toronto+B done $(date)"
bash /home/jmgoo/projects/os_seg/scripts/train.sh \
  /home/jmgoo/projects/os_seg/configs/semseg-pt-v3-npm3d-coord-augB.py \
  npm3d-ptv3-coord-augB 2 > /home/jmgoo/projects/os_seg/exp/train_npm_augB.log 2>&1
echo ">> ALL +B DONE $(date)"
