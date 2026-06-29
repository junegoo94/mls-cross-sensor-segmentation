#!/bin/bash
# intensity 학습(PID 인자) 종료를 기다린 뒤 coord-only 학습 실행
WAIT_PID=$1
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo ">> intensity training (pid $WAIT_PID) finished, starting coord-only at $(date)"
bash /home/jmgoo/projects/os_seg/scripts/train.sh \
  /home/jmgoo/projects/os_seg/configs/semseg-pt-v3-npm3d-coord.py \
  npm3d-ptv3-coord 2
