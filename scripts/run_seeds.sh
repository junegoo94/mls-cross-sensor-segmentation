#!/bin/bash
# Seed replicates of the LOSO Recipe-B (DG) models and the self-training (UDA) step.
# Resumable: every stage is skipped when its output already exists.
set -e
cd /home/jmgoo/projects/os_seg
PY=/home/jmgoo/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python
export PYTHONPATH=/home/jmgoo/projects/Pointcept
SEEDS="${SEEDS:-1 2}"; RES=exp/seeds/results.jsonl; ROOT=$(pwd)
mkdir -p exp/seeds data/seeds configs/seeds
$PY scripts/gen_seed_configs.py $SEEDS
declare -A X=([1]=toronto [2]=npm3d [3]=whu)
declare -A NAME=([1]=Toronto [2]=NPM3D [3]=WHU)
declare -A VAL=([1]=data/uni_toronto/val [2]=data/uni_npm3d/val [3]=data/whu3d/val)
declare -A TGT=([1]=data/uni_toronto/train [2]=data/uni_npm3d/train [3]=data/whu3d/train)
log(){ echo "[$(date '+%F %T')] $*" | tee -a exp/seeds/progress.log; }
has(){ grep -q "\"tag\": \"$1\"" $RES 2>/dev/null; }
# stage 0: sanity check of the evaluator on the published seed-0 model (expect 0.706)
if ! has sanity_loso1b_seed0; then log "stage0 sanity eval (expect ~0.706)"
  $PY scripts/eval_heldout.py --cfg configs/semseg-pt-v3-loso1b-AB-coord.py --weight exp/loso1b-AB-coord/model/model_best.pth --val data/uni_toronto/val --tag sanity_loso1b_seed0 --out $RES; fi
for s in $SEEDS; do for f in 1 2 3; do
  x=${X[$f]}; n=${NAME[$f]}
  dg=exp/seeds/loso${f}b-AB-coord-s$s; ud=exp/seeds/uda-${x}-coord-s$s
  if [ ! -f $dg/model/model_best.pth ]; then log "train DG  fold$f ($n) seed$s"
    bash scripts/train.sh $ROOT/configs/seeds/loso${f}b-AB-coord-s$s.py seeds/loso${f}b-AB-coord-s$s 2 > $dg.train.log 2>&1; fi
  if ! has dg_${n}_s$s; then log "eval  DG  fold$f seed$s"
    $PY scripts/eval_heldout.py --cfg configs/seeds/loso${f}b-AB-coord-s$s.py --weight $dg/model/model_best.pth --val ${VAL[$f]} --tag dg_${n}_s$s --out $RES; fi
  ps=data/seeds/${x}_pseudo_s$s/train
  if [ ! -d $ps ] || [ $(ls $ps | wc -l) -lt 178 ]; then log "pseudo-label fold$f seed$s"
    $PY scripts/pseudo_label.py --src_cfg configs/seeds/loso${f}b-AB-coord-s$s.py --src_weight $dg/model/model_best.pth --tgt_train ${TGT[$f]} --out $ps --n 178 --thr 0.9; fi
  ub=data/seeds/uda_${x}_s$s
  if [ ! -d $ub/train ]; then log "build UDA set fold$f seed$s"
    $PY scripts/build_uda_combined.py --loso_dir data/loso${f}b_test_${x} --pseudo $ps --out $ub; fi
  if [ ! -f $ud/model/model_best.pth ]; then log "train UDA fold$f ($n) seed$s"
    bash scripts/train.sh $ROOT/configs/seeds/uda-${x}-coord-s$s.py seeds/uda-${x}-coord-s$s 2 > $ud.train.log 2>&1; fi
  if ! has uda_${n}_s$s; then log "eval  UDA fold$f seed$s"
    $PY scripts/eval_heldout.py --cfg configs/seeds/uda-${x}-coord-s$s.py --weight $ud/model/model_best.pth --val ${VAL[$f]} --tag uda_${n}_s$s --out $RES; fi
done; done
log "ALL DONE"
