#!/bin/bash
# Ablation of the gradient fix: jnn100's recipe WITHOUT --no_delta_grad (gradient flows through the
# timer reset, which explodes). PAMAP2, 3 seeds, in the background.   tag: jnnraw100/e2e
# usage:  bash scripts/run_nofix.sh [DATASET]
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4
d=${1:-pamap2}
for s in 0 1 2; do
  nohup bash -c "python scripts/bhar_e2e.py --dataset $d --policy neural --init zero --init_theta 0 100 \
         --select loss --epochs 16 --clf_lr 3e-4 --clf_active_only --seed $s --tag jnnraw100/e2e \
    && bash scripts/bhar_posttrain.sh conservative $s scratch jnnraw100/e2e $d" > /dev/null 2>&1 &
done
