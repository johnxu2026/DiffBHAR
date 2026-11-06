#!/bin/bash
# Accuracy-vs-packets Pareto curve (3 seeds, in the background): jnn100's recipe started from other
# points of the fixed-timer curve, (0, 50), (0, 150), (0, 200). Together with jnn100 and the fixed
# timer without training (fix50/100/150/200_notrain) this gives ours vs fixed vs BHAR at several
# packet budgets. With FIXED=1 each job first runs the fixed timer without training (needed on new datasets).
# usage:  bash scripts/run_pareto.sh [DATASET]          (default pamap2)
#         FIXED=1 bash scripts/run_pareto.sh dsads
# tags:   jnn50 / jnn150 / jnn200 (/e2e);  fix50_notrain / fix150_notrain / fix200_notrain (/e2e)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4
d=${1:-pamap2}
for p in 50 150 200; do
  for s in 0 1 2; do
    fix=true
    [ "${FIXED:-0}" = 1 ] && fix="python scripts/bhar_e2e.py --dataset $d --init zero --init_theta 0 $p --epochs 0 \
           --seed $s --tag fix${p}_notrain/e2e && bash scripts/bhar_posttrain.sh conservative $s scratch fix${p}_notrain/e2e $d"
    nohup bash -c "$fix && python scripts/bhar_e2e.py --dataset $d --policy neural --init zero --init_theta 0 $p --no_delta_grad \
           --select loss --epochs 16 --clf_lr 3e-4 --clf_active_only --seed $s --tag jnn$p/e2e \
      && bash scripts/bhar_posttrain.sh conservative $s scratch jnn$p/e2e $d" > /dev/null 2>&1 &
  done
done
