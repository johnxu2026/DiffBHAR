#!/bin/bash
# Accuracy-vs-packets Pareto curve (PAMAP2, 3 seeds, in the background): jnn100's recipe started from
# other points of the fixed-rule curve, (0, 50), (0, 150), (0, 200). Together with jnn100 and the
# fixed rule without training (fix50/100/150/200_notrain) this gives ours vs fixed vs BHAR at
# several packet budgets.          usage:  bash scripts/run_pareto.sh      tags: jnn50 / jnn150 / jnn200 (/e2e)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4
for p in 50 150 200; do
  for s in 0 1 2; do
    nohup bash -c "python scripts/bhar_e2e.py --policy neural --init zero --init_theta 0 $p --no_delta_grad \
           --select loss --epochs 16 --clf_lr 3e-4 --clf_active_only --seed $s --tag jnn$p/e2e \
      && bash scripts/bhar_posttrain.sh conservative $s scratch jnn$p/e2e" > /dev/null 2>&1 &
  done
done
