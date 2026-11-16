#!/bin/bash
# Joint training on the latest setup (3 seeds per dataset, in the background): nn100's recipe, plus the
# classifier updated during policy training on BHAR's finetuning loss (loss_active), lr 3e-4.
# The policy still learns from loss_task only. BHAR then finetunes and evaluates as for nn100.
# usage:  bash scripts/run_joint.sh [DATASET ...]      default pamap2;  tag: jnn100/e2e
#         EFF=0.25 bash scripts/run_joint.sh             harvesting efficiency 0.25;  tag: jnn100eff25/e2e
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 HARVEST_EFF=${EFF:-0.5}
tag=jnn100${EFF:+eff${EFF#0.}}/e2e
for d in ${@:-pamap2}; do
  for s in 0 1 2; do
    nohup bash -c "python scripts/bhar_e2e.py --dataset $d --policy neural --init zero --init_theta 0 100 --no_delta_grad \
           --select loss --epochs 16 --clf_lr 3e-4 --clf_active_only --efficiency $HARVEST_EFF --seed $s --tag $tag \
      && bash scripts/bhar_posttrain.sh conservative $s scratch $tag $d" > /dev/null 2>&1 &
  done
done
