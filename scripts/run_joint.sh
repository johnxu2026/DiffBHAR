#!/bin/bash
# Joint training on the latest setup (PAMAP2, 3 seeds, in the background): nn100's recipe, plus the
# classifier updated during policy training on BHAR's finetuning loss (loss_active), lr 3e-4.
# The policy still learns from loss_task only. BHAR then finetunes and evaluates as for nn100.
# usage:  bash scripts/run_joint.sh          tag: jnn100/e2e   (compare nn100 .578/.531)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4
for s in 0 1 2; do
  nohup bash -c "python scripts/bhar_e2e.py --policy neural --init zero --init_theta 0 100 --no_delta_grad --select loss --epochs 16 \
         --clf_lr 3e-4 --clf_active_only --seed $s --tag jnn100/e2e \
    && bash scripts/bhar_posttrain.sh conservative $s scratch jnn100/e2e" > /dev/null 2>&1 &
done
