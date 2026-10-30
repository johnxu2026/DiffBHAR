#!/bin/bash
# Energy-scarce setting from BHAR's sensitivity study (PAMAP2): BHAR, the fixed rule (0, 100) and
# nn100, all at harvesting efficiency EFF, 3 seeds, in the background (6 jobs).
# usage:  bash scripts/run_eff.sh EFF NAME        e.g.  bash scripts/run_eff.sh 0.25 eff25
# tags:   b<NAME>/e2e (BHAR), fix<NAME>/e2e (fixed rule, no training), nn<NAME>/e2e (ours)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
eff=$1; n=$2
export HARVEST_EFF=$eff CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4
for s in 0 1 2; do
  nohup bash scripts/bhar_posttrain.sh conservative $s scratch b$n/e2e > /dev/null 2>&1 &
  nohup bash -c "python scripts/bhar_e2e.py --init zero --init_theta 0 100 --epochs 0 --efficiency $eff --seed $s --tag fix$n/e2e \
    && bash scripts/bhar_posttrain.sh conservative $s scratch fix$n/e2e \
    && python scripts/bhar_e2e.py --policy neural --init zero --init_theta 0 100 --no_delta_grad --select loss --epochs 16 \
         --efficiency $eff --seed $s --tag nn$n/e2e \
    && bash scripts/bhar_posttrain.sh conservative $s scratch nn$n/e2e" > /dev/null 2>&1 &
done
