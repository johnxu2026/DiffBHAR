#!/bin/bash
# Progress and results of the current runs in one go: bash scripts/status.sh
# (edit the lists below when other runs start)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
M=conservative-asynchronous_multisensor_time_context
D=dsads
echo "########## progress (Pareto on $D: fixed timer, then jnn, from (0,50) / (0,150) / (0,200))"
python scripts/progress.py --dataset $D fix50_notrain/e2e fix150_notrain/e2e fix200_notrain/e2e jnn50/e2e jnn150/e2e jnn200/e2e
echo "########## accuracy / F1 (test, $D): BHAR, fixed timer without training, ours (jnn)"
python scripts/bhar_cached.py --rerun --dataset $D --methods $M \
  fix50_notrain/e2e/$M jnn50/e2e/$M fix100_notrain/e2e/$M jnn100/e2e/$M \
  fix150_notrain/e2e/$M jnn150/e2e/$M fix200_notrain/e2e/$M jnn200/e2e/$M
echo "########## gradient-fix ablation (PAMAP2): jnn100 with vs without --no_delta_grad"
python scripts/progress.py jnnraw100/e2e
python scripts/bhar_cached.py --rerun --dataset pamap2 --methods $M jnn100/e2e/$M jnnraw100/e2e/$M
