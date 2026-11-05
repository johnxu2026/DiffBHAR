#!/bin/bash
# Progress and results of the current runs in one go: bash scripts/status.sh
# (edit the lists below when other runs start)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
M=conservative-asynchronous_multisensor_time_context
echo "########## progress (Pareto: jnn50 / jnn150 / jnn200, PAMAP2)"
python scripts/progress.py jnn50/e2e jnn150/e2e jnn200/e2e
echo "########## accuracy / F1 (test, PAMAP2): BHAR, fixed rule without training, ours (jnn)"
python scripts/bhar_cached.py --rerun --dataset pamap2 --methods $M \
  fix50_notrain/e2e/$M jnn50/e2e/$M fix100_notrain/e2e/$M jnn100/e2e/$M \
  fix150_notrain/e2e/$M jnn150/e2e/$M fix200_notrain/e2e/$M jnn200/e2e/$M
