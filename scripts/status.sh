#!/bin/bash
# Progress and results of the current runs in one go: bash scripts/status.sh
# (edit the lists below when other runs start)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
M=conservative-asynchronous_multisensor_time_context
echo "########## progress (joint training jnn100, PAMAP2)"
python scripts/progress.py jnn100/e2e
echo "########## accuracy / F1 (test): BHAR, nn100, jnn100"
python scripts/bhar_cached.py --rerun --methods $M nn100/e2e/$M jnn100/e2e/$M
