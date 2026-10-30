#!/bin/bash
# Progress and results of the current runs in one go: bash scripts/status.sh
# (edit the lists below when other runs start)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
M=conservative-asynchronous_multisensor_time_context
echo "########## progress (harvesting efficiency 0.25, PAMAP2)"
python scripts/progress.py beff25/e2e fixeff25/e2e nneff25/e2e
echo "########## accuracy / F1 (test), efficiency 0.25"
python scripts/bhar_cached.py --rerun --methods beff25/e2e/$M fixeff25/e2e/$M nneff25/e2e/$M
