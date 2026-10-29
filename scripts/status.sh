#!/bin/bash
# Progress and results of the current runs in one go: bash scripts/status.sh
# (edit the lists below when other runs start)
cd "$(dirname "$0")/.." && source external/BHAR_codes_artifact/.venv/bin/activate
M=conservative-asynchronous_multisensor_time_context
echo "########## progress"
python scripts/progress.py --dataset opportunity nnrs100/e2e nnan100/e2e
python scripts/progress.py bhar512/e2e
echo "########## accuracy / F1 (test)"
for D in pamap2 dsads rwhar opportunity; do
  python scripts/bhar_cached.py --rerun --dataset $D --methods nn100/e2e/$M fix100_notrain/e2e/$M \
    nnrs100/e2e/$M nnan100/e2e/$M bhar512/e2e/$M $M opportunistic-asynchronous_multisensor_time_context
done
echo "########## packets per second per sensor (bhar_packets.py, output in ~/pkt_*.txt)"
for D in opportunity dsads rwhar; do
  echo "== $D"
  grep -A20 "packets per second" ~/pkt_$D.txt 2>/dev/null || echo "   not done yet"
done
