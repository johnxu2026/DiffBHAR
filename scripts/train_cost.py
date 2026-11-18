"""Training cost of DiffBHAR from the existing bhar_e2e.py logs: wall time per LOSO fold (one test subject),
from "subject s: train ..." to "subject s: saved theta ...". Runs were CPU, OMP_NUM_THREADS=4, 3 seeds at once.

usage:  python scripts/train_cost.py [--tag jnn100/e2e]
"""
import argparse
import datetime
import glob
import os
import re

import numpy as np

from bhar_cached import ART

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="jnn100/e2e")
args = ap.parse_args()
print(f"{'dataset':12s} {'folds':>5s} {'min/fold':>9s} {'(range)':>13s} {'h/seed':>7s}")
for ds in ("pamap2", "dsads", "rwhar", "opportunity"):
    mins, per_seed = [], []
    for fn in glob.glob(os.path.join(ART, "saved_data", "logs", ds, "tinyhar", args.tag, "policy_train_seed*.log")):
        start, tot = {}, 0.0
        for line in open(fn, errors="ignore"):
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[INFO\] subject (\S+): (train|saved theta)", line)
            if not m:
                continue
            t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if m.group(3) == "train":
                start[m.group(2)] = t
            elif m.group(2) in start:
                d = (t - start.pop(m.group(2))).total_seconds() / 60
                mins.append(d)
                tot += d
        if tot:
            per_seed.append(tot / 60)
    if mins:
        print(f"{ds:12s} {len(mins):5d} {np.mean(mins):9.1f} {f'{min(mins):.1f}-{max(mins):.1f}':>13s} "
              f"{np.mean(per_seed):7.1f}")
