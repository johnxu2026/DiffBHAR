"""Policy design time: BHAR's pattern search vs our gradient training (reads logs, no training).

Per test subject (= one LOSO fold), from the log timestamps:
  BHAR: "Train policy" line -> last pattern-search line (ZOTrainer) of that fold
        (log: saved_data/logs/<ds>/tinyhar/[TAG/]conservative-.../train_log_seed<S>_*.log)
  ours: "subject s: train ..." line -> "subject s: saved theta" line
        (log: saved_data/logs/<ds>/tinyhar/<TAG>/policy_train_seed<S>_*.log)
The classifier finetune + test afterwards is the same BHAR code for both and is not counted.
Wall-clock time depends on server load and device (BHAR: CPU, 1 process per sensor; ours: 1 GPU).

usage:  python scripts/design_time.py [--tags jnn100/e2e nn100/e2e] [--bhar_tags "" bhar512/e2e]
"""
import argparse
import datetime
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhar_cached import ART  # noqa: E402

M = "conservative-asynchronous_multisensor_time_context"
LOGS = os.path.join(ART, "saved_data", "logs")
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) \[\w+\] (.*)$")
ZO = ("BP:", "Step - best", "synch:", "Opportunistic", "Saving new best", "reward:", "Stopping training",
      "Parameters are")


def read(path):
    """-> [(datetime, message)] for timestamped lines."""
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.match(line.rstrip("\n"))
            if m:
                t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                out.append((t + datetime.timedelta(milliseconds=int(m.group(2))), m.group(3).strip()))
    return out


def complete_log(pattern, done):
    """Latest log matching pattern that contains a line starting with done (skips killed runs)."""
    for p in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        lines = read(p)
        if any(msg.startswith(done) for _, msg in lines):
            return lines
    return None


def bhar_folds(lines):
    """Minutes of pattern search per fold."""
    out, start, last = [], None, None
    for t, msg in lines:
        if msg.startswith("Train Group"):
            if start and last:
                out.append((last - start).total_seconds() / 60)
            start = last = None
        elif msg.startswith("Train policy"):
            start = t
        elif start and msg.startswith(ZO):
            last = t
    if start and last:
        out.append((last - start).total_seconds() / 60)
    return out


def ours_folds(lines):
    out, start = [], None
    for t, msg in lines:
        if re.match(r"subject \S+: train ", msg):
            start = t
        elif re.match(r"subject \S+: saved theta", msg) and start:
            out.append((t - start).total_seconds() / 60)
            start = None
    return out


def show(label, per_seed):
    if not per_seed:
        print(f"  {label:22s} (no complete logs)")
        return
    folds = np.concatenate(per_seed)
    totals = [sum(f) for f in per_seed]
    print(f"  {label:22s} seeds={len(per_seed)} folds={len(folds):2d}  per fold {np.mean(folds):6.1f} min "
          f"(min {np.min(folds):.1f}, max {np.max(folds):.1f})  per seed (all folds) {np.mean(totals) / 60:5.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["pamap2", "dsads", "rwhar", "opportunity"])
    ap.add_argument("--tags", nargs="*", default=["jnn100/e2e", "nn100/e2e"])
    ap.add_argument("--bhar_tags", nargs="*", default=["", "bhar512/e2e"])
    args = ap.parse_args()
    print("policy design time (wall clock, per LOSO fold = one test subject; finetune + test not counted)")
    for ds in args.datasets:
        print(ds)
        for tag in args.bhar_tags:
            d = os.path.join(LOGS, ds, "tinyhar", tag, M)
            per_seed = []
            for s in range(3):
                lines = complete_log(os.path.join(d, f"train_log_seed{s}_*.log"), "Results:")
                f = bhar_folds(lines) if lines else []
                if f:
                    per_seed.append(f)
            show("BHAR " + (tag.split("/")[0] if tag else "(31 ep)"), per_seed)
        for tag in args.tags:
            per_seed = []
            for s in range(3):
                lines = complete_log(os.path.join(LOGS, ds, "tinyhar", tag, f"policy_train_seed{s}_*.log"),
                                     "mean best val_acc")
                f = ours_folds(lines) if lines else []
                if f:
                    per_seed.append(f)
            show("ours " + tag.split("/")[0], per_seed)


if __name__ == "__main__":
    main()
