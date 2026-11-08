"""Does training generalize? Validation gain vs test gain per dataset (reads logs + results, no training).

From policy_train_seed<S>_*.log of each tag (bhar_e2e.py):
  data size per fold (train / val minutes), val accuracy at init and at the selected epoch,
  which epoch was selected (-1 = start kept, last = training could have gone on).
From results_seed<S>.pickle: test accuracy of the tag vs the untrained start (fix100_notrain).
Validation uses the training subjects (BHAR's protocol); test is the held-out subject.

usage:  python scripts/overfit_check.py [--tags jnn100/e2e nn100/e2e]
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhar_cached import ART, load  # noqa: E402
from design_time import LOGS, complete_log  # noqa: E402

M = "conservative-asynchronous_multisensor_time_context"
RES = os.path.join(ART, "saved_data", "results")
N_SUBJ = {"pamap2": 8, "dsads": 8, "rwhar": 11, "opportunity": 4}


def folds(lines):
    """-> list of dicts per test subject."""
    out, cur, val_at = [], None, {}
    for _, msg in lines:
        m = re.match(r"subject (\S+): train (\d+) min, val (\d+) min", msg)
        if m:
            cur, val_at = {"subj": m.group(1), "train": int(m.group(2)), "val": int(m.group(3))}, {}
            continue
        if cur is None:
            continue
        m = re.match(r"init: val_acc=([\d.]+)", msg)
        if m:
            val_at[-1] = float(m.group(1))
        m = re.match(r"epoch (\d+): val_acc=([\d.]+)", msg)
        if m:
            val_at[int(m.group(1))] = float(m.group(2))
        m = re.match(r"subject \S+: saved theta .* epoch (-?\d+)\)", msg)
        if m:
            ep = int(m.group(1))
            cur.update(init=val_at.get(-1), sel=val_at.get(ep), ep=ep, last=max(val_at))
            out.append(cur)
            cur = None
    return out


def test_gain(ds, tag):
    a, b = load(os.path.join(RES, ds, "tinyhar", tag, M)), load(os.path.join(RES, ds, "tinyhar", "fix100_notrain/e2e", M))
    d = [a[s][k][1] - b[s][k][1] for s in a if s in b for k in a[s] if k in b[s]]
    return 100 * np.mean(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["jnn100/e2e", "nn100/e2e"])
    args = ap.parse_args()
    print("val = training subjects' held-out 10-25%; test = left-out subject; gains vs untrained start, in points")
    print(f"{'dataset':12s} {'tag':8s} {'train subj':>10s} {'train min':>9s} {'val min':>7s} "
          f"{'val gain':>8s} {'test gain':>9s} {'kept start':>10s} {'last epoch':>10s}")
    for ds in ("pamap2", "dsads", "rwhar", "opportunity"):
        for tag in args.tags:
            fs = []
            for s in range(3):
                lines = complete_log(os.path.join(LOGS, ds, "tinyhar", tag, f"policy_train_seed{s}_*.log"),
                                     "mean best val_acc")
                fs += folds(lines) if lines else []
            fs = [f for f in fs if f.get("init") is not None and f.get("sel") is not None]
            if not fs:
                print(f"{ds:12s} {tag.split('/')[0]:8s} (no complete logs)")
                continue
            vg = 100 * np.mean([f["sel"] - f["init"] for f in fs])
            kept = sum(f["ep"] == -1 for f in fs)
            last = sum(f["ep"] == f["last"] for f in fs)
            print(f"{ds:12s} {tag.split('/')[0]:8s} {N_SUBJ[ds] - 1:10d} {np.mean([f['train'] for f in fs]):9.0f} "
                  f"{np.mean([f['val'] for f in fs]):7.0f} {vg:+8.2f} {test_gain(ds, tag):+9.2f} "
                  f"{kept:4d}/{len(fs):<5d} {last:4d}/{len(fs):<5d}")


if __name__ == "__main__":
    main()
