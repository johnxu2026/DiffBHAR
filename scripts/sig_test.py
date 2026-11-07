"""Paired significance tests for the paper (reads results_seed*.pickle, no training).

Pairs = (test subject, seed): the same seed gives the same test sequence for every method.
For each comparison: mean difference, wins/ties/losses, Wilcoxon signed-rank and paired t-test.

  A. ours vs BHAR and vs the untrained start, on 4 datasets
  B. gradient-fix ablation (PAMAP2)
  C. Pareto: ours vs the fixed-timer curve at the same packet rate (PAMAP2, DSADS); the fixed
     curve is interpolated linearly between its two neighbouring points, per pair, with weights
     from the mean packet rates below (bhar_packets.py, EXPERIMENTS.md exp36/38)

usage:  python scripts/sig_test.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhar_cached import ART, load  # noqa: E402

M = "conservative-asynchronous_multisensor_time_context"
BASE = os.path.join(ART, "saved_data", "results")
BHAR = M
FIELD = {"acc": 1, "f1": 0}

# mean packets / s / sensor (3 seeds)
PKT = {
    "pamap2": {"fix50_notrain": .213, "fix100_notrain": .137, "fix150_notrain": .102, "fix200_notrain": .082,
               "jnn100": .149, "jnn150": .113, "jnn200": .093},
    "dsads": {"fix50_notrain": .217, "fix100_notrain": .147, "fix150_notrain": .109, "fix200_notrain": .087,
              "jnn100": .163, "jnn150": .122, "jnn200": .099},
}


def method_dir(dataset, name):
    return os.path.join(BASE, dataset, "tinyhar", name if name == BHAR else f"{name}/e2e/{M}")


def pairs(dataset, name, field):
    """-> {(subject, seed): value}"""
    res = load(method_dir(dataset, name))
    return {(s, seed): v[FIELD[field]] for seed, r in res.items() for s, v in r.items()}


def report(label, a, b):
    """a, b: {(subject, seed): value}; prints a - b over common pairs."""
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"  {label:44s} (missing results)")
        return
    d = 100 * np.array([a[k] - b[k] for k in keys])
    w, t, l = (d > 1e-9).sum(), (abs(d) <= 1e-9).sum(), (d < -1e-9).sum()
    p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    p_t = stats.ttest_rel(d, np.zeros_like(d)).pvalue if np.std(d) > 0 else 1.0
    print(f"  {label:44s} n={len(keys):2d}  diff {d.mean():+5.2f} pts  W/T/L {w:2d}/{t:2d}/{l:2d}"
          f"  Wilcoxon p={p_w:.3f}  t-test p={p_t:.3f}")


def fixed_curve(dataset, pkt, field):
    """Fixed-timer accuracy interpolated at packet rate pkt, per pair."""
    pts = sorted((v, k) for k, v in PKT[dataset].items() if k.startswith("fix"))
    for (p_lo, k_lo), (p_hi, k_hi) in zip(pts, pts[1:]):
        if p_lo <= pkt <= p_hi:
            w = (pkt - p_lo) / (p_hi - p_lo)
            lo, hi = pairs(dataset, k_lo, field), pairs(dataset, k_hi, field)
            return {k: (1 - w) * lo[k] + w * hi[k] for k in set(lo) & set(hi)}
    return {}


def main():
    for field in ("acc", "f1"):
        print(f"\n==================== {field} (difference in points, positive = first method better)")
        print("A. ours vs BHAR / vs untrained start")
        for ds in ("pamap2", "dsads", "rwhar", "opportunity"):
            print(f" {ds}")
            bhar = pairs(ds, BHAR, field)
            for ours in ("jnn100", "nn100"):
                report(f"{ours} - BHAR", pairs(ds, ours, field), bhar)
            report("jnn100 - fixed (0,100) no training", pairs(ds, "jnn100", field), pairs(ds, "fix100_notrain", field))
            report("fixed (0,100) no training - BHAR", pairs(ds, "fix100_notrain", field), bhar)
        print(" all 4 datasets pooled")
        for ours in ("jnn100", "nn100"):
            a, b = {}, {}
            for ds in ("pamap2", "dsads", "rwhar", "opportunity"):
                a.update({(ds,) + k: v for k, v in pairs(ds, ours, field).items()})
                b.update({(ds,) + k: v for k, v in pairs(ds, BHAR, field).items()})
            report(f"{ours} - BHAR", a, b)
        print("B. gradient-fix ablation (pamap2)")
        report("jnn100 - jnnraw100 (no fix)", pairs("pamap2", "jnn100", field), pairs("pamap2", "jnnraw100", field))
        report("jnnraw100 - fixed (0,100) no training", pairs("pamap2", "jnnraw100", field),
               pairs("pamap2", "fix100_notrain", field))
        print("C. Pareto: ours - fixed-timer curve at the same packet rate")
        for ds in ("pamap2", "dsads"):
            print(f" {ds}")
            for ours in ("jnn100", "jnn150", "jnn200"):
                p = PKT[ds][ours]
                report(f"{ours} ({p:.3f} pkt/s) - fixed curve", pairs(ds, ours, field), fixed_curve(ds, p, field))


if __name__ == "__main__":
    main()
