"""Print BHAR's test results per method: cached ones from Zenodo (saved_data_cached) or
our reruns of BHAR's own code (saved_data, see scripts/bhar_posttrain.sh).

Each results_seed{s}.pickle maps subject -> (f1, acc, active_region, passive_region,
active_error, passive_error), written at the end of the artifact's train_har_policy.py.

usage:  python scripts/bhar_cached.py --dataset pamap2 --arch tinyhar
"""
import argparse
import glob
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "external", "BHAR_codes_artifact")
FIELDS = ["f1", "acc", "R_active", "R_passive", "eps_a", "eps_p"]


def load(method_dir):
    """-> {seed: {subject: np.array(6)}}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(method_dir, "results_seed*.pickle"))):
        seed = int(p.rsplit("seed", 1)[1].split(".")[0])
        with open(p, "rb") as f:
            out[seed] = {s: np.array(v, dtype=float) for s, v in pickle.load(f).items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pamap2")
    ap.add_argument("--arch", default="tinyhar")
    ap.add_argument("--methods", nargs="*", default=None, help="default: all non-ablation methods")
    ap.add_argument("--per_subject", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="read saved_data (our reruns) instead of saved_data_cached")
    args = ap.parse_args()

    base = os.path.join(ART, "saved_data" if args.rerun else "saved_data_cached", "results",
                        args.dataset, args.arch)
    methods = args.methods or sorted(m for m in os.listdir(base)
                                     if "ablation" not in m and "classifier" not in m)
    print(f"{args.dataset} / {args.arch}: mean over subjects, then mean +- std over seeds")
    print(f"{'method':52s} {'seeds':>5s} {'acc':>13s} {'F1':>13s} {'R_active':>8s} {'eps_a':>6s} {'eps_p':>6s}")
    for m in methods:
        res = load(os.path.join(base, m))
        if not res:
            continue
        # [seeds, fields]: mean over subjects for each seed
        per_seed = np.stack([np.mean(list(r.values()), axis=0) for r in res.values()])
        mu, sd = per_seed.mean(0), per_seed.std(0)
        print(f"{m:52s} {len(res):5d} {mu[1]:.3f}+-{sd[1]:.3f} {mu[0]:.3f}+-{sd[0]:.3f} "
              f"{mu[2]:8.3f} {mu[4]:6.3f} {mu[5]:6.3f}")
        if args.per_subject:
            subs = sorted(next(iter(res.values())))
            accs = {s: np.mean([res[k][s][1] for k in res if s in res[k]]) for s in subs}
            print("    acc per subject (mean over seeds): "
                  + "  ".join(f"s{s}={a:.3f}" for s, a in accs.items()))


if __name__ == "__main__":
    main()
