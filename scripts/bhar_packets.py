"""Packets sent on the test sequences by each policy in BHAR's TinyHAR setup (--dataset, default PAMAP2).

BHAR's results pickles store accuracy and regions but not packet counts, so this rebuilds each
test sequence (same seed and order as train_har_policy.py) and replays every learned policy
with BHAR's own sensor simulator.

usage:  python scripts/bhar_packets.py [--dataset pamap2] [--tags e2e zero/e2e] [--seeds 0 1 2]
"""
import argparse
import os
import pickle

import numpy as np

import bhar_e2e as be
import bhar_hook  # noqa: F401  (replays neural policies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(be.DATASETS), default="pamap2")
    ap.add_argument("--tags", nargs="*", default=["e2e", "zero/e2e"], help="tags given to bhar_e2e.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    dirs = {"opportunistic": None, "BHAR": be.BHAR_POLICY}
    dirs.update({t: be.out_policy_dir(t) for t in args.tags})

    cfg = be.SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                    cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                    cfg.policy_frequency, cfg.init_energy)
    rate = {m: {} for m in dirs}                    # method -> seed -> [packets/s/sensor per subject]
    for seed in args.seeds:
        for s in be.SUBJECTS:
            X, y = be.sequences(s, seed, test=True)[2]
            for m, d in dirs.items():
                if d is None:
                    pol = {bp: "opportunistic" for bp in be.BODY_PARTS}
                else:
                    p = os.path.join(d, f"policy_[{s}]_seed{seed}.pkl")
                    if not os.path.exists(p):
                        continue
                    with open(p, "rb") as f:
                        th = pickle.load(f)["best"]
                    pol = {bp: f"conservative_{th[bp][0]}_{th[bp][1]}" if th[bp][0] == "neural"
                           else f"conservative_{float(th[bp][0])}_{float(th[bp][1])}" for bp in be.BODY_PARTS}
                n = sum(len(ehs.sparsify_data(pol[bp], X[i])) for i, bp in enumerate(be.BODY_PARTS))
                rate[m].setdefault(seed, []).append(n / (len(be.BODY_PARTS) * len(y) / cfg.fs))

    print("packets per second per sensor on the test subjects: mean over subjects, then mean +- std over seeds")
    for m, r in rate.items():
        per_seed = [np.mean(v) for v in r.values() if len(v) == len(be.SUBJECTS)]
        if per_seed:
            print(f"{m:16s} {len(per_seed)} seeds  {np.mean(per_seed):.4f} +- {np.std(per_seed):.4f}")


if __name__ == "__main__":
    main()
