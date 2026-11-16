"""Why the passive region stays: after each activity transition, how long until the next packet
arrives (from any sensor), and what that wait is spent on. Replays each policy with BHAR's
simulator on the test subjects (as energy_budget.py) and splits the passive time per transition into

  energy  : no sensor has E >= E_th yet (no policy could send)
  policy  : some sensor could send, but the policy waited
  sampling: the 8-sample packet being acquired (320 ms), incl. a packet already in progress

'policy' is an upper bound on what a perfect transition trigger could save (first-order: the
energy traces are those of the replayed policy). Also prints the share of transitions at which
no sensor has enough energy.

usage:  python scripts/transition_wait.py --dataset pamap2 --tags jnn100/e2e
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
    ap.add_argument("--tags", nargs="*", default=["jnn100/e2e"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    dirs = {"opportunistic": None, "BHAR": be.BHAR_POLICY}
    dirs.update({t: be.out_policy_dir(t) for t in args.tags})

    cfg = be.SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                    cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                    cfg.policy_frequency, cfg.init_energy)
    W = cfg.packet_size
    acc = {m: dict(n=0, starved=0, total=0.0, energy=0.0, policy=0.0, sampling=0.0, steps=0) for m in dirs}
    for seed in args.seeds:
        for s in be.SUBJECTS:
            X, y = be.sequences(s, seed, test=True)[2]
            trans = np.where(y[:-1] != y[1:])[0] + 1
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
                arrivals, ready = [], []
                for i, bp in enumerate(be.BODY_PARTS):
                    pk = np.asarray(ehs.sparsify_data(pol[bp], X[i]))
                    if len(pk):
                        arrivals.append(pk[:, 1] - 1)            # BHAR's arrival time (sparse_data_utils)
                    ready.append(np.asarray(ehs.e_trace, float) >= ehs.thresh)
                arr = np.sort(np.concatenate(arrivals)) if arrivals else np.array([], int)
                any_ready = np.any(np.stack(ready), 0)
                a = acc[m]
                a["steps"] += len(y)
                for t in trans:
                    j = np.searchsorted(arr, t)                  # first arrival at or after t
                    if j == len(arr):
                        continue
                    A = arr[j]
                    start = A - (W - 1)                          # first sample of that packet
                    a["n"] += 1
                    a["starved"] += int(not any_ready[t])
                    a["total"] += A - t
                    if start <= t:                               # packet already being sampled
                        a["sampling"] += A - t
                        continue
                    r = np.argmax(any_ready[t:start]) if any_ready[t:start].any() else start - t
                    a["energy"] += r
                    a["policy"] += (start - t) - r
                    a["sampling"] += A - start

    fs = cfg.fs
    print(f"{args.dataset}: wait from each activity transition to the next packet arrival (any sensor), "
          f"test subjects, {len(args.seeds)} seeds")
    print(f"{'method':16s} {'trans':>6s} {'no-energy@t':>11s} {'wait s':>7s} {'energy':>7s} {'policy':>7s} {'sampling':>8s}")
    for m, a in acc.items():
        if not a["n"]:
            continue
        n = a["n"]
        print(f"{m:16s} {n:6d} {100 * a['starved'] / n:10.1f}% {a['total'] / n / fs:7.2f} "
              + " ".join(f"{100 * a[k] / a['total']:6.1f}%" for k in ("energy", "policy"))
              + f" {100 * a['sampling'] / a['total']:7.1f}%")


if __name__ == "__main__":
    main()
