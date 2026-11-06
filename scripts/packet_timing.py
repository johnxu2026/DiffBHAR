"""When does each policy send, relative to activity transitions? (Geffen's packet-timing question)

Replays the test sequences (as bhar_packets.py) for BHAR, the fixed timer (0, 100) and ours, and reports
  - packet rate (packets/s/sensor) as a function of the time since the last activity transition,
  - latency from a transition to the first packet after it (any sensor), and the share of transitions
    followed by a packet within 2 s,
and draws report/fig_timing.{png,pdf}: (a) an example timeline, (b) packet rate vs time since transition.

usage:  python scripts/packet_timing.py [--dataset pamap2] [--seeds 0 1 2]
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import bhar_e2e as be
import bhar_hook  # noqa: F401  (replays neural policies)

BINS = [0, 2, 5, 10, 20, 40, np.inf]            # seconds since the last transition


def load_policy(d, s, seed):
    p = os.path.join(d, f"policy_[{s}]_seed{seed}.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        th = pickle.load(f)["best"]
    return {bp: f"conservative_{th[bp][0]}_{th[bp][1]}" if th[bp][0] == "neural"
            else f"conservative_{float(th[bp][0])}_{float(th[bp][1])}" for bp in be.BODY_PARTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(be.DATASETS), default="pamap2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ours", default="jnn100/e2e")
    ap.add_argument("--example", type=int, nargs=2, default=[0, 300], help="timeline window in seconds")
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    cfg = be.SimConfig()
    fs = cfg.fs
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh), cfg.packet_size, cfg.leakage,
                                    fs, cfg.max_energy, cfg.policy_frequency, cfg.init_energy)
    methods = {"BHAR": be.BHAR_POLICY, "fixed timer (0,100)": be.out_policy_dir("fix100_notrain/e2e"),
               "ours": be.out_policy_dir(args.ours)}
    N = len(be.BODY_PARTS)
    cnt = {m: np.zeros(len(BINS) - 1) for m in methods}     # packets per bin
    tim = np.zeros(len(BINS) - 1)                            # sensor-seconds per bin (same for all methods)
    lat = {m: [] for m in methods}
    example = {}
    for seed in args.seeds:
        for s in be.SUBJECTS:
            X, y = be.sequences(s, seed, test=True)[2]
            y = np.asarray(y)
            tr = np.flatnonzero(np.diff(y) != 0) + 1
            last = np.zeros(len(y))
            last[tr] = tr
            last = np.maximum.accumulate(last)                   # index of the last transition
            since = (np.arange(len(y)) - last) / fs
            b = np.digitize(since, BINS) - 1
            pols = {m: load_policy(d, s, seed) for m, d in methods.items()}
            if any(p is None for p in pols.values()):
                continue
            tim += N * np.bincount(b, minlength=len(BINS) - 1)[:len(BINS) - 1] / fs
            for m, pol in pols.items():
                starts = []
                for i, bp in enumerate(be.BODY_PARTS):
                    k = ehs.sparsify_data(pol[bp], X[i])[:, 0]
                    k = k[k < len(y)]
                    starts.append(k)
                    cnt[m] += np.bincount(b[k], minlength=len(BINS) - 1)[:len(BINS) - 1]
                allk = np.sort(np.concatenate(starts))
                for j, t0 in enumerate(tr):
                    t1 = tr[j + 1] if j + 1 < len(tr) else len(y)
                    nxt = allk[np.searchsorted(allk, t0):]
                    lat[m].append((nxt[0] - t0) / fs if len(nxt) and nxt[0] < t1 else np.inf)
                if seed == args.seeds[0] and s == be.SUBJECTS[0]:
                    example[m] = (starts, y)
            print(f"seed {seed} subject {s} done", flush=True)

    print(f"\n{args.dataset}: packet rate (packets/s/sensor) by time since the last activity transition")
    print("since (s)      " + "".join(f"{f'{BINS[i]}-{BINS[i + 1]}':>10s}" for i in range(len(BINS) - 1)))
    print(f"{'share of time':14s}" + "".join(f"{t / tim.sum():10.1%}" for t in tim))
    for m in methods:
        print(f"{m:14s}" + "".join(f"{c / t:10.3f}" for c, t in zip(cnt[m], tim)))
    print("\nfirst packet (any sensor) after a transition, before the next one")
    for m in methods:
        L = np.array(lat[m])
        print(f"{m:22s} median {np.median(L):5.2f} s   within 2 s {np.mean(L <= 2):.1%}   "
              f"none before next transition {np.mean(np.isinf(L)):.1%}   ({len(L)} transitions)")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a, b0 = args.example
    for r, m in enumerate(methods):
        starts, y = example[m]
        for i, k in enumerate(starts):
            k = k[(k >= a * fs) & (k < b0 * fs)] / fs
            ax[0].vlines(k, r * (N + 1) + i, r * (N + 1) + i + .8, color=f"C{r}", lw=.8)
    yy = example["ours"][1][a * fs:b0 * fs]
    for t0 in np.flatnonzero(np.diff(yy) != 0) + 1:
        ax[0].axvline(a + t0 / fs, color="k", ls=":", lw=.8)
    ax[0].set_yticks([r * (N + 1) + N / 2 for r in range(len(methods))], list(methods))
    ax[0].set_xlabel("time (s); dotted = activity transition")
    ax[0].set_title("(a) packets per sensor (one test subject)")
    w = .8 / len(methods)
    x = np.arange(len(BINS) - 1)
    for r, m in enumerate(methods):
        ax[1].bar(x + (r - 1) * w, cnt[m] / tim, w, label=m, color=f"C{r}")
    ax[1].set_xticks(x, [f"{BINS[i]}-{BINS[i + 1]}" if np.isfinite(BINS[i + 1]) else f">{BINS[i]}"
                         for i in range(len(BINS) - 1)])
    ax[1].set_xlabel("time since last transition (s)")
    ax[1].set_ylabel("packets / s / sensor")
    ax[1].set_title("(b) when packets are sent")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"report/fig_timing_{args.dataset}.{e}", dpi=200)
    print(f"figure: report/fig_timing_{args.dataset}.png")


if __name__ == "__main__":
    main()
