"""BHAR Fig. 3-style example (one test subject, one sensor, a short window): energy of each policy,
then for BHAR and ours the activity, the active (green) / passive (red) region and the packets.
Passive = the activity has changed since this sensor's last packet arrived.

usage:  python scripts/fig_example.py [--dataset pamap2] [--subject 1] [--seed 0] [--sensor 2]
                                      [--start 300 --length 40]   (or --scan: print candidate windows)
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import bhar_e2e as be
import bhar_hook  # noqa: F401


def policy(d, s, seed, bp):
    with open(os.path.join(d, f"policy_[{s}]_seed{seed}.pkl"), "rb") as f:
        th = pickle.load(f)["best"][bp]
    return f"conservative_{th[0]}_{th[1]}" if th[0] == "neural" else f"conservative_{float(th[0])}_{float(th[1])}"


def regions(y, arrive):
    """passive[t]: no packet has arrived since the last activity change before t."""
    last_tr = np.maximum.accumulate(np.where(np.r_[False, np.diff(y) != 0], np.arange(len(y)), 0))
    last_pk = np.full(len(y), -1)
    last_pk[arrive[arrive < len(y)]] = arrive[arrive < len(y)]
    last_pk = np.maximum.accumulate(last_pk)
    return last_pk < last_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(be.DATASETS), default="pamap2")
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensor", type=int, default=0, help="index into the body parts")
    ap.add_argument("--ours", default="jnn100/e2e")
    ap.add_argument("--start", type=float, default=None, help="window start (s); default: 10 s before the "
                                                              "first transition after 100 s")
    ap.add_argument("--length", type=float, default=40)
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    s = args.subject or be.SUBJECTS[0]
    bp = be.BODY_PARTS[args.sensor]
    cfg = be.SimConfig()
    fs = cfg.fs
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh), cfg.packet_size, cfg.leakage,
                                    fs, cfg.max_energy, cfg.policy_frequency, cfg.init_energy)
    X, y = be.sequences(s, args.seed, test=True)[2]
    y = np.asarray(y)
    runs = {}
    for name, d in [("BHAR", be.BHAR_POLICY), ("ours", be.out_policy_dir(args.ours))]:
        idx = ehs.sparsify_data(policy(d, s, args.seed, bp), X[args.sensor])
        runs[name] = (np.array(ehs.e_trace, dtype=float).copy(), idx[:, 1].astype(int), ehs.thresh)
    tr = np.flatnonzero(np.diff(y) != 0) + 1
    start = args.start if args.start is not None else tr[tr > 100 * fs][0] / fs - 10
    a, b = int(start * fs), int((start + args.length) * fs)
    t = np.arange(a, b) / fs

    for name, (e, arr, _) in runs.items():
        pas = regions(y, arr)
        print(f"{name}: {np.sum((arr >= a) & (arr < b))} packets in window, passive {pas[a:b].mean():.1%} "
              f"(whole sequence {pas.mean():.1%})")

    fig, ax = plt.subplots(3, 1, figsize=(5.5, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    col = {"BHAR": "tab:blue", "ours": "tab:green"}
    for name, (e, arr, th) in runs.items():
        ax[0].plot(t, e[a:b] * 1e6, color=col[name], lw=1, label=name)
    ax[0].axhline(runs["BHAR"][2] * 1e6, ls=":", color="k", lw=1)
    ax[0].axhline(cfg.max_energy * 1e6, ls="-.", color="k", lw=1)
    ax[0].set_ylabel("energy (µJ)")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title(f"{args.dataset}, subject {s}, sensor {bp}")
    for r, name in enumerate(["BHAR", "ours"], 1):
        e, arr, _ = runs[name]
        pas = regions(y, arr)[a:b]
        yy = y[a:b]
        ax[r].plot(t, np.where(pas, np.nan, yy), color="tab:green", lw=5, alpha=.6, solid_capstyle="butt")
        ax[r].plot(t, np.where(pas, yy, np.nan), color="tab:red", lw=5, alpha=.8, solid_capstyle="butt")
        ax[r].plot(t, yy, color="k", lw=1)
        k = arr[(arr >= a) & (arr < b)]
        ax[r].scatter(k / fs, y[k] + 0.8, marker="v", color=col[name], s=18)
        ax[r].set_ylabel("activity")
        ax[r].text(0.01, 0.85, name, transform=ax[r].transAxes, fontsize=9, weight="bold")
    ax[2].set_xlabel("time (s); green = active, red = passive, ▼ = packet")
    fig.tight_layout()
    out = f"report/fig_example_{args.dataset}_s{s}_{bp}_{int(start)}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out[:-4] + ".pdf")
    print("figure:", out)


if __name__ == "__main__":
    main()
