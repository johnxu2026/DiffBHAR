"""Energy budget per sensor on the test subjects: where the harvested energy goes, in uW.

Replays each policy with BHAR's sensor simulator (as bhar_packets.py) and splits the harvested
power into: leakage (sleep + 5 Hz policy step, E_L), transmission (packets x E_th), energy lost
because the capacitor was full (overflow at E_max), and the change of stored energy.
The neural policy's own compute is not in the simulator; it is added from the nRF52 estimate
(8-bit MLP, 0.09 uJ per decision, run only when E >= E_th).

usage:  python scripts/energy_budget.py --dataset pamap2 --tags fix100_notrain/e2e nn100/e2e jnn100/e2e
"""
import argparse
import os
import pickle

import numpy as np

import bhar_e2e as be
import bhar_hook  # noqa: F401  (replays neural policies)

E_DECISION = 0.09e-6     # J per 8-bit MLP decision on nRF52 (report/policy_nrf52_int8.c)


def budget(ehs, n_pkt):
    """Energy terms in J for the trace just simulated by ehs."""
    e, h = np.asarray(ehs.e_trace, float), np.asarray(ehs.e_harvest, float)
    H = h[1:].sum()
    tx = n_pkt * ehs.thresh
    d_e = e[-1] - e[0]
    full = np.where(e[1:] >= ehs.MAX_E * (1 - 1e-9))[0] + 1
    over = np.clip(e[full - 1] + h[full] - ehs.LEAKAGE_PER_SAMPLE - ehs.MAX_E, 0, None).sum()
    leak = H - tx - d_e - over      # leakage actually drawn (less than E_L*T when the capacitor is empty)
    return np.array([H, leak, tx, over, d_e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(be.DATASETS), default="pamap2")
    ap.add_argument("--tags", nargs="*", default=["nn100/e2e", "jnn100/e2e"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--efficiency", type=float, default=0.5)
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    dirs = {"opportunistic": None}
    if args.efficiency == 0.5:
        dirs["BHAR"] = be.BHAR_POLICY
    dirs.update({t: be.out_policy_dir(t) for t in args.tags})

    cfg = be.SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=args.efficiency),
                                    cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                    cfg.policy_frequency, cfg.init_energy)
    tot = {m: np.zeros(5) for m in dirs}    # summed J over seeds, subjects, sensors
    secs = {m: 0.0 for m in dirs}           # summed sensor-seconds
    dec = {m: 0 for m in dirs}              # network decisions (E >= E_th)
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
                for i, bp in enumerate(be.BODY_PARTS):
                    n = len(ehs.sparsify_data(pol[bp], X[i]))
                    tot[m] += budget(ehs, n)
                    secs[m] += len(X[i]) / cfg.fs
                    if pol[bp].startswith(bhar_hook.PREFIX):
                        dec[m] += ehs.n_feasible

    print(f"{args.dataset}, efficiency {args.efficiency}: average power per sensor in uW "
          f"(test subjects, {len(args.seeds)} seeds)")
    print(f"{'method':22s} {'harvest':>8s} {'leak':>7s} {'TX':>7s} {'policyNN':>8s} {'full-cap':>8s} "
          f"{'dStored':>7s} {'used%':>6s} {'pkt/s':>6s}")
    for m in dirs:
        if not secs[m]:
            continue
        H, leak, tx, over, d_e = tot[m] / secs[m] * 1e6
        nn = dec[m] * E_DECISION / secs[m] * 1e6
        pkt = tot[m][2] / ehs.thresh / secs[m]
        print(f"{m:22s} {H:8.2f} {leak:7.2f} {tx:7.2f} {nn:8.2f} {over:8.2f} {d_e:7.2f} "
              f"{100 * (leak + tx + nn) / H:6.1f} {pkt:6.3f}")
    print(f"dense (a packet every 12 samples, BHAR's unconstrained_stride) would need TX "
          f"{cfg.fs / 12 * cfg.tx_thresh * 1e6:.1f} uW + leakage {cfg.leakage * 1e6:.1f} uW")


if __name__ == "__main__":
    main()
