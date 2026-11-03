"""Does an 8-bit version of the neural policy make the same send decisions as the float one?

Replays each test sequence with the float policy (as BHAR's evaluation does, through our simulator)
and, at every decision where a packet is affordable (E >= E_th, sensor not busy), also evaluates an
8-bit emulation of scripts/../report/policy_nrf52_int8.c:
  inputs  int8, one scale per feature, calibrated on the subject's validation sequence
  W1      int8, one scale per hidden unit;  b1 in accumulator scale
  tanh    int8 lookup table, input range [-4, 4]
  w2      int8, one scale per sensor;  b2 float
It prints the fraction of affordable decisions that flip, and the packets/s/sensor when the 8-bit
policy drives the sensor itself.

usage:  python scripts/int8_check.py --dataset pamap2 --tags nn100/e2e jnn100/e2e
"""
import argparse
import os

import numpy as np
import torch

import bhar_e2e as be
from diffbhar.policies import NeuralPolicy
from diffbhar.sim import SimConfig, simulate

A_RANGE = 4.0   # tanh lookup-table input range


def q(x, s):
    return torch.clamp(torch.round(x / s), -128, 127)


class Int8Policy(torch.nn.Module):
    """8-bit emulation of a float NeuralPolicy; records float and 8-bit logits at every call."""

    def __init__(self, pol, in_scale, drive_int8=False):
        super().__init__()
        self.pol, self.sx, self.drive = pol, in_scale, drive_int8
        w1 = pol.w1.detach()                                     # [N, 5, H]
        self.sw1 = w1.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127   # per sensor, per unit
        self.w1q = q(w1, self.sw1)
        w2 = pol.w2.detach()                                     # [N, H]
        self.sw2 = w2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        self.w2q = q(w2, self.sw2)
        self.rec_f, self.rec_q = [], []

    def logit8(self, x):
        i = self.pol.sensor_index
        xq = q(x, self.sx)                                       # [N, 5]
        a = torch.einsum("ni,nih->nh", xq * self.sx, self.w1q[i] * self.sw1[i])   # int8 x int8, rescaled
        a = a + self.pol.b1.detach()[i]
        aq = q(a, A_RANGE / 127)
        t = torch.round(127 * torch.tanh(aq * A_RANGE / 127))   # lookup table output, int8
        return (self.w2q[i] * t).sum(-1) * self.sw2[i, 0] / 127 + self.pol.b2.detach()[i]

    def forward(self, f):
        x = self.pol.features(f)
        lf = self.pol(f)
        l8 = self.logit8(x)
        self.rec_f.append(lf)
        self.rec_q.append(l8)
        return l8 if self.drive else lf

    def binarize(self, logit, train=False):
        return self.pol.binarize(logit, train=False)


def features_max(pol, h, cfg):
    """Per-feature max |value| over all decisions of a float replay (input calibration)."""
    seen = []

    class Rec(torch.nn.Module):
        def forward(self, f):
            seen.append(pol.features(f))
            return pol(f)

        def binarize(self, logit, train=False):
            return pol.binarize(logit, train=False)

    simulate(h, Rec(), cfg, train=False)
    return torch.cat(seen).abs().amax(0).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(be.DATASETS), default="pamap2")
    ap.add_argument("--tags", nargs="+", default=["nn100/e2e"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    be.set_dataset(args.dataset)
    torch.set_default_dtype(torch.float64)
    c = be.SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=c.tx_thresh), c.packet_size, c.leakage,
                                    c.fs, c.max_energy, c.policy_frequency, c.init_energy)
    cfg = SimConfig(packet_size=ehs.packet_size, leakage=ehs.leakage, fs=ehs.FS, max_energy=ehs.MAX_E,
                    policy_frequency=ehs.FS // ehs.policy_period, tx_thresh=ehs.thresh,
                    init_energy=ehs.init_energy)
    N = len(be.BODY_PARTS)
    stats = {t: dict(flip=0, n=0, pf=[], p8=[]) for t in args.tags}
    for seed in args.seeds:
        for s in be.SUBJECTS:
            _, (Xv, _), (X, y) = be.sequences(s, seed, test=True)
            hv = torch.tensor(be.harvest(Xv, ehs))
            ht = torch.tensor(be.harvest(X, ehs))
            secs = N * ht.shape[1] / cfg.fs
            for t in args.tags:
                p = os.path.join(be.out_policy_dir(t), f"policy_[{s}]_seed{seed}.pt")
                if not os.path.exists(p):
                    continue
                sd = torch.load(p, map_location="cpu")
                pol = NeuralPolicy(torch.arange(N), N, hidden=sd["b1"].shape[1], noise=False)
                pol.load_state_dict(sd, strict=False)
                with torch.no_grad():
                    sx = features_max(pol, hv, cfg) / 127
                    w = Int8Policy(pol, sx)
                    o = simulate(ht, w, cfg, train=False)
                    m = o.feasible.bool()                                   # [N, J]
                    J = m.shape[1]                                          # last decision has no room
                    lf, l8 = torch.stack(w.rec_f, 1)[:, :J], torch.stack(w.rec_q, 1)[:, :J]
                    st = stats[t]
                    st["flip"] += int(((lf > 0) != (l8 > 0))[m].sum())
                    st["n"] += int(m.sum())
                    st["pf"].append(float(o.a.sum()) / secs)
                    o8 = simulate(ht, Int8Policy(pol, sx, drive_int8=True), cfg, train=False)
                    st["p8"].append(float(o8.a.sum()) / secs)
            print(f"seed {seed} subject {s} done", flush=True)

    print(f"\n{args.dataset}: 8-bit vs float, test subjects x seeds {args.seeds}")
    for t, st in stats.items():
        if st["n"]:
            print(f"{t:14s} decisions flipped (E >= E_th): {st['flip']}/{st['n']} = {st['flip'] / st['n']:.3%}   "
                  f"packets/s/sensor float {np.mean(st['pf']):.4f}  8-bit {np.mean(st['p8']):.4f}")


if __name__ == "__main__":
    main()
