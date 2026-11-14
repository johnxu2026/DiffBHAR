"""Gradient size vs. segment length, with and without the fix of Sec. III-D (--no_delta_grad).

Same setup as grad_check.py (fixed timer theta = (0, theta_E), noisy STE policy, frozen pretrained
classifier, loss_task), but on random segments of several lengths. Prints the median over batches
of |d loss_task / d theta_E| and saves report/fig_grad_horizon.{pdf,png}.
Expected: without the fix the gradient grows exponentially with the horizon once the timer binds
(theta_E >= 100); with the fix it stays bounded.

    python scripts/grad_horizon.py --subjects 1 2 3 4 5 6 7 8
With several subjects (LOSO folds) the plot shows the median over subjects and a min-max band.
"""
import argparse
import os

import numpy as np
import torch

import bhar_e2e as be
from diffbhar.pipeline import run_e2e
from diffbhar.policies import DiffConservativePolicy
from diffbhar.sim import SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--theta", type=float, nargs="+", default=[50, 100, 200])
    ap.add_argument("--lengths", type=int, nargs="+", default=[10, 25, 50, 100, 200], help="seconds")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--scale", type=float, default=5.0)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg0 = SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg0.tx_thresh, efficiency=0.5),
                                    cfg0.packet_size, cfg0.leakage, cfg0.fs, cfg0.max_energy,
                                    cfg0.policy_frequency, cfg0.init_energy)
    C = len(be.KW["activities"])
    res = {}
    for s in args.subjects:
        name = f"[{s}]_seed{args.seed}"
        (X, y), _ = be.sequences(s, args.seed)
        model = be.model_builder(**be.KW).float()
        sd = {k.removeprefix("multisensor_model."): v
              for k, v in torch.load(os.path.join(be.PRETRAINED_CLF, f"{name}.pth"),
                                     map_location="cpu")["model_state_dict"].items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def clf(x, ages):
            return model(x.float(), ages.float()).double()

        for L in args.lengths:
            be.SEG = L * cfg0.fs
            rng = np.random.default_rng(args.seed)
            batches = [be.sample_batch(X, y, ehs, rng, args.batch_size) for _ in range(args.batches)]
            for fix in (False, True):
                cfg = SimConfig(delta_grad=not fix)
                for th in args.theta:
                    torch.manual_seed(args.seed)
                    pol = DiffConservativePolicy(torch.arange(3), 3, init=(0.0, th), scale=args.scale, noise=True)
                    g = []
                    for xb, yb, hb in batches:
                        pol.sensor_index = torch.arange(3).repeat(len(yb))
                        pol.theta.grad = None
                        run_e2e(xb, yb, hb, pol, clf, cfg, C, train=True).loss_task.backward()
                        g.append(abs(pol.theta.grad[:, 1].mean().item()))
                    res[(fix, th, L, s)] = float(np.median(g))
                    print(f"subject {s}  L {L:4d} s  fix {fix!s:5s}  theta_E {th:4.0f}  |grad| {res[(fix, th, L, s)]:.3e}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(args.theta)))
    for c, th in zip(colors, args.theta):
        for fix, ls, lab in ((False, "--", "w/o fix"), (True, "-", "with fix")):
            v = np.array([[max(res[(fix, th, L, s)], 1e-12) for L in args.lengths] for s in args.subjects])
            ax.plot(args.lengths, np.median(v, 0), ls, color=c, marker="o", ms=2.5,
                    label=fr"$\theta_\mathrm{{E}}={th:.0f}$, {lab}")
            if len(v) > 1:
                ax.fill_between(args.lengths, v.min(0), v.max(0), color=c, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("segment length (s)")
    ax.set_ylabel(r"$|\partial \mathcal{L}_\mathrm{task}/\partial\theta_\mathrm{E}|$")
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    fig.tight_layout(pad=0.2)
    os.makedirs("report", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"report/fig_grad_horizon.{ext}", dpi=300)
    print("saved report/fig_grad_horizon.{pdf,png}")


if __name__ == "__main__":
    main()
