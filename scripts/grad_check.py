"""Does the straight-through gradient point the right way?

For theta = (0, theta_p) on all sensors (the fixed rules of experiment 12), on the same random
100 s training segments: the gradient of loss_task that training uses (STE, noisy policy, frozen
pretrained classifier), its mean and t value (mean / standard error) over the batches, and the
true loss of the hard rule on those segments and on the whole validation sequence. If the gradient
is right, d loss / d theta_p has the sign of the loss differences between neighbouring theta_p
and a |t| well above 2.

Found 2026-10-26: with the default settings the gradient explodes (1e9 to 1e20) for theta_p >= 100.
When the rule binds, each packet's timing depends on the previous one through the delta reset, and
every link multiplies the gradient by about (Delta + w + 1) * sigmoid' / scale, ~10 at theta_p = 200.
--no_delta_grad cuts that chain.

    python scripts/grad_check.py --subjects 1 2 --device cuda
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
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--theta_p", type=float, nargs="+", default=[50, 100, 150, 200, 300])
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--scale", type=float, default=5.0, help="policy temperature, as bhar_e2e.py")
    ap.add_argument("--no_delta_grad", action="store_true", help="no gradient through the delta reset")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    cfg = SimConfig(delta_grad=not args.no_delta_grad)
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                    cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                    cfg.policy_frequency, cfg.init_energy)
    C = len(be.KW["activities"])
    for s in args.subjects:
        name = f"[{s}]_seed{args.seed}"
        (X, y), (Xv, yv) = be.sequences(s, args.seed)
        val = [torch.tensor(a)[None].to(dev) for a in (Xv, yv, be.harvest(Xv, ehs))]
        rng = np.random.default_rng(args.seed)
        batches = [[t.to(dev) for t in be.sample_batch(X, y, ehs, rng, args.batch_size)]
                   for _ in range(args.batches)]
        model = be.model_builder(**be.KW).float().to(dev)
        sd = {k.removeprefix("multisensor_model."): v
              for k, v in torch.load(os.path.join(be.PRETRAINED_CLF, f"{name}.pth"),
                                     map_location=dev)["model_state_dict"].items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def clf(x, ages):
            with torch.backends.cudnn.flags(enabled=False):
                return model(x.float(), ages.float()).double()

        print(f"subject {s}: theta_p | val acc  pkt/s  loss | segment loss | "
              f"d loss/d theta_p (t) | d loss/d theta_e (t), mean over sensors, x1e3")
        for tp in args.theta_p:
            torch.manual_seed(args.seed)
            pol = DiffConservativePolicy(torch.arange(3), 3, init=(0.0, tp), scale=args.scale, noise=True).to(dev)
            hard = DiffConservativePolicy(torch.arange(3), 3, init=(0.0, tp), scale=args.scale, noise=False).to(dev)
            grads, seg_loss = [], []
            for xb, yb, hb in batches:
                pol.sensor_index = hard.sensor_index = torch.arange(3, device=dev).repeat(len(yb))
                pol.theta.grad = None
                run_e2e(xb, yb, hb, pol, clf, cfg, C, train=True).loss_task.backward()
                grads.append(pol.theta.grad.mean(0).cpu().numpy())
                with torch.no_grad():
                    seg_loss.append(run_e2e(xb, yb, hb, hard, clf, cfg, C, train=False,
                                            counterfactual=False).loss_task.item())
            g = np.array(grads)
            m, t = g.mean(0), g.mean(0) / (g.std(0, ddof=1) / np.sqrt(len(g)) + 1e-12)
            hard.sensor_index = torch.arange(3, device=dev)
            with torch.no_grad():
                o = run_e2e(*val, hard, clf, cfg, C, train=False, counterfactual=False)
            acc, pkt = (o.preds[0] == yv).mean(), o.sim.a.sum().item() / (3 * len(yv) / cfg.fs)
            print(f"  {tp:5.0f} | {acc:.3f}  {pkt:.3f}  {o.loss_task.item():.3f} | {np.mean(seg_loss):.3f} | "
                  f"{m[1] * 1e3:+8.3f} ({t[1]:+5.1f}) | {m[0] * 1e3:+8.3f} ({t[0]:+5.1f})", flush=True)


if __name__ == "__main__":
    main()
