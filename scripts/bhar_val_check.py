"""Does the classifier that scores theta favour opportunistic sending?

On BHAR's validation sequence of each test subject, held-prediction accuracy and packets/s/sensor
of theta = (0, 0) (= opportunistic) and of BHAR's learned theta, each scored with the cached
pretrained classifier (what --init zero trains against) and with BHAR's finetuned one.

    python scripts/bhar_val_check.py --seed 0 --device cuda
"""
import argparse
import os
import pickle

import numpy as np
import torch

import bhar_e2e as be
from diffbhar.pipeline import run_e2e
from diffbhar.policies import DiffConservativePolicy
from diffbhar.sim import SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, nargs="+", default=be.SUBJECTS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    cfg = SimConfig()
    ehs = be.EnergyHarvestingSensor(be.EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                    cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                    cfg.policy_frequency, cfg.init_energy)
    rows = []
    for s in args.subjects:
        name = f"[{s}]_seed{args.seed}"
        _, (Xv, yv) = be.sequences(s, args.seed)
        val = [torch.tensor(a)[None].to(dev) for a in (Xv, yv, be.harvest(Xv, ehs))]
        with open(os.path.join(be.BHAR_POLICY, f"policy_{name}.pkl"), "rb") as f:
            best = pickle.load(f)["best"]
        th_bhar = [[float(v) for v in best[bp]] for bp in be.BODY_PARTS]
        row = []
        for clf_path in (os.path.join(be.PRETRAINED_CLF, f"{name}.pth"),
                         os.path.join(be.BHAR_CLF, f"finetuned_classifier_{name}.pth")):
            model = be.model_builder(**be.KW).float().to(dev)
            sd = {k.removeprefix("multisensor_model."): v
                  for k, v in torch.load(clf_path, map_location=dev)["model_state_dict"].items()}
            model.load_state_dict(sd, strict=False)
            model.eval()

            def clf(x, ages):
                with torch.backends.cudnn.flags(enabled=False):
                    return model(x.float(), ages.float()).double()

            for th in ([[0.0, 0.0]] * 3, th_bhar):
                pol = DiffConservativePolicy(torch.arange(3), 3, init=th, noise=False).to(dev)
                with torch.no_grad():
                    o = run_e2e(*val, pol, clf, cfg, len(be.KW["activities"]), train=False, counterfactual=False)
                row += [(o.preds[0] == yv).mean(), o.sim.a.sum().item() / (3 * len(yv) / cfg.fs)]
        rows.append(row)
        print(f"s{s}: pretrained clf: opp {row[0]:.3f} ({row[1]:.3f} pkt/s)  bhar {row[2]:.3f} ({row[3]:.3f})"
              f" | finetuned clf: opp {row[4]:.3f}  bhar {row[6]:.3f}", flush=True)
    m = np.mean(rows, 0)
    print(f"mean: pretrained clf: opp {m[0]:.3f} ({m[1]:.3f} pkt/s)  bhar {m[2]:.3f} ({m[3]:.3f})"
          f" | finetuned clf: opp {m[4]:.3f}  bhar {m[6]:.3f}")


if __name__ == "__main__":
    main()
