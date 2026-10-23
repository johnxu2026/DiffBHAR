"""Train BHAR's conservative policy parameters with gradients instead of pattern search,
inside BHAR's own PAMAP2 + TinyHAR setup, and save them where BHAR's code will pick them up.

For each test subject: same train sequence as train_har_policy.py (same seed), the classifier
(TinyHAR) is frozen, and theta is trained on the held-prediction loss (diffbhar.pipeline.run_e2e)
over random 100 s segments, like BHAR's own policy training. Every --val_every epochs theta is
scored on BHAR's validation sequence (held-prediction accuracy with the same frozen classifier) and
the best one is kept, as BHAR keeps its best validated policy. Two starting points:

  --init bhar: BHAR's learned theta + BHAR's finetuned classifier (gradient refinement of BHAR;
               needs BHAR's scratch run first: bash scripts/bhar_posttrain.sh conservative SEED scratch)
  --init zero: theta = (0, 0) like BHAR's pattern search + the cached pretrained classifier
               (nothing from BHAR's policy search; replaces it)

Joint training (--clf_lr > 0) also updates the classifier, on loss_task + lam * loss_active
(BHAR's finetuning loss); with --save_clf the trained classifier is written where BHAR looks for
its finetuned one, so BHAR evaluates it as is instead of finetuning again.

Then run BHAR's finetuning and evaluation on the new theta, with the same --tag:

    python scripts/bhar_e2e.py --seed 0 --init zero --tag zero/e2e
    bash scripts/bhar_posttrain.sh conservative 0 scratch zero/e2e

Use the artifact's own venv.
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "external", "BHAR_codes_artifact")
sys.path[:0] = [ROOT, ART]

from datasets.dataset import HARClassifierDataset, generate_activity_sequence  # noqa: E402
from energy_harvesting.energy_harvest import EnergyHarvester  # noqa: E402
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor  # noqa: E402
from models.model_builder import model_builder  # noqa: E402
from utils.setup_funcs import DATA_ROOT, init_logger  # noqa: E402

from diffbhar.pipeline import run_e2e  # noqa: E402
from diffbhar.policies import DiffConservativePolicy  # noqa: E402
from diffbhar.sim import SimConfig  # noqa: E402

# settings of experiments/posttraining_scripts/conservative_time_context-pamap2.sh
BODY_PARTS = ["hand", "chest", "ankle"]
KW = dict(dataset="pamap2", architecture="tinyhar", sensors=["acc"], body_parts=BODY_PARTS,
          activities=list(range(12)), val_frac=0.2, window_size=8)
SUBJECTS = list(range(1, 9))
SEG = 100 * 25                     # BHAR's policy training segment, 100 s
CKPT = os.path.join(ART, "saved_data", "checkpoints", "pamap2")
BHAR_POLICY = os.path.join(CKPT, "conservative-asynchronous_single_sensor")
BHAR_CLF = os.path.join(CKPT, "tinyhar", "conservative-asynchronous_multisensor_time_context")
PRETRAINED_CLF = os.path.join(ART, "saved_data_cached", "checkpoints", "pamap2", "tinyhar",
                              "multisensor_classifier_window_8_acc")


def out_policy_dir(tag):
    """Where BHAR looks for the policy when run with logging prefix <tag>/<method>:
    two levels above pamap2/tinyhar/<tag>/<method> (train_har_policy.py, dataset_level_dir)."""
    level = os.path.dirname(os.path.dirname(os.path.join("pamap2", "tinyhar", tag, "method")))
    return os.path.join(ART, "saved_data", "checkpoints", level, "conservative-asynchronous_single_sensor")


def sequences(test_subject, seed, test=False):
    """Train and validation (and test) sequences, each ([b, L, 3] raw data, as BHAR uses it, and
    labels), generated as in train_har_policy.py (same seed, in the order train, val, test)."""
    kw = dict(KW, dataset_dir=os.path.join(os.path.expanduser(DATA_ROOT["pamap2"]), "preprocessed_data"),
              subjects=[s for s in SUBJECTS if s != test_subject], normalize=False)
    dss = [HARClassifierDataset(**kw, train=True, val=False), HARClassifierDataset(**kw, train=False, val=True)]
    if test:
        dss.append(HARClassifierDataset(**dict(kw, subjects=[test_subject]), train=False, val=False))
    cmap = dss[0].dataset_info["sensor_channel_map"]
    np.random.seed(seed)
    out = []
    for ds in dss:
        X, y = generate_activity_sequence(np.concatenate(list(ds.raw_data.values())),
                                          np.concatenate(list(ds.raw_labels.values())), 10, 30, 25)
        per_bp = [X[:, np.isin(ds.active_channels, cmap[bp]["acc"])] for bp in BODY_PARTS]
        out.append((np.stack(per_bp), y.astype(np.int64)))
    return out


def harvest(x, ehs):
    """[b, L, 3] raw data -> [b, L] harvested energy per step (J)."""
    h = []
    for i in range(len(BODY_PARTS)):
        ehs.init_harvester(x[i])
        h.append(ehs.e_harvest.copy())
    return np.stack(h)


def sample_batch(X, y, ehs, rng, n):
    """n random 100 s segments; harvest computed per segment, like BHAR's policy training."""
    xs, ys, hs = [], [], []
    for _ in range(n):
        st = min(rng.integers(len(y)), len(y) - SEG)
        x = X[:, st:st + SEG]
        xs.append(x)
        ys.append(y[st:st + SEG])
        hs.append(harvest(x, ehs))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)), torch.tensor(np.stack(hs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS, help="test subjects")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--segments", type=int, default=64, help="100 s segments per epoch")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=5.0, help="policy temperature: logit = (Delta - tau) / scale")
    ap.add_argument("--kappa", type=float, default=5e-6, help="width (J) of the energy-feasibility surrogate")
    ap.add_argument("--val_every", type=int, default=2, help="epochs between validation scores")
    ap.add_argument("--clf_lr", type=float, default=0.0, help="> 0: joint training of theta and the classifier")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of loss_active in joint training")
    ap.add_argument("--save_clf", action="store_true", help="joint: BHAR evaluates our classifier, no finetuning")
    ap.add_argument("--init", choices=["bhar", "zero"], default="bhar")
    ap.add_argument("--tag", default="e2e", help="same TAG as for bhar_posttrain.sh")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out_dir = out_policy_dir(args.tag)
    if out_dir == BHAR_POLICY:
        sys.exit("this tag would overwrite BHAR's own policy files")
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    logger = init_logger(f"pamap2/tinyhar/{args.tag}/policy_train_seed{args.seed}")
    logger.info(str(vars(args)))

    cfg = SimConfig(feas_kappa=args.kappa)
    ehs = EnergyHarvestingSensor(EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                 cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                 cfg.policy_frequency, cfg.init_energy)
    os.makedirs(out_dir, exist_ok=True)

    best_vals = []
    for s in args.subjects:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        name = f"[{s}]_seed{args.seed}"
        (X, y), (Xv, yv) = sequences(s, args.seed)
        val = [torch.tensor(a)[None].to(dev) for a in (Xv, yv, harvest(Xv, ehs))]

        if args.init == "bhar":
            with open(os.path.join(BHAR_POLICY, f"policy_{name}.pkl"), "rb") as f:
                th0 = pickle.load(f)["best"]
            th0 = [[float(v) for v in th0[bp]] for bp in BODY_PARTS]
            clf_path = os.path.join(BHAR_CLF, f"finetuned_classifier_{name}.pth")
        else:
            th0 = [[0.0, 0.0] for _ in BODY_PARTS]
            clf_path = os.path.join(PRETRAINED_CLF, f"{name}.pth")
        model = model_builder(**KW).float().to(dev)
        # finetuned checkpoints hold BHAR's wrapper (TemporalContextModel): drop its prefix
        sd = {k.removeprefix("multisensor_model."): v
              for k, v in torch.load(clf_path, map_location=dev)["model_state_dict"].items()}
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys or len(res.missing_keys) == len(model.state_dict()):
            sys.exit(f"classifier checkpoint does not match TinyHAR: {res}")
        if res.missing_keys:
            logger.info(f"not in checkpoint (kept at init, as in BHAR): {res.missing_keys}")
        model.eval()                                   # no dropout, also when training it jointly
        for p in model.parameters():
            p.requires_grad_(args.clf_lr > 0)

        def clf(x, ages):
            # cuDNN's LSTM has no backward in eval mode
            with torch.backends.cudnn.flags(enabled=False):
                return model(x.float(), ages.float()).double()

        def val_acc(theta):
            """Held-prediction accuracy and packets/s/sensor on the whole validation sequence."""
            pv = DiffConservativePolicy(torch.arange(3), 3, init=theta, scale=args.scale, noise=False).to(dev)
            with torch.no_grad():
                o = run_e2e(*val, pv, clf, cfg, len(KW["activities"]), train=False, counterfactual=False)
            return (o.preds[0] == yv).mean(), o.sim.a.sum().item() / (3 * len(yv) / cfg.fs)

        pol = DiffConservativePolicy(torch.arange(3), 3, init=th0, scale=args.scale, noise=True).to(dev)
        groups = [dict(params=pol.parameters(), lr=args.lr)]
        if args.clf_lr > 0:
            groups.append(dict(params=model.parameters(), lr=args.clf_lr))
        opt = torch.optim.Adam(groups)
        logger.info(f"subject {s}: train {len(y) / 25 / 60:.0f} min, val {len(yv) / 25 / 60:.0f} min, "
                    f"init theta {th0}, classifier {clf_path}")
        acc, pkt = val_acc(th0)
        best = (acc, -1, th0, copy.deepcopy(model.state_dict()))
        logger.info(f"  init: val_acc={acc:.3f} pkt/s={pkt:.3f}")
        for ep in range(args.epochs):
            losses, accs = [], []
            for _ in range(args.segments // args.batch_size):
                xb, yb, hb = (t.to(dev) for t in sample_batch(X, y, ehs, rng, args.batch_size))
                pol.sensor_index = torch.arange(3, device=dev).repeat(len(yb))
                o = run_e2e(xb, yb, hb, pol, clf, cfg, len(KW["activities"]), train=True)
                opt.zero_grad()
                loss = o.loss_task + (args.lam * o.loss_active if args.clf_lr > 0 else 0)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    pol.theta.clamp_(-1000, 1000)      # BHAR's parameter bounds
                losses.append(o.loss_task.item())
                accs.append((o.preds == yb.cpu().numpy()).mean())
            logger.info(f"  epoch {ep}: loss_task={np.mean(losses):.3f} train_acc={np.mean(accs):.3f} "
                        f"theta={pol.theta.detach().cpu().numpy().round(1).tolist()}")
            if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
                theta = pol.theta.detach().cpu().numpy().tolist()
                acc, pkt = val_acc(theta)
                logger.info(f"  epoch {ep}: val_acc={acc:.3f} pkt/s={pkt:.3f}")
                if acc > best[0]:
                    best = (acc, ep, theta, copy.deepcopy(model.state_dict()))

        with open(os.path.join(out_dir, f"policy_{name}.pkl"), "wb") as f:
            pickle.dump({"best": dict(zip(BODY_PARTS, best[2])), "init": dict(zip(BODY_PARTS, th0)),
                         "best_val_acc": float(best[0]), "best_epoch": best[1]}, f)
        if args.save_clf:
            d = os.path.join(ART, "saved_data", "checkpoints", "pamap2", "tinyhar", args.tag,
                             "conservative-asynchronous_multisensor_time_context")
            os.makedirs(d, exist_ok=True)
            torch.save({"model_state_dict": {"multisensor_model." + k: v for k, v in best[3].items()}},
                       os.path.join(d, f"finetuned_classifier_{name}.pth"))
        best_vals.append(best[0])
        logger.info(f"subject {s}: saved theta {best[2]} (val_acc {best[0]:.3f}, epoch {best[1]})")
    logger.info(f"mean best val_acc over subjects {args.subjects}: {np.mean(best_vals):.4f}")


if __name__ == "__main__":
    main()
