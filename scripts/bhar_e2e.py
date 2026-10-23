"""Train BHAR's conservative policy parameters with gradients instead of pattern search,
inside BHAR's own PAMAP2 + TinyHAR setup, and save them where BHAR's code will pick them up.

For each test subject: same train sequence as train_har_policy.py (same seed), theta starts
from BHAR's learned policy, the classifier is BHAR's finetuned TinyHAR (frozen), and theta is
trained on the held-prediction loss (diffbhar.pipeline.run_e2e) over random 100 s segments,
like BHAR's own policy training. Then run BHAR's finetuning and evaluation on the new theta:

    python scripts/bhar_e2e.py --seed 0
    bash scripts/bhar_posttrain.sh conservative 0 scratch e2e

Needs BHAR's scratch run first (bash scripts/bhar_posttrain.sh conservative SEED scratch).
Use the artifact's own venv.
"""
import argparse
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
# where BHAR looks for the policy when run with logging prefix e2e/... (see bhar_posttrain.sh)
OUT_POLICY = os.path.join(CKPT, "tinyhar", "conservative-asynchronous_single_sensor")


def train_sequence(test_subject, seed):
    """Train sequence [b, L, 3] (raw, as BHAR uses it) and labels, as in train_har_policy.py."""
    kw = dict(KW, dataset_dir=os.path.join(os.path.expanduser(DATA_ROOT["pamap2"]), "preprocessed_data"),
              subjects=[s for s in SUBJECTS if s != test_subject], normalize=False)
    ds = HARClassifierDataset(**kw, train=True, val=False)
    data = np.concatenate(list(ds.raw_data.values()))
    labels = np.concatenate(list(ds.raw_labels.values()))
    np.random.seed(seed)
    X, y = generate_activity_sequence(data, labels, 10, 30, 25)
    cmap = ds.dataset_info["sensor_channel_map"]
    per_bp = [X[:, np.isin(ds.active_channels, cmap[bp]["acc"])] for bp in BODY_PARTS]
    return np.stack(per_bp), y.astype(np.int64)


def sample_batch(X, y, ehs, rng, n):
    """n random 100 s segments; harvest computed per segment, like BHAR's policy training."""
    xs, ys, hs = [], [], []
    for _ in range(n):
        st = min(rng.integers(len(y)), len(y) - SEG)
        x = X[:, st:st + SEG]
        h = []
        for i in range(len(BODY_PARTS)):
            ehs.init_harvester(x[i])
            h.append(ehs.e_harvest.copy())
        xs.append(x)
        ys.append(y[st:st + SEG])
        hs.append(np.stack(h))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)), torch.tensor(np.stack(hs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS, help="test subjects")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--segments", type=int, default=64, help="100 s segments per epoch")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    logger = init_logger(f"pamap2/tinyhar/e2e/policy_train_seed{args.seed}")
    logger.info(str(vars(args)))

    cfg = SimConfig()
    ehs = EnergyHarvestingSensor(EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                 cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                 cfg.policy_frequency, cfg.init_energy)
    os.makedirs(OUT_POLICY, exist_ok=True)

    for s in args.subjects:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        name = f"[{s}]_seed{args.seed}"
        X, y = train_sequence(s, args.seed)

        with open(os.path.join(BHAR_POLICY, f"policy_{name}.pkl"), "rb") as f:
            th0 = pickle.load(f)["best"]
        th0 = [[float(v) for v in th0[bp]] for bp in BODY_PARTS]
        model = model_builder(**KW).float().to(dev)
        model.load_state_dict(torch.load(os.path.join(BHAR_CLF, f"finetuned_classifier_{name}.pth"),
                                         map_location=dev)["model_state_dict"], strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def clf(x, ages):
            return model(x.float(), ages.float()).double()

        pol = DiffConservativePolicy(torch.arange(3), 3, init=th0, noise=True).to(dev)
        opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
        logger.info(f"subject {s}: train {len(y) / 25 / 60:.0f} min, BHAR theta {th0}")
        for ep in range(args.epochs):
            losses = []
            for _ in range(args.segments // args.batch_size):
                xb, yb, hb = (t.to(dev) for t in sample_batch(X, y, ehs, rng, args.batch_size))
                pol.sensor_index = torch.arange(3, device=dev).repeat(len(yb))
                o = run_e2e(xb, yb, hb, pol, clf, cfg, len(KW["activities"]), train=True)
                opt.zero_grad()
                o.loss_task.backward()
                opt.step()
                with torch.no_grad():
                    pol.theta.clamp_(-1000, 1000)      # BHAR's parameter bounds
                losses.append(o.loss_task.item())
            logger.info(f"  epoch {ep}: loss_task={np.mean(losses):.3f} "
                        f"theta={pol.theta.detach().cpu().numpy().round(1).tolist()}")

        theta = pol.theta.detach().cpu().numpy().tolist()
        with open(os.path.join(OUT_POLICY, f"policy_{name}.pkl"), "wb") as f:
            pickle.dump({"best": dict(zip(BODY_PARTS, theta)), "bhar": dict(zip(BODY_PARTS, th0))}, f)
        logger.info(f"subject {s}: saved theta {theta}")


if __name__ == "__main__":
    main()
