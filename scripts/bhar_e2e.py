"""Train BHAR's conservative policy parameters with gradients instead of pattern search,
inside BHAR's own setup (TinyHAR; PAMAP2 by default, --dataset for DSADS, RWHAR, Opportunity), and save them where BHAR's code will pick them up.

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

Score-function gradient (--rf > 0): the straight-through gradient only sees the immediate
effect of a packet; the energy it spends shows up later, when some future packet is no longer
feasible, and almost none of that reaches theta. With --rf, each segment is simulated --rf_k
times with random send decisions (P(send) = sigmoid(logit)), and the decisions of runs that end
with a lower loss over the whole segment are made more likely (REINFORCE with a leave-one-out
baseline). This credits every decision with its full later effect, energy included.

Alternating training (--rounds R > 0): the frozen classifier decides which theta looks good, and
the pretrained one (never trained on old packets) favours opportunistic sending. So R times:
finetune the classifier on the packets of the current theta with BHAR's own finetuning (same data,
epochs and lr as train_har_policy.py), then train theta for --epochs against it. With
--rand_ft N the first finetuning also uses the packets of N random thetas, so the classifier starts
out used to fresh and old packets alike instead of favouring the policy it was finetuned on.
Works for --policy neural too: BHAR replays the current network through scripts/bhar_hook.py.

Packet penalty (--pkt w > 0): adds w * packets/s/sensor (through the STE decisions) to the loss and
validation keeps the theta with the lowest val loss_task + w * packets/s, the same objective as
training (until 2026-10-25 it used val_acc - w * packets/s, which mixed units). Sweeping w
traces an accuracy-vs-packets curve to compare with BHAR's single point.

Worst-subject training (--dro eta > 0, group DRO, Sagawa et al. 2020): validation comes from the
same 7 people as training, so a rule that suits some of them and fails others still looks good on
average, and fails on the new test person (EXPERIMENTS.md, experiments 23-24). With --dro each
batch takes the same number of segments from every training subject, the loss is the weighted
mean of the per-subject losses, the weight of a subject growing by exp(eta * its loss) at every
step (so the worst subjects dominate), and validation keeps the theta whose worst subject (on
that subject's own validation data) has the lowest loss. Same data and split as BHAR.

Neural policy (--policy neural): a small per-sensor network (diffbhar/policies.py NeuralPolicy)
replaces the two-parameter rule. It first imitates the --init theta for --distill batches, so it
starts where BHAR is, then trains like theta. BHAR replays it through scripts/bhar_hook.py, which
bhar_posttrain.sh always runs (other policies are untouched).

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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "external", "BHAR_codes_artifact")
sys.path[:0] = [ROOT, ART]

from datasets.dataset import HARClassifierDataset, generate_activity_sequence  # noqa: E402
from energy_harvesting.energy_harvest import EnergyHarvester  # noqa: E402
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor  # noqa: E402
from models.model_builder import model_builder  # noqa: E402
from utils.setup_funcs import DATA_ROOT, init_logger  # noqa: E402
from datasets.sparse_data_utils import SparseHarDataset  # noqa: E402
from experiments.train import train  # noqa: E402
from models.sparse_wrappers import TemporalContextModel  # noqa: E402

import bhar_hook  # noqa: E402,F401  (BHAR's sparsify_data also runs our neural policies)
from diffbhar.pipeline import run_e2e  # noqa: E402
from diffbhar.policies import DiffConservativePolicy, NeuralPolicy  # noqa: E402
from diffbhar.sim import SimConfig, simulate  # noqa: E402

# settings of experiments/posttraining_scripts/conservative-asynchronous_multisensor_time_context-<dataset>.sh
# (energy model and sampling rate are the same for all four; ft_* = classifier finetuning, which
# --rounds runs with shuffled batches; BHAR's final finetuning samples by class weight except on PAMAP2)
DATASETS = {
    "pamap2": dict(subjects=list(range(1, 9)), body_parts=["hand", "chest", "ankle"],
                   activities=list(range(12)), val_frac=0.2, ft_bs=32, ft_lr=3e-4),
    "dsads": dict(subjects=list(range(1, 9)), body_parts=["torso", "right_arm", "left_arm", "right_leg", "left_leg"],
                  activities=list(range(19)), val_frac=0.1, ft_bs=32, ft_lr=3e-4),
    "rwhar": dict(subjects=[1, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15],
                  body_parts=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"],
                  activities=list(range(8)), val_frac=0.25, ft_bs=32, ft_lr=3e-4),
    "opportunity": dict(subjects=[1, 2, 3, 4], body_parts=["BACK", "RUA", "RLA", "LUA", "LLA", "L-SHOE", "R-SHOE"],
                        activities=list(range(5)), val_frac=0.1, ft_bs=16, ft_lr=1e-3),
}
SEG = 100 * 25                     # BHAR's policy training segment, 100 s


def set_dataset(name):
    """Point the module-level settings and paths at one of BHAR's datasets."""
    global DATASET, BODY_PARTS, KW, SUBJECTS, FT, CKPT, BHAR_POLICY, BHAR_CLF, PRETRAINED_CLF
    d = DATASETS[name]
    DATASET, BODY_PARTS, SUBJECTS, FT = name, d["body_parts"], d["subjects"], (d["ft_bs"], d["ft_lr"])
    KW = dict(dataset=name, architecture="tinyhar", sensors=["acc"], body_parts=BODY_PARTS,
              activities=d["activities"], val_frac=d["val_frac"], window_size=8)
    CKPT = os.path.join(ART, "saved_data", "checkpoints", name)
    BHAR_POLICY = os.path.join(CKPT, "conservative-asynchronous_single_sensor")
    BHAR_CLF = os.path.join(CKPT, "tinyhar", "conservative-asynchronous_multisensor_time_context")
    PRETRAINED_CLF = os.path.join(ART, "saved_data_cached", "checkpoints", name, "tinyhar",
                                  "multisensor_classifier_window_8_acc")


set_dataset("pamap2")


def out_policy_dir(tag):
    """Where BHAR looks for the policy when run with logging prefix <tag>/<method>:
    two levels above pamap2/tinyhar/<tag>/<method> (train_har_policy.py, dataset_level_dir)."""
    level = os.path.dirname(os.path.dirname(os.path.join(DATASET, "tinyhar", tag, "method")))
    return os.path.join(ART, "saved_data", "checkpoints", level, "conservative-asynchronous_single_sensor")


def sequences(test_subject, seed, test=False, per_subject=False):
    """Train and validation (and test) sequences, each ([b, L, 3] raw data, as BHAR uses it, and
    labels), generated as in train_har_policy.py (same seed, in the order train, val, test).
    per_subject: then also one train and one validation sequence per training subject (made after
    BHAR's, so those stay the same), as two lists."""
    kw = dict(KW, dataset_dir=os.path.join(os.path.expanduser(DATA_ROOT[DATASET]), "preprocessed_data"),
              subjects=[s for s in SUBJECTS if s != test_subject], normalize=False)
    dss = [HARClassifierDataset(**kw, train=True, val=False), HARClassifierDataset(**kw, train=False, val=True)]
    if test:
        dss.append(HARClassifierDataset(**dict(kw, subjects=[test_subject]), train=False, val=False))
    cmap = dss[0].dataset_info["sensor_channel_map"]
    np.random.seed(seed)

    def make(data, labels, ds):
        X, y = generate_activity_sequence(data, labels, 10, 30, 25)
        per_bp = [X[:, np.isin(ds.active_channels, cmap[bp]["acc"])] for bp in BODY_PARTS]
        return np.stack(per_bp), y.astype(np.int64)

    out = [make(np.concatenate(list(ds.raw_data.values())), np.concatenate(list(ds.raw_labels.values())), ds)
           for ds in dss]
    if per_subject:
        out += [[make(ds.raw_data[k], ds.raw_labels[k], ds) for k in ds.raw_data] for ds in dss[:2]]
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


class Recorder(torch.nn.Module):
    """Wraps a policy and keeps the features it was called with."""

    def __init__(self, policy):
        super().__init__()
        self.policy, self.feats = policy, []

    def forward(self, f):
        self.feats.append(f)
        return self.policy(f)

    def binarize(self, logit, train=True):
        return self.policy.binarize(logit, train)


def distill(student, teacher, X, y, ehs, cfg, rng, dev, steps, batch_size):
    """Fit the neural policy to the teacher's decisions. The first half of the batches runs the
    teacher; the second half runs the student and labels the states it reaches with the teacher's
    decision (DAgger; the teacher sees the same features), so small errors cannot drift the
    sensor into states the student never learned. Sends are rare, so sends and holds get equal
    weight. Returns the share of the teacher's sends and of its holds that the student repeats
    (where the sensor could send) in the last batch."""
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    for it in range(steps):
        _, _, hb = sample_batch(X, y, ehs, rng, batch_size)
        idx = torch.arange(len(BODY_PARTS), device=dev).repeat(batch_size)
        teacher.sensor_index = student.sensor_index = idx
        rec = Recorder(teacher if it < steps // 2 else student)
        with torch.no_grad():
            o = simulate(hb.to(dev).reshape(-1, hb.shape[-1]), rec, cfg, train=False)
            feats = rec.feats[:o.logit.shape[1]]
            tl = torch.stack([teacher(f) for f in feats], 1)
        lg = torch.stack([student(f) for f in feats], 1)
        pos, neg = (tl >= 0) & o.feasible, (tl < 0) & o.feasible
        w = pos / pos.sum().clamp(min=1) + neg / neg.sum().clamp(min=1)
        loss = (F.binary_cross_entropy_with_logits(lg, torch.sigmoid(tl), reduction="none") * w).sum() / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    sends = ((lg >= 0) & pos).sum().item() / max(pos.sum().item(), 1)
    holds = ((lg < 0) & neg).sum().item() / max(neg.sum().item(), 1)
    return sends, holds


def snapshot(pol):
    """Parameters of a policy, without the row -> sensor map."""
    if isinstance(pol, NeuralPolicy):
        return {k: v.detach().clone() for k, v in pol.state_dict().items() if k != "sensor_index"}
    return pol.theta.detach().cpu().numpy().tolist()


def finetune(model, thetas, train_seq, val_seq, ehs, logname, logger, dev):
    """BHAR's classifier finetuning (train_har_policy.py) on the packets of the thetas, in place.
    A theta is [theta_e, theta_p] per sensor, or ["neural", "<file.pt>#<sensor>"] (scripts/bhar_hook.py)."""
    loaders = []
    for (X, y), shuffle, bs in ((train_seq, True, FT[0]), (val_seq, False, 128)):
        data = {bp: X[i] for i, bp in enumerate(BODY_PARTS)}
        dss = []
        for theta in thetas:
            pol = [f"conservative_{t[0]}_{t[1]}" if t[0] == "neural" else
                   f"conservative_{float(t[0])}_{float(t[1])}" for t in theta]
            idxs = {bp: ehs.sparsify_data(pol[i], X[i]) for i, bp in enumerate(BODY_PARTS)}
            dss.append(SparseHarDataset(data, y, idxs))
        loaders.append(torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(dss), batch_size=bs,
                                                   shuffle=shuffle, drop_last=True, num_workers=4))
    wrapped = TemporalContextModel(model, dev)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=FT[1])
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    train(model=wrapped, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=opt, train_logname=logname,
          epochs=10, ese=10, device=dev, train_loader=loaders[0], val_loader=loaders[1], logger=logger,
          lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10), log_freq=200)
    torch.set_default_dtype(dtype)
    # train() keeps the checkpoint with the best validation F1, as BHAR then loads it
    path = os.path.join(ART, "saved_data", "checkpoints", logname + ".pth")
    wrapped.load_state_dict(torch.load(path, map_location=dev)["model_state_dict"])
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", choices=list(DATASETS), default="pamap2")
    ap.add_argument("--subjects", type=int, nargs="+", default=None, help="test subjects (default: all)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--segments", type=int, default=64, help="100 s segments per epoch")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=5.0, help="policy temperature: logit = (Delta - tau) / scale")
    ap.add_argument("--kappa", type=float, default=5e-6, help="width (J) of the energy-feasibility surrogate")
    ap.add_argument("--no_delta_grad", action="store_true",
                    help="no gradient through the delta reset (it explodes once theta_p binds, scripts/grad_check.py)")
    ap.add_argument("--val_every", type=int, default=2, help="epochs between validation scores")
    ap.add_argument("--clf_lr", type=float, default=0.0, help="> 0: joint training of theta and the classifier")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of loss_active in joint training")
    ap.add_argument("--save_clf", action="store_true", help="joint: BHAR evaluates our classifier, no finetuning")
    ap.add_argument("--ste", type=float, default=1.0, help="weight of the straight-through gradient")
    ap.add_argument("--rf", type=float, default=0.0, help="> 0: weight of the score-function gradient")
    ap.add_argument("--rf_k", type=int, default=4, help="random runs per segment for --rf")
    ap.add_argument("--rounds", type=int, default=0,
                    help="> 0: alternate BHAR's classifier finetuning and --epochs of theta training")
    ap.add_argument("--rand_ft", type=int, default=0,
                    help="--rounds: random thetas (each parameter in [0, 400]) added to the first finetuning")
    ap.add_argument("--pkt", type=float, default=0.0,
                    help="weight of a packets/s/sensor penalty; validation then selects on loss_task + pkt * pkt/s")
    ap.add_argument("--select", choices=["acc", "loss"], default="acc",
                    help="validation selection without --pkt: val_acc, or val loss_task (the training objective)")
    ap.add_argument("--dro", type=float, default=0.0,
                    help="> 0: worst-subject (group DRO) training with this step size; validation then "
                         "selects on the worst training subject's val loss_task (+ pkt * pkt/s)")
    ap.add_argument("--init", choices=["bhar", "zero"], default="bhar",
                    help="bhar: BHAR's theta and finetuned classifier; zero: --init_theta, pretrained classifier")
    ap.add_argument("--init_theta", type=float, nargs=2, default=[0.0, 0.0],
                    help="--init zero: (theta_e, theta_p) for every sensor, no BHAR search involved")
    ap.add_argument("--policy", choices=["conservative", "neural"], default="conservative")
    ap.add_argument("--hidden", type=int, default=16, help="neural: hidden units per sensor")
    ap.add_argument("--neural_lr", type=float, default=1e-3, help="neural: learning rate")
    ap.add_argument("--distill", type=int, default=100,
                    help="neural: batches spent imitating the --init theta before training")
    ap.add_argument("--tag", default="e2e", help="same TAG as for bhar_posttrain.sh")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_dataset(args.dataset)
    args.subjects = args.subjects or SUBJECTS
    N = len(BODY_PARTS)
    out_dir = out_policy_dir(args.tag)
    if out_dir == BHAR_POLICY:
        sys.exit("this tag would overwrite BHAR's own policy files")
    if args.dro > 0 and args.rf > 0:
        sys.exit("--dro does not support --rf")
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    logger = init_logger(f"{DATASET}/tinyhar/{args.tag}/policy_train_seed{args.seed}")
    logger.info(str(vars(args)))

    cfg = SimConfig(feas_kappa=args.kappa, delta_grad=not args.no_delta_grad)
    ehs = EnergyHarvestingSensor(EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5),
                                 cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                 cfg.policy_frequency, cfg.init_energy)
    os.makedirs(out_dir, exist_ok=True)

    best_vals = []
    for s in args.subjects:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        name = f"[{s}]_seed{args.seed}"
        (X, y), (Xv, yv), *groups = sequences(s, args.seed, per_subject=args.dro > 0)
        val = [torch.tensor(a)[None].to(dev) for a in (Xv, yv, harvest(Xv, ehs))]
        if args.dro > 0:
            train_g = groups[0]
            val_g = [[torch.tensor(a)[None].to(dev) for a in (xg, yg, harvest(xg, ehs))] for xg, yg in groups[1]]
            q = torch.ones(len(train_g), device=dev) / len(train_g)      # subject weights

        if args.init == "bhar":
            with open(os.path.join(BHAR_POLICY, f"policy_{name}.pkl"), "rb") as f:
                th0 = pickle.load(f)["best"]
            th0 = [[float(v) for v in th0[bp]] for bp in BODY_PARTS]
            clf_path = os.path.join(BHAR_CLF, f"finetuned_classifier_{name}.pth")
        else:
            th0 = [list(args.init_theta) for _ in BODY_PARTS]
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

        def make_policy(theta, noise):
            if args.policy == "neural":
                p = NeuralPolicy(torch.arange(N), N, hidden=args.hidden, noise=noise)
                if theta is not None:
                    p.load_state_dict(theta, strict=False)
                return p.to(dev)
            return DiffConservativePolicy(torch.arange(N), N, init=theta, scale=args.scale, noise=noise).to(dev)

        def val_acc(theta):
            """Held-prediction accuracy, packets/s/sensor and loss_task on the whole validation sequence."""
            pv = make_policy(theta, False)
            with torch.no_grad():
                o = run_e2e(*val, pv, clf, cfg, len(KW["activities"]), train=False, counterfactual=False)
                vl = o.loss_task.item()
                if args.dro > 0:   # the worst training subject, on its own validation data
                    vl = max(run_e2e(*v, pv, clf, cfg, len(KW["activities"]), train=False,
                                     counterfactual=False).loss_task.item() for v in val_g)
            return (o.preds[0] == yv).mean(), o.sim.a.sum().item() / (N * len(yv) / cfg.fs), vl

        def score(acc, pkt, loss):
            """Validation selection; with --pkt the training objective itself, so both agree."""
            return acc if args.pkt == 0 and args.select == "acc" and args.dro == 0 else -(loss + args.pkt * pkt)

        pol = make_policy(None if args.policy == "neural" else th0, True)
        if args.policy == "neural":
            teacher = DiffConservativePolicy(torch.arange(N), N, init=th0, scale=args.scale, noise=False).to(dev)
            sends, holds = distill(pol, teacher, X, y, ehs, cfg, rng, dev, args.distill, args.batch_size)
            logger.info(f"  neural policy imitates init theta: repeats {sends:.3f} of its sends, {holds:.3f} of its holds")
        start = snapshot(pol)
        groups = [dict(params=pol.parameters(), lr=args.lr if args.policy == "conservative" else args.neural_lr)]
        if args.clf_lr > 0:
            groups.append(dict(params=model.parameters(), lr=args.clf_lr))
        opt = torch.optim.Adam(groups)
        logger.info(f"subject {s}: train {len(y) / 25 / 60:.0f} min, val {len(yv) / 25 / 60:.0f} min, "
                    f"init theta {th0}, classifier {clf_path}")
        acc, pkt, vl = val_acc(start)
        best = (score(acc, pkt, vl), -1, start, copy.deepcopy(model.state_dict()))
        logger.info(f"  init: val_acc={acc:.3f} pkt/s={pkt:.3f} loss={vl:.3f}")
        for ep in range(args.epochs * max(args.rounds, 1)):
            if args.rounds > 0 and ep % args.epochs == 0:
                theta = snapshot(pol)
                if args.policy == "neural":     # BHAR replays it from a file, as in the final evaluation
                    pt = os.path.join(out_dir, f"policy_{name}_round.pt")
                    torch.save({k: v.cpu() for k, v in theta.items()}, pt)
                    thetas = [[["neural", f"{pt}#{i}"] for i in range(len(BODY_PARTS))]]
                else:
                    thetas = [theta]
                if ep == 0:
                    thetas += np.random.default_rng([args.seed, s]).uniform(0, 400, (args.rand_ft, N, 2)).tolist()
                finetune(model, thetas, (X, y), (Xv, yv), ehs,
                         f"{DATASET}/tinyhar/{args.tag}/alternating_clf_{name}", logger, dev)
                for p in model.parameters():
                    p.requires_grad_(args.clf_lr > 0)
                acc, pkt, vl = val_acc(theta)
                logger.info(f"  round {ep // args.epochs}: classifier finetuned, val_acc={acc:.3f} pkt/s={pkt:.3f}")
                if score(acc, pkt, vl) > best[0]:
                    best = (score(acc, pkt, vl), ep - 1, theta, copy.deepcopy(model.state_dict()))
            losses, accs = [], []
            k = args.rf_k if args.rf > 0 else 1
            for _ in range(args.segments // args.batch_size):
                if args.dro > 0:
                    n = max(1, args.batch_size // len(train_g))
                    xb, yb, hb = (torch.cat(t).to(dev) for t in
                                  zip(*(sample_batch(xg, yg, ehs, rng, n) for xg, yg in train_g)))
                else:
                    xb, yb, hb = (t.to(dev).repeat_interleave(k, 0)
                                  for t in sample_batch(X, y, ehs, rng, args.batch_size // k))
                pol.sensor_index = torch.arange(N, device=dev).repeat(len(yb))
                o = run_e2e(xb, yb, hb, pol, clf, cfg, len(KW["activities"]), train=True)
                opt.zero_grad()
                task = o.loss_task
                if args.dro > 0:
                    ls_g = o.loss_seq.view(len(train_g), -1).mean(1)
                    q = q * torch.exp(args.dro * ls_g.detach())
                    q = q / q.sum()
                    task = (q * ls_g).sum()
                loss = args.ste * task + (args.lam * o.loss_active if args.clf_lr > 0 else 0)
                if args.rf > 0:
                    ls = o.loss_seq.detach().view(-1, k)
                    adv = (ls - (ls.sum(1, keepdim=True) - ls) / (k - 1)).view(-1)
                    a, lg = o.sim.a.detach(), o.sim.logit
                    # log-probability of the decisions that mattered (sensor free and charged)
                    logp = o.sim.feasible * (a * F.logsigmoid(lg) + (1 - a) * F.logsigmoid(-lg))
                    loss = loss + args.rf * (adv * logp.view(len(yb), -1).sum(1)).mean()
                if args.pkt > 0:
                    loss = loss + args.pkt * o.sim.a.sum() / (o.sim.a.shape[0] * SEG / 25)
                loss.backward()
                opt.step()
                if args.policy == "conservative":
                    with torch.no_grad():
                        pol.theta.clamp_(-1000, 1000)      # BHAR's parameter bounds
                losses.append(o.loss_task.item())
                accs.append((o.preds == yb.cpu().numpy()).mean())
            logger.info(f"  epoch {ep}: loss_task={np.mean(losses):.3f} train_acc={np.mean(accs):.3f}"
                        + (f" subject_weights={np.round(q.cpu().numpy(), 2).tolist()}" if args.dro > 0 else "")
                        + (f" theta={np.round(snapshot(pol), 1).tolist()}" if args.policy == "conservative" else ""))
            if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
                theta = snapshot(pol)
                acc, pkt, vl = val_acc(theta)
                logger.info(f"  epoch {ep}: val_acc={acc:.3f} pkt/s={pkt:.3f} loss={vl:.3f}")
                if score(acc, pkt, vl) > best[0]:
                    best = (score(acc, pkt, vl), ep, theta, copy.deepcopy(model.state_dict()))

        saved = best[2]
        if args.policy == "neural":
            # BHAR turns best[bp] into "conservative_<0>_<1>"; scripts/bhar_hook.py runs these
            pt = os.path.join(out_dir, f"policy_{name}.pt")
            torch.save({k: v.cpu() for k, v in best[2].items()}, pt)
            saved = [["neural", f"{pt}#{i}"] for i in range(len(BODY_PARTS))]
        with open(os.path.join(out_dir, f"policy_{name}.pkl"), "wb") as f:
            pickle.dump({"best": dict(zip(BODY_PARTS, saved)), "init": dict(zip(BODY_PARTS, th0)),
                         "best_val_acc": float(best[0]), "best_epoch": best[1]}, f)
        if args.save_clf:
            d = os.path.join(ART, "saved_data", "checkpoints", DATASET, "tinyhar", args.tag,
                             "conservative-asynchronous_multisensor_time_context")
            os.makedirs(d, exist_ok=True)
            torch.save({"model_state_dict": {"multisensor_model." + k: v for k, v in best[3].items()}},
                       os.path.join(d, f"finetuned_classifier_{name}.pth"))
        best_vals.append(best[0])
        logger.info(f"subject {s}: saved theta {saved} (val_acc {best[0]:.3f}, epoch {best[1]})")
    logger.info(f"mean best val_acc over subjects {args.subjects}: {np.mean(best_vals):.4f}")


if __name__ == "__main__":
    main()
