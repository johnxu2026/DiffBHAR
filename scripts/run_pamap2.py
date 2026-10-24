"""PAMAP2 experiment: train the acquisition policy on the classification loss
instead of BHAR's active-region objective.

All methods use the same energy model (BHAR's harvester and sensor parameters),
the same policy class or none, and the same small classifier:

  opp   opportunistic policy (send whenever there is energy)
  bhar  conservative policy, theta chosen to maximise the active region R_pi, i.e. BHAR's
        objective (grid search stands in for BHAR's pattern search)
  e2e   same conservative policy, theta trained by gradient on the held-prediction
        cross-entropy through the differentiable simulator (DiffBHAR)
  joint same policy, theta and classifier trained together in one backward per batch

Each classifier is fine-tuned for the same number of epochs on its own policy's packets.
One leave-one-subject-out fold per test subject.

usage:  python scripts/run_pamap2.py --exp NAME --test_subjects 1 --seed 0
Everything goes to results/<date>_NAME/<run>.{log,json} and <run>_{summary,timeline}.png,
where <run> is the test subjects, the seed and any setting that differs from its default.
The first run downloads PAMAP2 (~650 MB) into data/ unless data/pamap2.zip already exists.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLBACKEND", "Agg")      # BHAR modules import pyplot; no window needed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "external", "BHAR_codes_artifact")
sys.path[:0] = [ART, ROOT]
from energy_harvesting.energy_harvest import EnergyHarvester            # noqa: E402
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor  # noqa: E402
from datasets.preprocess_raw_data import preprocess_PAMAP2              # noqa: E402
from datasets.dataset import generate_activity_sequence                 # noqa: E402
from diffbhar.sim import SimConfig                                      # noqa: E402
from diffbhar.policies import DiffConservativePolicy, OpportunisticPolicy  # noqa: E402
from diffbhar.pipeline import run_e2e, region_metrics                   # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
BODY_PARTS = ["hand", "chest", "ankle"]   # channel order of BHAR's PAMAP2 preprocessing
N_CLASSES = 12
SEG = 2500                                # training segment length (100 s at 25 Hz)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def find(top, name):
    for d, dirs, files in os.walk(top):
        if name in files or name in dirs:
            return os.path.join(d, name)
    return None


def get_pamap2(data_dir):
    """Download + unzip PAMAP2 if needed, run BHAR's preprocessing once, return its folder."""
    pre = find(data_dir, "preprocessed_data")
    if pre and os.path.exists(os.path.join(pre, "metadata.pickle")):
        return pre
    os.makedirs(data_dir, exist_ok=True)
    if find(data_dir, "subject101.dat") is None:
        z = os.path.join(data_dir, "pamap2.zip")
        if not os.path.exists(z):
            log("downloading PAMAP2 (~650 MB) ...")
            urllib.request.urlretrieve(URL, z + ".part")
            os.replace(z + ".part", z)
        log("unzipping ...")
        zipfile.ZipFile(z).extractall(data_dir)
        inner = find(data_dir, "PAMAP2_Dataset.zip")
        if inner:
            zipfile.ZipFile(inner).extractall(os.path.dirname(inner))
    # the zip also has an "Optional" folder with a few subjects; BHAR uses "Protocol"
    prot = find(data_dir, "Protocol")
    log(f"preprocessing {prot} with BHAR's preprocess_PAMAP2 (100 Hz -> 25 Hz) ...")
    preprocess_PAMAP2(prot)
    return os.path.join(prot, "preprocessed_data")


def sequence(pre, subjects):
    """BHAR's activity sequence: random 10-30 s segments of each activity, concatenated."""
    X = np.concatenate([np.load(os.path.join(pre, f"data_{s}.npy")) for s in subjects])
    y = np.concatenate([np.load(os.path.join(pre, f"labels_{s}.npy")) for s in subjects])
    X, y = generate_activity_sequence(X, y.astype(int), 10, 30, 25)
    n = len(X)
    while n > 0 and not X[n - 1].any():      # drop the unfilled tail of the output buffer
        n -= 1
    return X[:n], y[:n].astype(np.int64)


def harvest(X, cfg):
    """Raw acceleration [L, 9] -> harvested energy per step [3, L] (J), BHAR's model."""
    eh = EnergyHarvester(tx_thresh=cfg.tx_thresh, efficiency=0.5)
    ehs = EnergyHarvestingSensor(eh, cfg.packet_size, cfg.leakage, cfg.fs, cfg.max_energy,
                                 cfg.policy_frequency, cfg.init_energy)
    out = []
    for i in range(len(BODY_PARTS)):
        ehs.init_harvester(X[:, 3 * i:3 * i + 3])
        out.append(ehs.e_harvest.copy())
    return np.stack(out)


# ----------------------------------------------------------------------- classifier
class SmallMSCT(nn.Module):
    """MSC-T in miniature: per-sensor window statistics, scaled by BHAR's temporal gate
    exp(-relu(r * age + c)) so stale packets count less, then an MLP."""

    def __init__(self, b=3, hidden=64):
        super().__init__()
        self.b = b
        self.r = nn.Parameter(torch.full((b,), 0.02))
        self.c = nn.Parameter(torch.zeros(b))
        self.net = nn.Sequential(nn.Linear(b * 14, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_CLASSES))

    def forward(self, x, ages):                        # x [M, w, 3b], ages [M, b]
        M, w, _ = x.shape
        x = x.reshape(M, w, self.b, 3)
        mag = x.norm(dim=-1)
        feat = torch.cat([x.mean(1), x.std(1), x.amin(1), x.amax(1),
                          mag.mean(1)[..., None], mag.std(1)[..., None]], -1)   # [M, b, 14]
        gate = torch.exp(-torch.relu(self.r * ages.clamp_min(0) + self.c))
        return self.net((feat * gate[..., None]).flatten(1))


def pretrain(clf, Xn, y, w, epochs, lr=2e-3):
    """Dense pretraining: every sensor fresh (age 0), like BHAR's MSC pretraining."""
    starts = np.arange(0, len(y) - w - 1, w // 2)
    idx = torch.tensor(starts)[:, None] + torch.arange(w)[None]
    Xw = torch.tensor(Xn)[idx]                          # [M, w, 9]
    yw = torch.tensor(y)[starts + w - 1]                # last label in the window, as in BHAR
    dev = next(clf.parameters()).device
    Xw, yw = Xw.to(dev), yw.to(dev)
    A = torch.zeros(len(starts), clf.b, dtype=Xw.dtype, device=dev)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        for bi in torch.randperm(len(starts)).split(512):
            loss = nn.functional.cross_entropy(clf(Xw[bi], A[bi]), yw[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()


# --------------------------------------------------------------------------- runner
class Fold:
    def __init__(self, Xn_tr, y_tr, H_tr, cfg, maxp, n_train_segments, bs=16, dev="cpu"):
        n = min(len(y_tr) // SEG, n_train_segments)
        cut = lambda a: np.stack([a[..., i * SEG:(i + 1) * SEG] for i in range(n)])  # noqa
        self.X = torch.tensor(cut(Xn_tr.T.reshape(3, 3, -1))).permute(0, 1, 3, 2).to(dev)  # [n,b,SEG,3]
        self.Y = torch.tensor(cut(y_tr)).to(dev)
        self.H = torch.tensor(cut(H_tr)).to(dev)
        self.cfg, self.maxp, self.bs, self.n = cfg, maxp, bs, n

    def batches(self, shuffle=True):
        idx = torch.randperm(self.n) if shuffle else torch.arange(self.n)
        return idx.split(self.bs)

    def run(self, pol, clf, idx, train):
        pol, mp = prep(pol, len(idx), self.maxp)
        return run_e2e(self.X[idx], self.Y[idx], self.H[idx], pol, clf, self.cfg, N_CLASSES,
                       max_p=mp, train=train)


def prep(pol, B, maxp):
    """Rows are (sequence, sensor) pairs: point each row at its sensor's parameters."""
    sid = torch.arange(3, device=maxp.device).repeat(B)
    if hasattr(pol, "sensor_index"):
        pol.sensor_index = sid
    return pol, maxp[sid]


def finetune(fold, pol, clf, epochs, lr=1e-3):
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in fold.batches():
            o = fold.run(pol, clf, idx, train=False)
            opt.zero_grad()
            o.loss_active.backward()
            opt.step()


def active_region(fold, pol, n_seg):
    dummy = lambda X, A: torch.zeros(X.shape[0], N_CLASSES, dtype=X.dtype, device=X.device)  # noqa: E731
    vals = []
    with torch.no_grad():
        for idx in torch.arange(min(n_seg, fold.n)).split(fold.bs):
            o = fold.run(pol, dummy, idx, train=False)
            M = ((1 - torch.prod(1 - o.sim.a.reshape(len(idx), 3, -1), 1)) > 0.5).cpu()
            for r in range(len(idx)):
                m = region_metrics(np.zeros(SEG, int), np.zeros(SEG, int), M[r].numpy(),
                                   o.arrivals.cpu().numpy())
                vals.append(1 - m["passive_region"])
    return float(np.mean(vals))


def macro_f1(y, p):
    f = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def evaluate(pol, clf, Xn_te, y_te, H_te, cfg, maxp):
    dev = maxp.device
    x = torch.tensor(Xn_te.T.reshape(3, 3, -1)).permute(0, 2, 1)[None].to(dev)     # [1,b,L,3]
    pol, mp = prep(pol, 1, maxp)
    with torch.no_grad():
        o = run_e2e(x, torch.tensor(y_te)[None].to(dev), torch.tensor(H_te)[None].to(dev), pol, clf,
                    cfg, N_CLASSES, max_p=mp, train=False)
    a = o.sim.a.reshape(3, -1).cpu().numpy() > 0.5
    M = a.any(0)
    arrivals = o.arrivals.cpu().numpy()
    m = region_metrics(y_te, o.preds[0], M, arrivals)
    L = len(y_te)
    res = dict(acc=float(m["acc"]), f1=macro_f1(y_te, o.preds[0]),
               active_region=float(1 - m["passive_region"]),
               active_err=float(m["active_err"]), passive_err=float(m["passive_err"]),
               pkt_per_s=float(a.sum() / (3 * L / cfg.fs)),
               pkt_per_s_sensor={bp: float(a[i].sum() / (L / cfg.fs)) for i, bp in enumerate(BODY_PARTS)})
    trace = dict(preds=o.preds[0], sent=a, ticks=o.sim.ticks.cpu().numpy(), arrivals=arrivals)
    return res, trace


# ----------------------------------------------------------------------------- main
def run_fold(pre, test_subject, args, cfg):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    train_subjects = [s for s in range(1, 9) if s != test_subject]
    X_tr, y_tr = sequence(pre, train_subjects)
    X_te, y_te = sequence(pre, [test_subject])
    log(f"subject {test_subject}: train {len(y_tr) / 25 / 60:.0f} min, test {len(y_te) / 25 / 60:.0f} min")

    H_tr, H_te = harvest(X_tr, cfg), harvest(X_te, cfg)
    # causal normaliser of the harvest rate: per-sensor constant from the training data
    # (BHAR uses the per-sequence maximum, which looks into the future)
    maxp = torch.tensor([np.convolve(h, np.ones(25), "valid").max() for h in H_tr]).to(dev)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-5
    Xn_tr, Xn_te = (X_tr - mu) / sd, (X_te - mu) / sd
    fold = Fold(Xn_tr, y_tr, H_tr, cfg, maxp, args.train_segments, dev=dev)

    clf0 = SmallMSCT().double().to(dev)
    pretrain(clf0, Xn_tr, y_tr, cfg.packet_size, args.pretrain_epochs)
    log("classifier pretrained on dense data")

    def tuned(pol):
        clf = SmallMSCT().double().to(dev)
        clf.load_state_dict(clf0.state_dict())
        finetune(fold, pol, clf, args.finetune_epochs)
        return clf

    out, traces, theta = {}, {}, {}
    # 1) opportunistic
    opp = OpportunisticPolicy()
    clf_opp = tuned(opp)
    out["opp"], traces["opp"] = evaluate(opp, clf_opp, Xn_te, y_te, H_te, cfg, maxp)
    log(f"opp   {out['opp']}")

    # 2) BHAR objective: maximise the active region (no classifier involved)
    best = None
    for th in itertools.product([0, 100, 300, 600, 1000], [0, 10, 25, 50]):
        pol = DiffConservativePolicy(torch.arange(3), 3, init=th, noise=False).to(dev)
        r = active_region(fold, pol, args.search_segments)
        if best is None or r > best[0]:
            best = (r, th)
    bhar = DiffConservativePolicy(torch.arange(3), 3, init=best[1], noise=False).to(dev)
    theta["bhar"] = bhar.theta.detach().cpu().numpy().tolist()
    clf_bhar = tuned(bhar)
    out["bhar"], traces["bhar"] = evaluate(bhar, clf_bhar, Xn_te, y_te, H_te, cfg, maxp)
    log(f"bhar  theta={best[1]} train R_pi={best[0]:.3f}  {out['bhar']}")

    # 3) DiffBHAR: same policy class, theta trained on the held-prediction loss
    #    (classifier frozen at BHAR's, so only the policy objective differs)
    e2e = DiffConservativePolicy(torch.arange(3), 3, init=best[1], noise=True).to(dev)
    for p in clf_bhar.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(e2e.parameters(), lr=args.policy_lr)
    for ep in range(args.policy_epochs):
        tot = []
        for idx in fold.batches():
            o = fold.run(e2e, clf_bhar, idx, train=True)
            opt.zero_grad()
            o.loss_task.backward()
            opt.step()
            with torch.no_grad():
                e2e.theta.clamp_(-1000, 2000)
            tot.append(o.loss_task.item())
        log(f"  e2e epoch {ep}: loss_task={np.mean(tot):.3f} theta={e2e.theta.detach().cpu().numpy().round(1).tolist()}")
    e2e_eval = DiffConservativePolicy(torch.arange(3), 3, init=e2e.theta.detach().tolist(), noise=False).to(dev)
    theta["e2e"] = e2e_eval.theta.detach().cpu().numpy().tolist()
    clf_e2e = tuned(e2e_eval)
    out["e2e"], traces["e2e"] = evaluate(e2e_eval, clf_e2e, Xn_te, y_te, H_te, cfg, maxp)
    log(f"e2e   {out['e2e']}")

    # 4) joint: theta and classifier trained together, one backward per batch on
    #    loss_task + lambda * loss_active; classifier starts from clf0 (dense) or clf_bhar
    jp = DiffConservativePolicy(torch.arange(3), 3, init=best[1], noise=True).to(dev)
    clf_j = SmallMSCT().double().to(dev)
    clf_j.load_state_dict((clf0 if args.joint_init == "dense" else clf_bhar).state_dict())
    opt = torch.optim.Adam([dict(params=jp.parameters(), lr=args.policy_lr),
                            dict(params=clf_j.parameters(), lr=args.clf_lr)])
    for ep in range(args.policy_epochs):
        tot, act = [], []
        for idx in fold.batches():
            o = fold.run(jp, clf_j, idx, train=True)
            opt.zero_grad()
            (o.loss_task + args.joint_lambda * o.loss_active).backward()
            opt.step()
            with torch.no_grad():
                jp.theta.clamp_(-1000, 2000)
            tot.append(o.loss_task.item())
            act.append(o.loss_active.item())
        log(f"  joint epoch {ep}: loss_task={np.mean(tot):.3f} loss_active={np.mean(act):.3f} "
            f"theta={jp.theta.detach().cpu().numpy().round(1).tolist()}")
    # theta is fixed (no noise) at test time: adapt the classifier to it with the usual budget
    joint_eval = DiffConservativePolicy(torch.arange(3), 3, init=jp.theta.detach().tolist(), noise=False).to(dev)
    theta["joint"] = joint_eval.theta.detach().cpu().numpy().tolist()
    finetune(fold, joint_eval, clf_j, args.finetune_epochs)
    out["joint"], traces["joint"] = evaluate(joint_eval, clf_j, Xn_te, y_te, H_te, cfg, maxp)
    log(f"joint {out['joint']}")
    return out, traces, theta, y_te, H_te


def plot(results, traces, y_te, H_te, path_prefix, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"opp": "Opportunistic", "bhar": "BHAR objective (max $R_\\pi$)", "e2e": "DiffBHAR (task loss)",
             "joint": "DiffBHAR joint"}
    cols = {"opp": "#9aa0a6", "bhar": "#4c78a8", "e2e": "#e45756", "joint": "#f58518"}

    # summary bars, averaged over test subjects
    keys = [("acc", "Accuracy"), ("f1", "Macro-F1"), ("active_region", "Active region $R_\\pi$"),
            ("pkt_per_s", "Packets / s / sensor")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (k, title) in zip(axs, keys):
        for i, m in enumerate(names):
            v = [r[m][k] for r in results.values()]
            ax.bar(i, np.mean(v), color=cols[m], yerr=np.std(v) if len(v) > 1 else None)
        ax.set_xticks(range(4), ["Opp", "BHAR", "DiffBHAR", "Joint"])
        ax.set_title(title)
    fig.suptitle(f"PAMAP2, test subjects {list(results)}")
    fig.tight_layout()
    fig.savefig(path_prefix + "_summary.png", dpi=150)

    # timeline of the first test subject around the first activity change after 60 s
    s = list(traces)[0]
    y, H, tr = y_te[s], H_te[s], traces[s]
    ch = np.nonzero(np.diff(y[1500:]))[0]
    t0 = 1500 + (ch[0] - 750 if len(ch) else 0)
    t1 = t0 + 3000
    t = np.arange(t0, t1) / cfg.fs
    fig, axs = plt.subplots(6, 1, figsize=(12, 8.2), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.4, 1, 1, 1, 1]))
    axs[0].step(t, y[t0:t1], where="post", color="k")
    axs[0].set_ylabel("activity")
    for i, bp in enumerate(BODY_PARTS):
        axs[1].plot(t, np.convolve(H[i], np.ones(25), "same")[t0:t1] * 1e6, label=bp)
    axs[1].set_ylabel("harvest\n(uJ/s)")
    axs[1].legend(loc="upper right", fontsize=7)
    for ax, m in zip(axs[2:], names):
        d = tr[m]
        for i in range(3):
            k = d["ticks"][d["sent"][i]]
            k = k[(k >= t0) & (k < t1)]
            ax.vlines(k / cfg.fs, i, i + 0.8, color=cols[m], lw=1)
        ok = d["preds"][t0:t1] == y[t0:t1]
        ax.fill_between(t, -1.2, -0.2, where=ok, color="#54a24b", step="post", lw=0)
        ax.fill_between(t, -1.2, -0.2, where=~ok, color="#e45756", alpha=0.6, step="post", lw=0)
        ax.set_yticks([-0.7, 0.4, 1.4, 2.4], ["correct?", *BODY_PARTS], fontsize=7)
        ax.set_ylabel(names[m], fontsize=8)
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path_prefix + "_timeline.png", dpi=150)


def git_version():
    """Commit of the code that produced a result, with a note if there were uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules")
    return git("rev-parse", "--short", "HEAD") + (" (uncommitted changes)" if dirty else "")


def run_name(args, ap):
    """Short file name: test subjects, seed, then only the settings that differ from the defaults."""
    s = sorted(args.test_subjects)
    name = ["all" if s == list(range(1, 9)) else "s" + "-".join(map(str, s)), f"seed{args.seed}"]
    for k, short in [("finetune_epochs", "ft"), ("policy_epochs", "ep"), ("policy_lr", "plr"),
                     ("clf_lr", "clr"), ("joint_lambda", "lam"), ("joint_init", "init"),
                     ("train_segments", "seg"), ("pretrain_epochs", "pre"), ("device", "")]:
        v = getattr(args, k)
        if v != ap.get_default(k):
            name.append(f"{short}{v:g}" if isinstance(v, float) else f"{short}{v}")
    return "_".join(name)


class Tee:
    """Write to the terminal and to the run's log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for st in self.streams:
            st.write(x)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "pamap2"))
    ap.add_argument("--exp", required=True, help="experiment name; outputs go to results/<date>_<exp>/")
    ap.add_argument("--out_dir", default=None, help="overrides results/<date>_<exp>/")
    ap.add_argument("--train_segments", type=int, default=64, help="100 s segments used for training")
    ap.add_argument("--search_segments", type=int, default=32, help="segments for BHAR's grid search")
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    ap.add_argument("--finetune_epochs", type=int, default=3)
    ap.add_argument("--policy_epochs", type=int, default=4)
    ap.add_argument("--policy_lr", type=float, default=10.0)
    ap.add_argument("--clf_lr", type=float, default=1e-3, help="classifier lr in joint training")
    ap.add_argument("--joint_lambda", type=float, default=1.0, help="weight of loss_active in joint training")
    ap.add_argument("--joint_init", choices=["dense", "bhar"], default="dense",
                    help="joint classifier start: dense-pretrained clf0 or BHAR-finetuned clf_bhar")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg = SimConfig()            # BHAR's PAMAP2 settings: w=8, 6.6 uW leak, 200 uJ cap, 50 uJ/packet

    if args.out_dir is None:
        args.out_dir = os.path.join(ROOT, "results", time.strftime("%Y-%m-%d") + "_" + args.exp)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, run_name(args, ap))
    logf = open(prefix + ".log", "w")
    sys.stdout, sys.stderr = Tee(sys.stdout, logf), Tee(sys.stderr, logf)

    pre = get_pamap2(args.data_dir)
    results, traces, ys, Hs, thetas = {}, {}, {}, {}, {}
    for s in args.test_subjects:
        results[s], traces[s], thetas[s], ys[s], Hs[s] = run_fold(pre, s, args, cfg)
    with open(prefix + ".json", "w") as f:
        json.dump(dict(args=vars(args), command=" ".join(sys.argv), git=git_version(),
                       results=results, theta=thetas), f, indent=1)
    plot(results, traces, ys, Hs, prefix, cfg)

    print("\n=== test results (mean over test subjects) ===")
    print(f"{'method':6s} {'acc':>6s} {'F1':>6s} {'R_pi':>6s} {'eps_a':>6s} {'eps_p':>6s} {'pkt/s':>6s}")
    for m in ["opp", "bhar", "e2e", "joint"]:
        v = lambda k: np.mean([results[s][m][k] for s in results])  # noqa: E731
        print(f"{m:6s} {v('acc'):6.3f} {v('f1'):6.3f} {v('active_region'):6.3f} "
              f"{v('active_err'):6.3f} {v('passive_err'):6.3f} {v('pkt_per_s'):6.3f}")
    print(f"\nsaved {prefix}.log/.json/_summary.png/_timeline.png")
    log("done")


if __name__ == "__main__":
    main()
