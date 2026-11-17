"""Does BHAR's proxy reward track accuracy?  x = active-region fraction R_active (the reward BHAR's
pattern search maximizes, experiments/sensor_reward.py reward_type="active_region"), y = test accuracy,
one point per policy run on PAMAP2 (fixed-rule grid, BHAR, BHAR 16x budget, heuristic, ours, ...).

    python scripts/plot_proxy.py [--dataset pamap2]   ->  report/fig_proxy.{pdf,png} + table on stdout
"""
import argparse
import glob
import os

import matplotlib
import numpy as np
from scipy.stats import spearmanr

from bhar_cached import ART, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pamap2")
args = ap.parse_args()
base = os.path.join(ART, "saved_data", "results", args.dataset, "tinyhar")
dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(base, "**", "results_seed*.pickle"), recursive=True)})

rows = []
for d in dirs:
    name = os.path.relpath(d, base)
    if any(k in name for k in ("unconstrained", "ablation", "classifier", "eff")) or name.startswith("joint_s50/"):
        continue          # dense: no energy limit; eff*: other harvest efficiency; joint_s50: broken classifier
    res = load(d)
    per_seed = np.stack([np.mean(list(r.values()), axis=0) for r in res.values()])
    acc, r_act = per_seed[:, 1].mean(), per_seed[:, 2].mean()
    if name.startswith("conservative"):
        kind = "BHAR"
    elif name.startswith("opportunistic"):
        kind = "Opportunistic"
    elif name.startswith("heuristic"):
        kind = "Heuristic"
    elif name.startswith("bhar512"):
        kind = "BHAR, 16x search"
    elif name.startswith("jnn100/"):
        kind = "Ours"
    elif "notrain" in name:
        kind = "Fixed rule"
    else:
        kind = "Other trained"
    rows.append((name, kind, len(res), r_act, acc))

print(f"{'run':60s} {'kind':18s} {'seeds':>5s} {'R_active':>8s} {'acc':>6s}")
for n, k, s, r, a in sorted(rows, key=lambda x: -x[3]):
    print(f"{n:60s} {k:18s} {s:5d} {r:8.3f} {a:6.3f}")
R, A = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
rho, p = spearmanr(R, A)
best_r, best_a = rows[int(R.argmax())], rows[int(A.argmax())]
print(f"\n{len(rows)} runs  Spearman(R_active, acc) = {rho:.2f} (p = {p:.2g})")
print(f"best proxy:    {best_r[0]}  R_active {best_r[3]:.3f}  acc {best_r[4]:.3f}")
print(f"best accuracy: {best_a[0]}  R_active {best_a[3]:.3f}  acc {best_a[4]:.3f}")
HI = 0.79                                   # near-maximal proxy: where BHAR, the heuristic and ours lie
top = R >= HI
rho_hi, p_hi = spearmanr(R[top], A[top])
print(f"R_active >= {HI}: {top.sum()} runs, acc {A[top].min():.3f}..{A[top].max():.3f}, "
      f"Spearman {rho_hi:.2f} (p = {p_hi:.2g})")

style = {"Fixed rule": ("o", "0.6", 3), "Other trained": ("o", "0.75", 3),
         "Opportunistic": ("v", "k", 4), "Heuristic": ("D", "tab:green", 3.5),
         "BHAR, 16x search": ("*", "tab:orange", 6), "BHAR": ("*", "tab:red", 7), "Ours": ("s", "tab:blue", 4)}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": .6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.45, 1.57))
ax.axvspan(HI, R.max() + .005, color="0.93", lw=0)
ax.text(R.max() + .004, A.min() + .002, f"$\\rho$ = {rho_hi:.2f}", fontsize=7, color="0.3", ha="right")
ax.text(R.min(), A.max() - .004, f"all: $\\rho$ = {rho:.2f}", fontsize=7, color="0.3", va="top")
for kind, (m, c, ms) in style.items():
    pts = [(r[3], r[4]) for r in rows if r[1] == kind]
    if pts:
        x, y = zip(*pts)
        lab = "Trained variants" if kind == "Other trained" else kind
        ax.plot(x, y, m, color=c, ms=ms, ls="none", label=lab, alpha=.8 if c.startswith("0.") else 1)
ax.set_xlabel("proxy reward of BHAR (active-region fraction)", labelpad=1)
ax.set_ylabel("Accuracy", labelpad=1)
ax.grid(alpha=.3, lw=.4); ax.tick_params(length=2, pad=1.5)
ax.set_xlim(R.min() - .005, R.max() + .005)
fig.legend(fontsize=6, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.0), handletextpad=.2, columnspacing=.6, labelspacing=.2)
fig.tight_layout(pad=.2, rect=(0, 0, 1, .80))
for e in ("pdf", "png"):
    fig.savefig(f"report/fig_proxy.{e}", dpi=300)
print("-> report/fig_proxy.{pdf,png}")
