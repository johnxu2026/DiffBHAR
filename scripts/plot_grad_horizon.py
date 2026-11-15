"""Paper figure from the log of grad_horizon.py (lines "subject S  L N s  fix B  theta_E T  |grad| G").

    python scripts/plot_grad_horizon.py ~/grad_horizon8.txt   ->  report/fig_grad_horizon.{pdf,png}

Reader-first version: one timer that limits sending (theta_E = 200 samples = 8 s); red dashed = without the
stop-gradient, blue solid = with it (ours); shaded zone where a gradient step is meaningless;
direct labels instead of a legend. Median over subjects with min-max band.
"""
import re
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pat = re.compile(r"subject (\d+)\s+L\s+(\d+) s\s+fix (True|False)\s+theta_E\s+(\d+)\s+\|grad\| (\S+)")
res = defaultdict(dict)                       # (fix, theta) -> {(L, subject): grad}
for line in open(sys.argv[1] if len(sys.argv) > 1 else "grad_horizon8.txt"):
    m = pat.search(line)
    if m:
        s, L, fix, th, g = m.groups()
        res[(fix == "True", int(th))][(int(L), int(s))] = max(float(g), 1e-12)
lengths = sorted({L for d in res.values() for L, _ in d})
subjects = sorted({s for d in res.values() for _, s in d})

RED, BLUE = "#c0392b", "#2166ac"
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.45, 1.9))
ax.axhspan(1e3, 1e26, color="0.92", lw=0)
ax.text(12, 3e23, "exploding gradient (> 10$^3$)", fontsize=6.5, color="0.35", va="center")


def curve(fix, th, color, ls, alpha_line=1.0):
    v = np.array([[res[(fix, th)][(L, s)] for L in lengths] for s in subjects])
    ax.plot(lengths, np.median(v, 0), ls, color=color, marker="o", ms=2.5, lw=1.3, alpha=alpha_line)
    if len(subjects) > 1:
        ax.fill_between(lengths, v.min(0), v.max(0), color=color, alpha=0.12, lw=0)
    return np.median(v, 0)


m200 = curve(False, 200, RED, "--")
f200 = curve(True, 200, BLUE, "-")
ax.text(115, 1e16, "without stop-gradient", ha="right", color=RED, fontsize=7)
ax.annotate("with stop-gradient (ours)", (lengths[-1], f200[-1]), xytext=(0, 6),
            textcoords="offset points", ha="right", color=BLUE, fontsize=7)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1e26)
ax.set_yticks([1e0, 1e5, 1e10, 1e15, 1e20, 1e25])
ax.set_xlabel("length of training segment (s)")
ax.set_ylabel("gradient magnitude (log scale)")
ax.set_xticks(lengths)
ax.set_xlim(lengths[0] - 5, lengths[-1] + 5)
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(f"report/fig_grad_horizon.{ext}", dpi=300)
print(f"{len(subjects)} subjects -> report/fig_grad_horizon.{{pdf,png}}")
