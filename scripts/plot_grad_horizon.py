"""Paper figure from the log of grad_horizon.py (lines "subject S  L N s  fix B  theta_E T  |grad| G").

    python scripts/plot_grad_horizon.py ~/grad_horizon8.txt   ->  report/fig_grad_horizon.{pdf,png}
Median over subjects, min-max band; color = theta_E, dashed = without fix, solid = with fix.
"""
import re
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

pat = re.compile(r"subject (\d+)\s+L\s+(\d+) s\s+fix (True|False)\s+theta_E\s+(\d+)\s+\|grad\| (\S+)")
res = defaultdict(dict)                       # (fix, theta) -> {(L, subject): grad}
for line in open(sys.argv[1] if len(sys.argv) > 1 else "grad_horizon8.txt"):
    m = pat.search(line)
    if m:
        s, L, fix, th, g = m.groups()
        res[(fix == "True", int(th))][(int(L), int(s))] = max(float(g), 1e-12)
thetas = sorted({th for _, th in res})
lengths = sorted({L for d in res.values() for L, _ in d})
subjects = sorted({s for d in res.values() for _, s in d})

plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.45, 1.9))
colors = dict(zip(thetas, ["#1b9e77", "#7570b3", "#d95f02"]))
for th in thetas:
    for fix, ls in ((False, "--"), (True, "-")):
        v = np.array([[res[(fix, th)][(L, s)] for L in lengths] for s in subjects])
        ax.plot(lengths, np.median(v, 0), ls, color=colors[th], marker="o", ms=2.5, lw=1.2)
        if len(subjects) > 1:
            ax.fill_between(lengths, v.min(0), v.max(0), color=colors[th], alpha=0.15, lw=0)
ax.set_yscale("log")
ax.set_xlabel("segment length (s)")
ax.set_ylabel(r"$|\partial \mathcal{L}_\mathrm{task}/\partial\theta_\mathrm{E}|$")
ax.set_xticks(lengths)
leg1 = ax.legend([Line2D([], [], color=colors[t], lw=1.5) for t in thetas],
                 [fr"$\theta_\mathrm{{E}}={t}$" for t in thetas], loc="upper left", frameon=False)
ax.add_artist(leg1)
ax.legend([Line2D([], [], color="k", ls="--", lw=1), Line2D([], [], color="k", ls="-", lw=1)],
          ["without fix", "with fix (Sec. III-D)"], loc="upper center", frameon=False)
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(f"report/fig_grad_horizon.{ext}", dpi=300)
print(f"{len(subjects)} subjects -> report/fig_grad_horizon.{{pdf,png}}")
