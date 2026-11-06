"""Accuracy vs packets (TinyHAR, 3 seeds): fixed timer (0, theta_p) without training, ours (jnn: trained
from the same starts), BHAR.  Numbers from EXPERIMENTS.md (exp 13, 35, 36, 38)."""
import matplotlib.pyplot as plt
import numpy as np

D = {  # dataset: fixed {theta_p: (pkt/s, acc, acc std)}, ours {...}, BHAR
    "PAMAP2": ({300: (.057, .450, .008), 200: (.082, .520, .006), 150: (.102, .544, .005),
                100: (.137, .574, .005), 50: (.213, .581, .007)},
               {200: (.0934, .542, .002), 150: (.1129, .568, .014), 100: (.149, .584, .004), 50: (.2189, .571, .001)},
               (.196, .573, .004)),
    "DSADS": ({200: (.0867, .510, .009), 150: (.1087, .543, .010), 100: (.147, .577, .016), 50: (.2172, .579, .009)},
              {200: (.0989, .545, .008), 150: (.1224, .565, .009), 100: (.163, .587, .015), 50: (.2237, .572, .009)},
              (.178, .575, .007)),
}

fig, axs = plt.subplots(1, len(D), figsize=(4.2 * len(D), 3.4))
for ax, (name, (fixed, ours, bhar)) in zip(axs, D.items()):
    f = np.array(sorted(fixed.values()))
    ax.errorbar(f[:, 0], f[:, 1], f[:, 2], fmt="o-", color="gray", capsize=2, label="fixed timer (no training)")
    o = np.array(sorted(ours.values()))
    ax.errorbar(o[:, 0], o[:, 1], o[:, 2], fmt="s-", color="tab:blue", capsize=2, label="ours (trained)")
    for p, (x, y, _) in fixed.items():
        if p in ours:
            ax.annotate("", ours[p][:2], (x, y), arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=.4))
    ax.errorbar(*bhar[:2], bhar[2], fmt="*", ms=12, color="tab:red", capsize=2, label="BHAR")
    ax.set_xlabel("packets / s / sensor")
    ax.set_ylabel("accuracy")
    ax.set_title(name)
    ax.grid(alpha=.3)
axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"report/fig_pareto.{e}", dpi=200)
