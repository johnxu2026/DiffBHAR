"""Accuracy vs packets (PAMAP2, TinyHAR, 3 seeds): fixed timer (0, theta_p) without training,
ours (jnn: trained from the same starts), BHAR.  Numbers from EXPERIMENTS.md (exp 13, 35, 36)."""
import matplotlib.pyplot as plt
import numpy as np

fixed = {300: (.057, .450, .008), 200: (.082, .520, .006), 150: (.102, .544, .005),
         100: (.137, .574, .005), 50: (.213, .581, .007)}            # pkt/s, acc, acc std
ours = {200: (.0934, .542, .002), 150: (.1129, .568, .014), 100: (.149, .584, .004), 50: (.2189, .571, .001)}
bhar = (.196, .573, .004)

fig, ax = plt.subplots(figsize=(4.6, 3.4))
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
ax.set_title("PAMAP2")
ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"report/fig_pareto.{e}", dpi=200)
