"""Fig. 2 of the paper at single-column width (3.45 in): accuracy vs packets, fixed timer vs ours (joint)
vs BHAR, PAMAP2 and DSADS. Same numbers as plot_pareto.py; theta_E = 300 on PAMAP2 omitted (no trained run)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": .6, "pdf.fonttype": 42})
D = {
 "PAMAP2": ({200: (.082, .520, .006), 150: (.102, .544, .005), 100: (.137, .574, .005), 50: (.213, .581, .007)},
            {200: (.0934, .542, .002), 150: (.1129, .568, .014), 100: (.149, .584, .004), 50: (.2189, .571, .001)},
            (.196, .573, .004)),
 "DSADS": ({200: (.0867, .510, .009), 150: (.1087, .543, .010), 100: (.147, .577, .016), 50: (.2172, .579, .009)},
           {200: (.0989, .545, .008), 150: (.1224, .565, .009), 100: (.163, .587, .015), 50: (.2237, .572, .009)},
           (.178, .575, .007)),
}
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75))
for ax, (name, (fixed, ours, bhar)) in zip(axs, D.items()):
    f = np.array(sorted(fixed.values())); o = np.array(sorted(ours.values()))
    ax.errorbar(f[:,0], f[:,1], f[:,2], fmt="o-", color="0.45", ms=3, lw=1, capsize=1.5, elinewidth=.6, label="Fixed timer")
    ax.errorbar(o[:,0], o[:,1], o[:,2], fmt="s-", color="tab:blue", ms=3, lw=1, capsize=1.5, elinewidth=.6, label="Ours (joint)")
    for p, (x, y, _) in fixed.items():
        ax.annotate("", ours[p][:2], (x, y), arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=.45, lw=.6))
    ax.errorbar(*bhar[:2], bhar[2], fmt="*", ms=6, color="tab:red", capsize=1.5, elinewidth=.6, label="BHAR")
    ax.set_title(name, pad=2); ax.set_xlabel("packets/s/sensor", labelpad=1)
    ax.grid(alpha=.3, lw=.4); ax.tick_params(length=2, pad=1.5)
axs[0].set_ylabel("Accuracy", labelpad=1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03), handletextpad=.3, columnspacing=1)
fig.tight_layout(pad=.2, w_pad=.6, rect=(0, 0, 1, .9))
for e in ("pdf", "png"):
    fig.savefig(f"report/fig_pareto_col.{e}", dpi=300)
