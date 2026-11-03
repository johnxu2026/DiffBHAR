"""Accuracy / F1 breakdown in the style of BHAR's Fig. 6 (TinyHAR, mean over 3 seeds).

Each dataset gets an F1 panel and an accuracy panel; in the accuracy panel the bar is stacked with
active error (green) and passive error (red), so the three add up to 1.
Numbers: `python scripts/bhar_cached.py [--rerun] --dataset D` (EXPERIMENTS.md, experiment 34/35).
Dense = BHAR's cached unconstrained-synchronous_multisensor; Opportunistic on PAMAP2 = cache (3 seeds).

usage:  python scripts/plot_breakdown.py [--ours nn100|jnn100] [--out report/fig_breakdown]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataset -> method -> (acc, F1, active error, passive error)
R = {
    "PAMAP2": {"Dense": (.772, .717, .218, .010), "BHAR": (.573, .522, .259, .168),
               "nn100": (.578, .531, .232, .191), "jnn100": (.584, .531, .226, .190),
               "Opp.": (.522, .467, .247, .231)},
    "DSADS": {"Dense": (.735, .708, .254, .010), "BHAR": (.575, .564, .291, .134),
              "nn100": (.586, .574, .265, .150), "Opp.": (.547, .526, .266, .186)},
    "RWHAR": {"Dense": (.765, .758, .226, .009), "BHAR": (.638, .653, .228, .135),
              "nn100": (.635, .647, .203, .162), "Opp.": (.614, .636, .230, .156)},
    "OPPORTUNITY": {"Dense": (.705, .706, .275, .020), "BHAR": (.581, .544, .322, .097),
                    "nn100": (.555, .541, .341, .104), "Opp.": (.575, .537, .299, .126)},
}
COL = {"Dense": "#333333", "BHAR": "#2a4b9b", "Ours": "#e08a1e", "Opp.": "#b0b0b0"}
LABEL = {"Dense": "Battery-powered (dense)", "BHAR": "Cons. Policy + MSC-T (BHAR)",
         "Ours": "Learned policy (ours)", "Opp.": "Opp. Policy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="nn100", help="method shown as ours (falls back to nn100 if missing)")
    ap.add_argument("--out", default="report/fig_breakdown")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(1, 8, figsize=(13, 3.0), sharey=True)
    order = ["Dense", "BHAR", "Ours", "Opp."]
    for i, (ds, res) in enumerate(R.items()):
        vals = {m: res[m] for m in ["Dense", "BHAR", "Opp."]}
        vals["Ours"] = res.get(args.ours, res["nn100"])
        for j, metric in enumerate(["F1", "Acc"]):
            ax = axes[2 * i + j]
            for k, m in enumerate(order):
                acc, f1, ae, pe = vals[m]
                v = f1 if metric == "F1" else acc
                ax.bar(k, v, color=COL[m], edgecolor="black", linewidth=0.6, width=0.8)
                ax.text(k, v / 2, f"{v:.3f}".lstrip("0"), ha="center", va="center", fontsize=6.5,
                        color="white" if m in ("Dense", "BHAR") else "black", rotation=90)
                if metric == "Acc":
                    ax.bar(k, ae, bottom=acc, color="#5a9a3a", edgecolor="black", linewidth=0.6, width=0.8)
                    ax.bar(k, pe, bottom=acc + ae, color="#b22a2a", edgecolor="black", linewidth=0.6, width=0.8)
                    ax.text(k, acc + ae / 2, f"{ae:.3f}".lstrip("0"), ha="center", va="center", fontsize=6, rotation=90)
                    if pe > 0.04:
                        ax.text(k, acc + ae + pe / 2, f"{pe:.3f}".lstrip("0"), ha="center", va="center",
                                fontsize=6, color="white", rotation=90)
            ax.set_xticks([])
            ax.set_xlabel(metric, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", ls="--", lw=0.4, alpha=0.6)
            ax.set_axisbelow(True)
        axes[2 * i].set_title(ds, fontweight="bold", loc="left", x=0.35)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[m], ec="black") for m in order]
    handles += [plt.Rectangle((0, 0), 1, 1, fc="#5a9a3a", ec="black"),
                plt.Rectangle((0, 0), 1, 1, fc="#b22a2a", ec="black")]
    fig.legend(handles, [LABEL[m] for m in order] + ["Active Error", "Passive Error"],
               loc="lower center", ncol=6, frameon=True, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png (ours = {args.ours})")


if __name__ == "__main__":
    main()
