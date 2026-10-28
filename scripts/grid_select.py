"""Pick a policy without looking at the test subject: nested leave-one-subject-out selection.

For each seed and each test subject s, among the given tags choose the one with the best mean
test accuracy over the OTHER subjects (their folds never test on s), then report its accuracy on s.
Picking the best tag by its accuracy on all subjects (s included) uses s's test data to choose;
this does not, so it is a fair grid-search baseline.

    python scripts/grid_select.py fix50_notrain/e2e fixem100p100_notrain/e2e ...
"""
import argparse
import os
from collections import Counter

import numpy as np

from bhar_cached import ART, load

METHOD = "conservative-asynchronous_multisensor_time_context"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--dataset", default="pamap2")
    args = ap.parse_args()
    base = os.path.join(ART, "saved_data", "results", args.dataset, "tinyhar")
    res = {t: load(os.path.join(base, t, METHOD)) for t in args.tags}
    seeds = sorted(set.intersection(*(set(r) for r in res.values())))
    per_seed, picks = [], Counter()
    for k in seeds:
        subs = sorted(res[args.tags[0]][k])
        acc = {t: {s: res[t][k][s][1] for s in subs} for t in args.tags}
        out = []
        for s in subs:
            t = max(args.tags, key=lambda t: np.mean([acc[t][o] for o in subs if o != s]))
            picks[t] += 1
            out.append(acc[t][s])
        per_seed.append(np.mean(out))
    print(f"nested selection over {len(args.tags)} tags, seeds {seeds}: "
          f"acc {np.mean(per_seed):.3f}+-{np.std(per_seed):.3f}")
    for t, n in picks.most_common():
        print(f"  picked {n:3d} times  {t}")


if __name__ == "__main__":
    main()
