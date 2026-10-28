"""Progress of bhar_e2e.py + bhar_posttrain.sh runs, one line per tag and seed.

    python scripts/progress.py altr/e2e altr6/e2e
    python scripts/progress.py --dataset dsads nn100/e2e         (other datasets; default pamap2)
    watch -n 60 python scripts/progress.py altr/e2e altr6/e2e     (refreshes every minute, Ctrl+C to quit)
"""
import glob
import os
import subprocess
import sys

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "BHAR_codes_artifact")
args = sys.argv[1:]
dataset = "pamap2"
if args[:1] == ["--dataset"]:
    dataset, args = args[1], args[2:]
N = {"pamap2": 8, "dsads": 8, "rwhar": 11, "opportunity": 4}[dataset]      # test subjects
LOGS = os.path.join(ART, "saved_data", "logs", dataset, "tinyhar")
RESULTS = os.path.join(ART, "saved_data", "results", dataset, "tinyhar")
METHOD = "conservative-asynchronous_multisensor_time_context"


def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def lines(path):
    with open(path, errors="replace") as f:
        return f.read().splitlines()


ps = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "args="], capture_output=True, text=True).stdout
print("experiments running" if "bhar_e2e.py" in ps or "train_har_policy.py" in ps else "nothing running")
for tag in args:
    for seed in range(3):
        log = latest(os.path.join(LOGS, tag, f"policy_train_seed{seed}_*.log"))
        if log is None:
            print(f"{tag} seed{seed}: not started")
            continue
        if os.path.exists(os.path.join(RESULTS, tag, METHOD, f"results_seed{seed}.pickle")):
            print(f"{tag} seed{seed}: DONE")
            continue
        ls = lines(log)
        n_theta = sum("saved theta" in l for l in ls)
        if n_theta < N:
            last = ls[-1][:19] + " " + ls[-1][ls[-1].find("]") + 1:].strip()[:45] if ls else ""
            print(f"{tag} seed{seed}: theta training, subject {n_theta + 1}/{N} | last log: {last}")
            continue
        ft = latest(os.path.join(LOGS, tag, METHOD, f"train_log_seed{seed}_*.log"))
        n_ft = sum("Test Group" in l for l in lines(ft)) if ft else 0
        print(f"{tag} seed{seed}: theta done, BHAR finetune/eval subject {max(n_ft, 1)}/{N}")
