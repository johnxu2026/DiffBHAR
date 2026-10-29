"""Progress of bhar_e2e.py + bhar_posttrain.sh runs, one line per tag and seed.

    python scripts/progress.py altr/e2e altr6/e2e
    python scripts/progress.py --dataset dsads nn100/e2e         (other datasets; default pamap2)
    python scripts/progress.py --dataset dsads,rwhar,opportunity bhar   (BHAR's own runs, no tag)
    python scripts/progress.py bhar512/e2e      (BHAR's own runs under a tag, e.g. POLICY_EPOCHS=512)
"""
import glob
import os
import subprocess
import sys

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "BHAR_codes_artifact")
args = sys.argv[1:]
datasets = ["pamap2"]
if args[:1] == ["--dataset"]:
    datasets, args = args[1].split(","), args[2:]
METHOD = "asynchronous_multisensor_time_context"


def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def lines(path):
    with open(path, errors="replace") as f:
        return f.read().splitlines()


def last_line(ls):
    return ls[-1][:19] + " " + ls[-1][ls[-1].find("]") + 1:].strip()[:45] if ls else ""


ps = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "args="], capture_output=True, text=True).stdout
print("experiments running" if "bhar_e2e.py" in ps or "train_har_policy.py" in ps else "nothing running")
for dataset in datasets:
    N = {"pamap2": 8, "dsads": 8, "rwhar": 11, "opportunity": 4}[dataset]      # test subjects
    logs = os.path.join(ART, "saved_data", "logs", dataset, "tinyhar")
    results = os.path.join(ART, "saved_data", "results", dataset, "tinyhar")
    pre = f"{dataset} " if len(datasets) > 1 else ""
    for tag in args:
        if tag == "bhar":                 # bhar_posttrain.sh without a tag: policy search, finetune, eval
            for m in ("conservative", "opportunistic"):
                for seed in range(3):
                    name = f"{pre}BHAR {m} seed{seed}"
                    if os.path.exists(os.path.join(results, f"{m}-{METHOD}", f"results_seed{seed}.pickle")):
                        print(f"{name}: DONE")
                        continue
                    log = latest(os.path.join(logs, f"{m}-{METHOD}", f"train_log_seed{seed}_*.log"))
                    if log is None:
                        print(f"{name}: not started")
                        continue
                    ls = lines(log)
                    n = sum("Test Group" in l for l in ls)
                    print(f"{name}: subject {max(n, 1)}/{N} | last log: {last_line(ls)}")
            continue
        for seed in range(3):
            name = f"{pre}{tag} seed{seed}"
            log = latest(os.path.join(logs, tag, f"policy_train_seed{seed}_*.log"))
            if os.path.exists(os.path.join(results, tag, f"conservative-{METHOD}", f"results_seed{seed}.pickle")):
                print(f"{name}: DONE")
                continue
            ft = latest(os.path.join(logs, tag, f"conservative-{METHOD}", f"train_log_seed{seed}_*.log"))
            if log is None and ft is not None:   # bhar_posttrain.sh alone under a tag (e.g. bhar512/e2e)
                ls = lines(ft)
                n = sum("Test Group" in l for l in ls)
                print(f"{name}: BHAR policy search + finetune, subject {max(n, 1)}/{N} | last log: {last_line(ls)}")
                continue
            if log is None:
                print(f"{name}: not started")
                continue
            ls = lines(log)
            n_theta = sum("saved theta" in l for l in ls)
            if n_theta < N:
                print(f"{name}: theta training, subject {n_theta + 1}/{N} | last log: {last_line(ls)}")
                continue
            ft = latest(os.path.join(logs, tag, f"conservative-{METHOD}", f"train_log_seed{seed}_*.log"))
            n_ft = sum("Test Group" in l for l in lines(ft)) if ft else 0
            print(f"{name}: theta done, BHAR finetune/eval subject {max(n_ft, 1)}/{N}")
