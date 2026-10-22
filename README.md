# DiffBHAR

End-to-end differentiable data acquisition for batteryless HAR, built on top of
[BHAR](https://github.com/SLDGroup/BHAR_codes_artifact) (CODES 2026).

## Layout

```
external/BHAR_codes_artifact   original BHAR code (git submodule, pinned, not modified)
diffbhar/                      differentiable simulator, policies, end-to-end pipeline
scripts/run_pamap2.py          PAMAP2 experiment (opp / bhar / e2e / joint)
data/, results/                created locally, not in git
```

## Setup

```bash
git clone --recurse-submodules <this repo>
```

## Small PAMAP2 experiment

Opportunistic vs. BHAR's objective (maximise active region) vs. DiffBHAR (policy trained on
the classification loss), one leave-one-subject-out fold, CPU only.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy scipy pandas matplotlib
python scripts/run_pamap2.py --exp smoke --test_subjects 1   # downloads PAMAP2 into data/ on first run
```

Outputs go to `results/<date>_<exp>/`: `<run>.log`, `<run>.json` (all metrics plus git commit and command), `<run>_summary.png`, `<run>_timeline.png`, where `<run>` is e.g. `s1_seed0` or `all_seed2_ft7`.

## Roadmap

- [ ] 1. Run original BHAR evaluation (PAMAP2 + tinyhar) from cached checkpoints
- [ ] 2. Re-implement the energy simulator; test packet-by-packet against BHAR
- [ ] 3. Re-implement packet hold + active/passive evaluation; test against BHAR
- [ ] 4. Make the simulator differentiable (hard forward, surrogate backward)
- [ ] 5. Oracle upper bound; joint policy + classifier training
