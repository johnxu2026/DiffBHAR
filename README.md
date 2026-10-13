# DiffBHAR

End-to-end differentiable data acquisition for batteryless HAR, built on top of
[BHAR](https://github.com/SLDGroup/BHAR_codes_artifact) (CODES 2026).

## Layout

```
external/BHAR_codes_artifact   original BHAR code (git submodule, pinned, not modified)
```

## Setup

```bash
git clone --recurse-submodules <this repo>
```

## Roadmap

- [ ] 1. Run original BHAR evaluation (PAMAP2 + tinyhar) from cached checkpoints
- [ ] 2. Re-implement the energy simulator; test packet-by-packet against BHAR
- [ ] 3. Re-implement packet hold + active/passive evaluation; test against BHAR
- [ ] 4. Make the simulator differentiable (hard forward, surrogate backward)
- [ ] 5. Oracle upper bound; joint policy + classifier training
