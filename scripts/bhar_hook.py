"""Run a BHAR experiment script with our neural policies available.

BHAR builds each sensor's policy string as "conservative_<best[0]>_<best[1]>" from the
policy pkl. scripts/bhar_e2e.py --policy neural writes best[bp] = ["neural", "<file.pt>#<sensor>"],
so the string becomes "conservative_neural_<file.pt>#<sensor>". For those strings this hook
replaces EnergyHarvestingSensor.sparsify_data with our simulator (hard forward, the same
energy model and packet timing as BHAR) driving the neural policy; every other policy string
goes to BHAR's own code unchanged.

usage (from external/BHAR_codes_artifact/experiments):
    python ../../../scripts/bhar_hook.py train_har_policy.py <its arguments>
"""
import os
import runpy
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.getcwd(), os.path.join(ROOT, "external", "BHAR_codes_artifact"), ROOT]

from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor  # noqa: E402
from diffbhar.policies import NeuralPolicy  # noqa: E402
from diffbhar.sim import SimConfig, simulate  # noqa: E402

PREFIX = "conservative_neural_"
bhar_sparsify = EnergyHarvestingSensor.sparsify_data


def sparsify_data(self, policy, data):
    if not policy.startswith(PREFIX):
        return bhar_sparsify(self, policy, data)
    path, i = policy[len(PREFIX):].rsplit("#", 1)
    sd = torch.load(path, map_location="cpu")
    pol = NeuralPolicy(torch.tensor([int(i)]), sd["b2"].shape[0], hidden=sd["b1"].shape[1], noise=False)
    pol.load_state_dict(sd, strict=False)
    self.init_harvester(data)
    cfg = SimConfig(packet_size=self.packet_size, leakage=self.leakage, fs=self.FS, max_energy=self.MAX_E,
                    policy_frequency=self.FS // self.policy_period, tx_thresh=self.thresh,
                    init_energy=self.init_energy)
    with torch.no_grad():
        o = simulate(torch.tensor(self.e_harvest, dtype=torch.float64)[None], pol, cfg, train=False)
    k = o.ticks[o.a[0] > 0.5].numpy()
    self.n_feasible = int(o.feasible.sum())   # decisions with E >= E_th (only these need the network)
    # a packet decided at step k covers samples k+1 .. k+w, as in BHAR's send_packet
    self.packet_idxs = np.stack([k + 1, k + 1 + self.packet_size]).T
    return self.packet_idxs


EnergyHarvestingSensor.sparsify_data = sparsify_data

if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
