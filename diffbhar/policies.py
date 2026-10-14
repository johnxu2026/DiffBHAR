"""Transmit policies with a hard-forward / soft-backward binarisation.

Each policy maps sensor-local, causally available features to a logit; ``binarize`` turns
it into a {0,1} action whose gradient is the straight-through (optionally Gumbel/logistic
noise) estimator. Sensors are simulated as rows; ``sensor_index`` maps rows to sensors so
each body part gets its own parameters, as in BHAR.
"""
import torch
import torch.nn as nn


class STEBinarizer(nn.Module):
    """a = 1[sigmoid((l + noise)/T) >= 0.5] forward, d a / d l = d sigmoid((l+noise)/T) / d l backward."""

    def __init__(self, temperature=1.0, noise=True):
        super().__init__()
        self.temperature = temperature
        self.noise = noise

    def binarize(self, logit, train=True):
        if train and self.noise:
            u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            logit = logit + torch.log(u) - torch.log1p(-u)       # logistic noise = binary Gumbel
        soft = torch.sigmoid(logit / self.temperature)
        hard = (logit >= 0).to(soft.dtype)
        return hard + soft - soft.detach(), soft


class DiffConservativePolicy(STEBinarizer):
    """BHAR's conservative policy made differentiable.

    hard:  send iff Delta_k >= tau,  tau = theta_e * (1 - E~_k) * P~_k + theta_p
    (the artifact's ordering, see harvesting_sensor.py; the paper names the two
    parameters the other way round). Logit = (Delta - tau) / scale.
    """

    def __init__(self, sensor_index, n_sensors, init=(0.0, 0.0), scale=5.0, **kw):
        super().__init__(**kw)
        self.register_buffer("sensor_index", sensor_index)
        theta = torch.tensor(init, dtype=torch.float64)
        if theta.dim() == 1:
            theta = theta.expand(n_sensors, 2).clone()
        self.theta = nn.Parameter(theta)
        self.scale = scale

    def forward(self, f):
        th = self.theta[self.sensor_index]
        tau = th[:, 0] * (1 - f["e_n"]) * f["p_n"] + th[:, 1]
        return (f["delta"] - tau) / self.scale


class NeuralPolicy(STEBinarizer):
    """Tiny per-sensor MLP on sensor-local features (cheap enough for an nRF52 MCU).

    Features: energy surplus, 1 s harvest rate, last-200 ms harvest rate, their difference
    (a free proxy for an activity change), and time since last packet (seconds).
    """

    def __init__(self, sensor_index, n_sensors, hidden=8, **kw):
        super().__init__(**kw)
        self.register_buffer("sensor_index", sensor_index)
        self.n_in = 5
        self.w1 = nn.Parameter(torch.randn(n_sensors, self.n_in, hidden, dtype=torch.float64) * 0.3)
        self.b1 = nn.Parameter(torch.zeros(n_sensors, hidden, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.randn(n_sensors, hidden, dtype=torch.float64) * 0.3)
        self.b2 = nn.Parameter(torch.zeros(n_sensors, dtype=torch.float64))

    def features(self, f):
        fs = f["cfg"].fs
        return torch.stack([f["e_n"], f["p_n"], f["p_short"], f["p_short"] - f["p_n"],
                            f["delta"] / fs], -1)

    def forward(self, f):
        x = self.features(f)
        i = self.sensor_index
        hdn = torch.tanh(torch.einsum("ni,nih->nh", x, self.w1[i]) + self.b1[i])
        return (hdn * self.w2[i]).sum(-1) + self.b2[i]


class OpportunisticPolicy(STEBinarizer):
    """Always wants to send; the feasibility gate decides. No parameters."""

    def __init__(self):
        super().__init__(noise=False)

    def forward(self, f):
        return torch.ones_like(f["e_n"])
