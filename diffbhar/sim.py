"""Differentiable batteryless sensor simulator (hard forward / soft backward).

Forward pass reproduces ``energy_harvesting.harvesting_sensor.EnergyHarvestingSensor
.sparsify_data`` from the BHAR artifact exactly (when ``compat=True`` and the policy is
deterministic), so packets and metrics stay comparable with the paper. The backward pass
uses explicit surrogate gradients:

* transmit decision      a  = STE(policy prob)                   (policies.py)
* energy feasibility     g  = 1[E >= E_th]  backward: sigmoid'((E - E_th)/kappa)
* energy consumption     dE_{k+1}/da_k = -E_th * usage_factor
* time since last packet dDelta_{k+1}/da_k = -(Delta_k + w + 1)

All sensors of all sequences are simulated in parallel as N = B * b independent rows.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class SimConfig:
    packet_size: int = 8            # w, samples per packet
    leakage: float = 6.6e-6         # W, idle + policy power (E_L = leakage / fs per step)
    fs: int = 25                    # Hz
    max_energy: float = 200e-6      # J
    policy_frequency: int = 5       # Hz, policy evaluated every fs/policy_frequency steps
    tx_thresh: float = 50e-6        # J, E_th
    init_energy: float = 0.0        # J, extra energy for the first packet after dying
    feas_kappa: float = 5e-6        # J, width of the sigmoid surrogate for the feasibility gate
    compat: bool = True             # replicate artifact energy bookkeeping during a packet

    @property
    def policy_period(self):
        return int(self.fs / self.policy_frequency)

    @property
    def hbar_window(self):
        return int(1 / (self.policy_period / self.fs))

    @property
    def leak_per_step(self):
        return (1 / self.fs) * self.leakage   # same rounding as the artifact


def segment_max_p(h, window=25):
    """Per-row max of the 25-sample sliding sum of harvested energy.

    This is what the artifact uses to normalise the harvesting rate (it is computed on the
    whole segment, i.e. it peeks at the future). Use it only for exact reproduction; for a
    causal policy pass a constant computed on the training set instead."""
    h_np = h.detach().cpu().numpy()
    m = [np.convolve(row, np.ones(window, dtype=int), 'valid').max() for row in h_np]
    return torch.tensor(m, dtype=h.dtype, device=h.device)


def _seqsum(x):
    # left-to-right sum, matches Python's sum() over a numpy row bit for bit
    out = torch.zeros_like(x[:, 0])
    for i in range(x.shape[1]):
        out = out + x[:, i]
    return out


@dataclass
class SimOutput:
    ticks: torch.Tensor        # [J] decision steps k_j (only those whose packet fits)
    a: torch.Tensor            # [N, J] executed transmit decisions, STE (forward in {0,1})
    feasible: torch.Tensor     # [N, J] bool, hard feasibility at the tick
    prob: torch.Tensor         # [N, J] policy probability (diagnostics)
    logit: torch.Tensor        # [N, J] policy logit before noise; P(send) = sigmoid(logit)
    energy: torch.Tensor       # [N, L] energy trace (forward values)
    n_deaths: torch.Tensor     # [N]


def simulate(h, policy, cfg: SimConfig, max_p=None, train=True, record_energy=False):
    """Unroll the sensor + policy.

    h:       [N, L] harvested energy per step (J), from EnergyHarvester (no grad needed)
    policy:  callable(features: dict) -> logits [N]; see policies.py
    max_p:   [N] or scalar normaliser for the harvest rate. None -> segment_max_p(h)
    """
    N, L = h.shape
    dt = h.dtype
    dev = h.device
    w, P, W = cfg.packet_size, cfg.policy_period, cfg.hbar_window
    thr, emax = cfg.tx_thresh, cfg.max_energy
    if max_p is None:
        max_p = segment_max_p(h)
    max_p = torch.as_tensor(max_p, dtype=dt, device=dev).expand(N)
    # linear energy use over the packet (numpy linspace for bit-exact compat)
    usage = torch.tensor(np.linspace(0, thr, w + 1)[1:], dtype=dt, device=dev)

    E = torch.zeros(N, dtype=dt, device=dev)
    delta = torch.zeros(N, dtype=dt, device=dev)       # k - last_sent_idx (differentiable)
    need_init = torch.ones(N, dtype=torch.bool, device=dev)
    resume = torch.zeros(N, dtype=torch.long, device=dev)  # first non-busy step
    k0 = torch.zeros(N, dtype=torch.long, device=dev)      # step of last decision to send
    E_ref = torch.zeros(N, dtype=dt, device=dev)            # energy at k0 (carries grad)
    uf = torch.ones(N, dtype=dt, device=dev)
    buf = torch.zeros(N, W, dtype=dt, device=dev)
    p_n = torch.zeros(N, dtype=dt, device=dev)
    e_n = torch.zeros(N, dtype=dt, device=dev)
    p_short = torch.zeros(N, dtype=dt, device=dev)
    deaths = torch.zeros(N, dtype=torch.long, device=dev)

    ticks, a_list, feas_list, prob_list, logit_list = [], [], [], [], []
    e_trace = [E] if record_energy else None

    for k in range(1, L):
        busy = k < resume
        hk = h[:, k]
        # --- normal step: harvest, leak, saturate (clamp has zero grad when saturated)
        E_raw = E + hk - cfg.leak_per_step
        died = (~busy) & (E_raw < 0)
        E_norm = torch.clamp(E_raw, min=0.0, max=emax)
        # --- step inside a packet acquisition
        j = (k - k0).clamp(1, w) - 1
        if cfg.compat:
            # artifact: E[k0+j] = E[k0] + h[k0+j] - usage_j  (no leak, no clip, no accumulation)
            E_busy = E_ref + hk - usage[j] * uf
        else:
            E_busy = torch.clamp(E + hk - cfg.leak_per_step - (thr / w) * uf, min=0.0, max=emax)
        E = torch.where(busy, E_busy, E_norm)
        delta = torch.where(died, torch.zeros_like(delta), delta + 1)
        need_init = need_init | died
        deaths = deaths + died.long()

        if k % P == 0:
            upd = ~busy
            newval = _seqsum(h[:, k - P:k])
            buf = torch.where(upd[:, None], torch.cat([buf[:, 1:], newval[:, None]], 1), buf)
            p_n = torch.where(upd, _seqsum(buf) / max_p, p_n)
            p_short = torch.where(upd, newval * W / max_p, p_short)
            e_n = torch.where(upd, (E - thr) / (emax - thr), e_n)

            feats = dict(e_n=e_n, p_n=p_n, p_short=p_short, delta=delta, k=k, cfg=cfg)
            logit = policy(feats)
            a_pol, prob = policy.binarize(logit, train=train)

            thr_eff = thr + need_init.to(dt) * cfg.init_energy
            fits = (k + w + 1) < L
            feas_hard = upd & (E >= thr_eff) & fits
            g_soft = torch.sigmoid((E - thr_eff) / cfg.feas_kappa)
            g = feas_hard.to(dt) + (g_soft - g_soft.detach()) * upd.to(dt)
            a = a_pol * g                                  # forward: a_pol_hard & feasible
            sent = a.detach() > 0.5

            # surrogate-only terms (forward value unchanged): spending energy, resetting delta
            uf_now = torch.where(need_init, (thr + cfg.init_energy) / thr, torch.ones_like(uf))
            E = E - thr * uf_now * (a - a.detach())
            delta = delta - (delta.detach() + w + 1) * a   # forward: -> -(w+1) when sent

            E_ref = torch.where(sent, E, E_ref)
            uf = torch.where(sent, uf_now, uf)
            k0 = torch.where(sent, torch.full_like(k0, k), k0)
            resume = torch.where(sent, torch.full_like(resume, k + w + 1), resume)
            need_init = need_init & ~sent

            if fits:
                ticks.append(k)
                a_list.append(a)
                feas_list.append(feas_hard)
                prob_list.append(prob)
                logit_list.append(logit)
        if record_energy:
            e_trace.append(E)

    return SimOutput(
        ticks=torch.tensor(ticks, device=dev),
        a=torch.stack(a_list, 1),
        feasible=torch.stack(feas_list, 1),
        prob=torch.stack(prob_list, 1),
        logit=torch.stack(logit_list, 1),
        energy=torch.stack(e_trace, 1).detach() if record_energy else None,
        n_deaths=deaths,
    )
