"""End-to-end differentiable BHAR pipeline: policy -> sensor energy -> packet hold ->
MSC-T classifier -> held prediction -> per-time-step loss.

Forward semantics follow the artifact's evaluation loop (train_har_policy.py): a packet
decided at tick k covers samples k+1..k+w and arrives at t = k+w+1; at every arrival the
classifier sees the latest packet of every sensor plus its age, and its prediction is held
until the next arrival. The per-step loss over the held prediction is the CE surrogate of
the total error  eps = active error + passive error  (BHAR Eq. 13), so the policy is trained
on the quantity BHAR's region objective R_pi only approximates.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .sim import simulate, SimConfig


@dataclass
class E2EOutput:
    loss_task: torch.Tensor      # held-prediction CE over all time steps (policy objective)
    loss_active: torch.Tensor    # CE at actual arrivals only (BHAR Eq. 19, classifier objective)
    n_packets: torch.Tensor      # STE count of sent packets (for an optional resource term)
    preds: np.ndarray            # [B, L] hard held predictions
    sim: object
    arrivals: torch.Tensor       # [J] arrival steps


def gather_packets(x, ticks, w):
    """x: [B, b, L, 3] -> candidate packets [B, b, J, w, 3] for decisions at ``ticks``."""
    idx = ticks[:, None] + 1 + torch.arange(w, device=x.device)[None]   # [J, w]
    return x[:, :, idx]


def run_e2e(x, y, h, policy, classifier, cfg: SimConfig, n_classes, max_p=None,
            class_weights=None, train=True, counterfactual=True, init_pred=None):
    """x: [B,b,L,3] classifier data, y: [B,L] long labels, h: [B,b,L] harvested energy (J).

    classifier(x [M, w, 3b], ages [M, b]) -> logits [M, C]   (the artifact's model signature)
    """
    B, b, L, _ = x.shape
    w = cfg.packet_size
    sim = simulate(h.reshape(B * b, L), policy, cfg,
                   max_p=None if max_p is None else max_p, train=train)
    J = len(sim.ticks)
    a = sim.a.reshape(B, b, J).to(x.dtype)                 # STE, forward in {0,1}
    feas = sim.feasible.reshape(B, b, J)
    ticks = sim.ticks
    arr = ticks + w + 1                                    # arrival steps t_j
    Xnew = gather_packets(x, ticks, w)                     # [B,b,J,w,3]

    # --- per-sensor packet hold + ages (differentiable through a) ---
    Xhat = torch.zeros(B, b, w, 3, dtype=x.dtype, device=x.device)
    age = torch.zeros(B, b, dtype=x.dtype, device=x.device)
    has = torch.zeros(B, b, dtype=torch.bool, device=x.device)
    X_act, A_act, X_cf, A_cf = [], [], [], []
    prev_t = None
    for j in range(J):
        m = a[:, :, j]
        mh = m.detach() > 0.5
        if prev_t is not None:
            age = age + float(arr[j] - prev_t)
        prev_t = arr[j]
        Xhat = m[..., None, None] * Xnew[:, :, j] + (1 - m)[..., None, None] * Xhat
        age = (1 - m) * age
        has = has | mh
        X_act.append(Xhat)
        A_act.append(torch.where(has, age, torch.full_like(age, -1.0)))
        if counterfactual:
            # "what would the classifier say had every feasible sensor sent now"
            fresh = (feas[:, :, j] | mh)
            X_cf.append(torch.where(fresh[..., None, None], Xnew[:, :, j], Xhat))
            A_cf.append(torch.where(fresh, torch.zeros_like(age), A_act[-1]))

    def to_clf(Xs, As):
        X = torch.stack(Xs, 1)                             # [B,J,b,w,3]
        X = X.permute(0, 1, 3, 2, 4).reshape(B * J, w, 3 * b)
        A = torch.stack(As, 1).reshape(B * J, b)
        return classifier(X, A).reshape(B, J, -1)

    logits = to_clf(X_act, A_act)                          # [B,J,C]
    probs = logits.softmax(-1)
    M = 1 - torch.prod(1 - a, dim=1)                       # any sensor arrived, [B,J]
    Mh = M.detach() > 0.5
    if counterfactual:
        probs_cf = to_clf(X_cf, A_cf).softmax(-1)
        s = torch.where(Mh[..., None], probs, probs_cf)    # forward identical (M=0 there)
    else:
        s = probs

    # --- held prediction recursion ---
    if init_pred is None:
        init_pred = torch.full((B, n_classes), 1.0 / n_classes, dtype=probs.dtype, device=x.device)
    held = []
    cur = init_pred
    for j in range(J):
        cur = M[:, j, None] * s[:, j] + (1 - M[:, j, None]) * cur
        held.append(cur)
    held = torch.stack(held, 1)                            # [B,J,C]

    # --- per-step CE over the held prediction (intervals [t_j, t_{j+1})) ---
    cw = torch.ones(n_classes, device=x.device) if class_weights is None else class_weights
    seg = torch.zeros(L, dtype=torch.long, device=x.device)   # 0 -> init_pred, j+1 -> held_j
    for j in range(J):
        seg[int(arr[j]):] = j + 1
    allp = torch.cat([init_pred[:, None], held], 1)            # [B,J+1,C]
    p_step = allp.gather(1, seg[None, :, None].expand(B, L, n_classes))  # [B,L,C]
    logp_y = torch.log(p_step.gather(2, y[..., None]).squeeze(-1).clamp_min(1e-8))
    wy = cw[y]
    loss_task = -(wy * logp_y).sum() / wy.sum()

    # --- classifier loss on actual arrivals only (BHAR's active-region finetuning) ---
    y_arr = y[:, arr]                                          # label at arrival step
    ce = F.cross_entropy(logits.reshape(B * J, -1), y_arr.reshape(-1), weight=cw.to(logits.dtype),
                         reduction="none")
    mask = Mh.reshape(-1).to(ce.dtype)
    loss_active = (ce * mask).sum() / mask.sum().clamp_min(1.0)

    preds = p_step.argmax(-1).detach().cpu().numpy()
    return E2EOutput(loss_task=loss_task, loss_active=loss_active, n_packets=a.sum(),
                     preds=preds, sim=sim, arrivals=arr)


def region_metrics(y, preds, arrivals_mask, arr):
    """Active/passive decomposition as in SparseHarDataset.region_decomposition.

    y, preds: [L] numpy; arrivals_mask: [J] bool (any sensor arrived); arr: [J] steps."""
    L = len(y)
    at = np.asarray(arr)[np.asarray(arrivals_mask)]
    passive = np.zeros(L, dtype=bool)
    trans = np.concatenate([[0], np.where(y[:-1] != y[1:])[0] + 1])
    if len(at) == 0:
        passive[:] = True
    else:
        passive[:at[0]] = True
        bounds = list(at) + [L]
        for i in range(len(at)):
            lo, hi = bounds[i], bounds[i + 1]
            tb = trans[(trans > lo) & (trans < hi)]
            if len(tb):
                passive[tb[0]:hi] = True
    correct = preds == y
    pr = passive.mean()
    return dict(acc=correct.mean(), passive_region=pr,
                active_err=1 - correct[~passive].mean() if (~passive).any() else np.nan,
                passive_err=1 - correct[passive].mean() if passive.any() else np.nan)
