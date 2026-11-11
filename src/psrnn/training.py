"""Training of the bed model on measured phases, with or without state-of-charge labels."""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .data import make_batch
from .observer import observe_water_fraction


@dataclass
class LossWeights:
    """Weights and scales of the loss terms.

    L = w_T mean((T - T_mes)^2) / s_T^2 + w_pv mean((pv - pv_mes)^2) / s_pv^2
        + w_soc mean((SoC - SoC_label)^2) / s_soc^2 + w_prior * prior(isotherm)
    Set ``w_soc=0`` to train from outlet measurements only.
    """
    w_T: float = 1.0
    w_pv: float = 1.0
    w_soc: float = 1.0
    w_prior: float = 0.1
    s_T: float = 5.0
    s_pv: float = 150.0
    s_soc: float = 0.05


def loss_terms(pred, batch, model, w: LossWeights):
    def mse(x, y, m, s):
        return ((x - y) ** 2 / s ** 2 * m).sum() / m.sum().clamp(min=1)
    lT = mse(pred['T_out'], batch.T_out, batch.mask_T, w.s_T)
    lpv = mse(pred['pv_out'], batch.pv_out, batch.mask_pv, w.s_pv)
    lsoc = mse(pred['soc'], batch.soc, batch.mask_soc, w.s_soc)
    lp = model.prior()
    total = w.w_T * lT + w.w_pv * lpv + w.w_soc * lsoc + w.w_prior * lp
    return total, dict(T=float(lT.detach()), pv=float(lpv.detach()), soc=float(lsoc.detach()), prior=float(lp.detach()))


@dataclass
class TrainResult:
    model: nn.Module
    loss: float                      # best training loss
    history: list = field(default_factory=list)
    initial_soc: dict = None         # learnt initial SoC of each phase (latent initial state)


def perturb(model, seed):
    """Spread the starting points of an ensemble: NTU and network weights perturbed (no change for seed 0)."""
    if not seed:
        return model
    with torch.no_grad():
        model.l_ntum += 0.3 * torch.randn((), dtype=torch.float64)
        model.l_ntuh += 0.3 * torch.randn((), dtype=torch.float64)
        for p in model.mlp.parameters():
            p += 0.01 * torch.randn_like(p)
    return model


def train(model, phases, epochs=150, lr=1e-2, seed=0, latent_init=False, weights=None,
          latent_lr=0.02, log_every=10, verbose=True):
    """Train on a list of phases (full batch, Adam, cosine learning rate, best state kept).

    ``model`` is a :class:`SorptionBedRNN` or a callable returning one; with a callable the model is
    built after seeding, which makes the run reproducible. With ``latent_init=True`` the initial water
    content of each phase is unknown: it is learnt with the model, starting from the observer.
    """
    w = weights or LossWeights()
    torch.manual_seed(seed)
    model = model() if callable(model) and not isinstance(model, nn.Module) else model
    perturb(model, seed)
    batch = make_batch(phases, model.K)
    groups = [dict(params=list(model.parameters()), lr=lr)]
    z = None
    if latent_init:
        x0 = torch.tensor([observe_water_fraction(model, p) for p in phases], dtype=torch.float64).clamp(-0.05, 1.05)
        z = nn.Parameter(torch.atanh(((x0 - 0.5) / 0.6).clamp(-0.99, 0.99)))
        groups.append(dict(params=[z], lr=latent_lr))
    elif batch.water0 is None:
        raise ValueError('some phases have no initial water content: use latent_init=True')
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state, history = np.inf, None, []
    for ep in range(epochs):
        opt.zero_grad()
        w0 = model.water_from_soc(1 - (0.5 + 0.6 * torch.tanh(z)), batch.scale) if latent_init else batch.water0
        L, parts = loss_terms(model(batch, w0), batch, model, w)
        if not torch.isfinite(L):
            if verbose:
                print(f'    epoch {ep}: non-finite loss, step skipped', flush=True)
            sched.step(); continue
        L.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            if verbose:
                print(f'    epoch {ep}: non-finite gradient, step skipped', flush=True)
            opt.zero_grad(); sched.step(); continue
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        lv = float(L.detach()); parts['loss'] = lv; history.append(parts)
        if lv < best:
            best, state = lv, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and log_every and ep % log_every == 0:
            print(f"    epoch {ep}: loss {lv:.4f} (T {parts['T']:.3f}, pv {parts['pv']:.3f}, "
                  f"SoC {parts['soc']:.3f}, prior {parts['prior']:.3f})", flush=True)
    if state is not None:
        model.load_state_dict(state)
    soc0 = None
    if latent_init:
        with torch.no_grad():
            soc0 = {p.name: float(s) for p, s in zip(phases, 1 - (0.5 + 0.6 * torch.tanh(z)))}
    return TrainResult(model=model, loss=best, history=history, initial_soc=soc0)


def leave_one_group_out(phases):
    """Yield (group, training phases, test phases) for each test group (cross-validation by test)."""
    groups = list(dict.fromkeys(p.group for p in phases))
    for g in groups:
        yield g, [p for p in phases if p.group != g], [p for p in phases if p.group == g]
