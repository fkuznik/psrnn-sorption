"""Synthetic phases: piecewise-constant inlet conditions and outlets simulated with a reference model.

Used by the tests and the examples, and to check what the training recovers when the truth is known.
"""
import numpy as np
import torch

from .data import Phase, make_batch
from .physics import psat


def inlet_profile(name, kind, hours, dt, flow, T_in, pv_in, seed=0, steps=3, spread=0.15, group=None):
    """Phase with piecewise-constant inlet conditions around (flow, T_in, pv_in), changed ``steps`` times."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, hours * 3600.0 + dt / 2, dt)
    edges = np.sort(rng.uniform(0.15, 0.85, steps - 1)) * t[-1] if steps > 1 else []
    seg = np.searchsorted(edges, t)
    f = 1 + spread * rng.uniform(-1, 1, (steps, 3))
    Q = flow * f[seg, 0]
    Tin = 293.15 + (T_in - 293.15) * f[seg, 1]
    pv = np.minimum(pv_in * f[seg, 2], 0.9 * psat(torch.tensor(Tin)).numpy())
    return Phase(name=name, time=t, flow=Q, T_in=Tin, pv_in=pv, kind=kind, group=group)


def simulate(model, phases, soc0, noise_T=0.3, noise_pv=20.0, Ts0=None, seed=0):
    """Fill T_out, pv_out and SoC labels of ``phases`` with the outputs of ``model`` started from ``soc0``
    (one value per phase), plus Gaussian measurement noise. Returns new Phase objects."""
    rng = np.random.default_rng(seed)
    out = []
    for p, s0 in zip(phases, soc0):
        ts0 = Ts0 if Ts0 is not None else (p.Ts0 if p.Ts0 is not None else p.T_in[0])
        p = Phase(**{**p.__dict__, 'Ts0': float(ts0)})
        b = make_batch([p], model.K)
        w0 = model.water_from_soc(torch.tensor([float(s0)], dtype=torch.float64), b.scale)
        with torch.no_grad():
            r = model(b, w0)
        n = len(p.time)
        T = r['T_out'][0, :n].numpy(); pv = r['pv_out'][0, :n].numpy()
        out.append(Phase(**{**p.__dict__,
                            'T_out': T + noise_T * rng.standard_normal(n),
                            'pv_out': pv + noise_pv * rng.standard_normal(n),
                            'soc': r['soc'][0, :n].numpy(),
                            'water0': w0[0].numpy().copy()}))
    return out
