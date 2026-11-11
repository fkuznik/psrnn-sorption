"""Observer of the initial state: the initial water content that best reproduces the measured outlet."""
import numpy as np
import torch

from .data import make_batch


def observe_water_fraction(model, phase, window_h=None, s_T=5.0, s_pv=150.0):
    """Water fraction x = 1 - SoC at the start of ``phase`` that reproduces the measured outlet temperature
    and vapour pressure over the first ``window_h`` hours (whole phase if None).

    Grid search on [-0.05, 1.05] (45 points), then refinement (11 points over +/- 0.025). The initial
    profile is uniform along the bed. Needs ``phase.T_out``; ``phase.pv_out`` is used where available.
    """
    if phase.T_out is None:
        raise ValueError(f'{phase.name}: the observer needs the measured outlet temperature')
    batch = make_batch([phase], model.K)
    t = phase.time - phase.time[0]
    n = len(t) if window_h is None else max(3, int(np.searchsorted(t, window_h * 3600.0, side='right')))
    bw = batch.window(n)

    def cost(xs):
        G = len(xs)
        b = bw.repeat(G)
        pred = model(b, model.water_from_soc(1 - xs, b.scale))
        lT = ((pred['T_out'] - b.T_out) ** 2 * b.mask_T).sum(1) / b.mask_T.sum(1).clamp(min=1) / s_T ** 2
        lpv = ((pred['pv_out'] - b.pv_out) ** 2 * b.mask_pv).sum(1) / b.mask_pv.sum(1).clamp(min=1) / s_pv ** 2
        return lT + lpv

    with torch.no_grad():
        xs = torch.linspace(-0.05, 1.05, 45, dtype=torch.float64)
        x = xs[int(torch.argmin(cost(xs)))]
        xs = x + torch.linspace(-0.025, 0.025, 11, dtype=torch.float64)
        x = xs[int(torch.argmin(cost(xs)))]
    return float(x)


def observe_soc(model, phase, window_h=None):
    """Initial state of charge of ``phase`` estimated from its outlet measurements."""
    return 1.0 - observe_water_fraction(model, phase, window_h)
