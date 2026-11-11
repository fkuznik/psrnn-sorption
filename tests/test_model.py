import numpy as np
import torch

import psrnn
from psrnn.physics import humidity_ratio

from conftest import small_bed


def test_water_balance_is_exact(synthetic):
    """Water removed from the air equals water gained by the bed (no duct effect on humidity)."""
    truth, phases = synthetic
    model = psrnn.SorptionBedRNN(small_bed(), psrnn.DubininAstakhov(0.36, 3400.0, 0.9))
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.mlp.parameters():
            p += 0.3 * torch.randn_like(p)          # non-trivial network
        for ph in phases:
            b = psrnn.make_batch([ph], model.K)
            r = model(b, traces=True)
            ma = model.bed.rho_air * b.flow[0].clamp(min=model.bed.min_flow) / 3600
            uptake = (ma * b.dt * (humidity_ratio(b.pv_in[0]) - humidity_ratio(r['pv_out'][0]))).sum()
            gained = r['water'][0, -1].sum() - b.water0[0].sum()
            assert abs(float(uptake - gained)) < 1e-9 * float(b.water0[0].sum()) + 1e-12


def test_state_stays_physical(synthetic):
    truth, phases = synthetic
    with torch.no_grad():
        for ph in phases:
            b = psrnn.make_batch([ph], truth.K)
            r = truth(b, traces=True)
            q = r['water'] / (truth.bed.adsorbent_mass / truth.K)
            assert torch.all(q > 0) and torch.all(q < truth.isotherm.q_max())
            assert torch.all(torch.isfinite(r['T_out'])) and torch.all(r['pv_out'] > 0)


def test_charge_heats_out_and_discharge_releases_heat(synthetic):
    truth, (charge, discharge) = synthetic
    assert np.mean(discharge.T_out - discharge.T_in) > 3.0       # adsorption heats the air
    assert discharge.soc[-1] < discharge.soc[0]                     # the bed takes up water
    assert charge.soc[-1] > charge.soc[0]                           # the bed dries


def test_soc_water_roundtrip():
    model = psrnn.SorptionBedRNN(small_bed(), psrnn.DubininAstakhov(0.38, 3600.0, 0.82))
    soc = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)
    scale = torch.ones(3, dtype=torch.float64)
    w = model.water_from_soc(soc, scale)
    assert w.shape == (3, model.K)
    assert torch.allclose(model.soc_from_water(w.sum(1), scale), soc)


def test_batch_of_two_equals_two_single_runs(synthetic):
    truth, phases = synthetic
    with torch.no_grad():
        both = truth(psrnn.make_batch(phases, truth.K))
        for i, ph in enumerate(phases):
            one = truth(psrnn.make_batch([ph], truth.K))
            n = len(ph.time)
            assert torch.allclose(both['T_out'][i, :n], one['T_out'][0, :n])
