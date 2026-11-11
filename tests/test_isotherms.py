import pytest
import torch

import psrnn
from psrnn import presets

T = torch.tensor([293.15, 323.15, 353.15], dtype=torch.float64)
P = torch.tensor([1500.0, 800.0, 400.0], dtype=torch.float64)


@pytest.mark.parametrize('iso', [presets.staid_isotherm(), psrnn.Langmuir(0.3, -2.5, 50000.0),
                                 psrnn.DubininAstakhov(0.38, 3600.0, 0.82)])
def test_pressure_inverts_loading(iso):
    with torch.no_grad():
        q = iso.loading(P, T)
        assert torch.all(q > 0) and torch.all(q < iso.q_max())
        assert torch.allclose(iso.pressure(q, T), P, rtol=1e-6)


@pytest.mark.parametrize('iso', [presets.staid_isotherm(), psrnn.DubininAstakhov(0.38, 3600.0, 0.82)])
def test_loading_increases_with_pressure_and_decreases_with_temperature(iso):
    with torch.no_grad():
        p = torch.linspace(100.0, 2000.0, 20, dtype=torch.float64)
        q20 = iso.loading(p, torch.full_like(p, 293.15))
        q60 = iso.loading(p, torch.full_like(p, 333.15))
    assert torch.all(torch.diff(q20) > 0)
    assert torch.all(q60 < q20)


def test_prior_is_zero_at_reference_and_grows():
    iso = psrnn.DubininAstakhov(0.38, 3600.0, 0.82)
    assert float(iso.prior().detach()) == pytest.approx(0.0, abs=1e-20)
    with torch.no_grad():
        iso.l_E += 0.1
    assert float(iso.prior().detach()) > 0.5
