import numpy as np
import pytest
import torch

import psrnn
from psrnn.synthetic import inlet_profile

from conftest import small_bed


def test_water_profile_keeps_total():
    for z, K in (([1.0, 2.0, 3.0, 4.0], 6), ([1.0, 2.0, 3.0, 4.0], 2), ([1.0, 2.0, 3.0], 5), (2.4, 4)):
        prof = psrnn.water_profile(z, K)
        assert len(prof) == K and prof.sum() == pytest.approx(np.sum(z))


def test_batch_padding_and_masks():
    a = inlet_profile('a', 'charge', 1, 60.0, 5.0, 353.15, 1000.0)
    b = inlet_profile('b', 'charge', 2, 60.0, 5.0, 353.15, 1000.0)
    a.T_out = np.full(len(a.time), 300.0); a.T_out[3] = np.nan
    batch = psrnn.make_batch([a, b], 3)
    assert batch.flow.shape == (2, len(b.time))
    assert batch.mask[0].sum() == len(a.time)
    assert batch.mask_T[0].sum() == len(a.time) - 1 and batch.mask_T[1].sum() == 0
    assert batch.water0 is None


def test_variable_time_step_is_rejected():
    p = inlet_profile('a', 'charge', 1, 60.0, 5.0, 353.15, 1000.0)
    p.time = p.time ** 1.1
    with pytest.raises(ValueError):
        p.dt


def test_observer_recovers_initial_soc(synthetic):
    truth, phases = synthetic
    for p, s0 in zip(phases, (0.2, 0.9)):
        assert psrnn.observe_soc(truth, p) == pytest.approx(s0, abs=0.02)


def test_label_free_training_reduces_the_loss(synthetic):
    truth, phases = synthetic
    def build():
        return psrnn.SorptionBedRNN(small_bed(), psrnn.DubininAstakhov(0.34, 3300.0, 0.9))
    r = psrnn.train(build, phases, epochs=8, latent_init=True, weights=psrnn.LossWeights(w_soc=0.0), verbose=False)
    assert r.history[-1]['loss'] < r.history[0]['loss']
    assert set(r.initial_soc) == {'C', 'D'}


def test_evaluate_returns_metrics(synthetic):
    truth, phases = synthetic
    met, pred = psrnn.evaluate(truth, phases, use_label=True)
    assert met.RMSE_T_K.max() < 1e-9 and met.RMSE_SoC.max() < 1e-9
    assert {'T_out_pred_C', 'soc_pred', 'soc_label'} <= set(pred.columns)


def test_exchanged_energy_and_power():
    t = np.linspace(0, 3600, 61); Q = np.full(61, 100.0)
    e = psrnn.exchanged_energy(t, Q, np.full(61, 293.15), np.full(61, 313.15), 'discharge')
    assert e == pytest.approx(1.2 * 100 / 3600 * 1005 * 20 * 3600 / 3.6e6)
    assert psrnn.heat_power(Q, 293.15, 313.15)[0] == pytest.approx(1.2 * 100 / 3600 * 1005 * 20)


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    cal, new = rng.standard_normal(2000), rng.standard_normal(20000)
    lo, hi, q = psrnn.conformal_interval(cal, np.zeros_like(new), alpha=0.1)
    cover = np.mean((new >= lo) & (new <= hi))
    assert 0.88 < cover < 0.92
    assert q == pytest.approx(1.645, abs=0.1)
