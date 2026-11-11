"""The package reproduces the companion code of the STAID article (Zenodo 10.5281/zenodo.23477614).

Reference values were produced with ``hybride_v3.py`` of that archive on the two demonstration phases:
``mode=test K=6 sub=1`` (untrained model) and ``mode=cv plis=8 seeds=1 epochs=5 K=6 sub=1`` with and
without state-of-charge labels.
"""
import pandas as pd
import pytest

import psrnn
from psrnn import presets

from conftest import ROOT

REF = pd.read_csv(ROOT / 'tests' / 'data' / 'staid_untrained_reference.csv')


def build():
    return psrnn.SorptionBedRNN(presets.staid_bed(6), presets.staid_isotherm())


def test_untrained_model_matches_reference(staid_phases):
    model = build()
    for p in staid_phases:
        pr = psrnn.predict(model, p, use_label=True)
        r = REF[REF.module == p.name]
        assert len(r) == len(pr)
        assert (pr.T_out_pred_C.values - r.T_out_hyb_C.values).__abs__().max() < 0.01      # 5 significant digits
        assert (pr.soc_pred.values - r.SoC_hyb.values).__abs__().max() < 2e-5


def test_training_matches_reference(staid_phases):
    train = [p for p in staid_phases if p.group != 8]
    r = psrnn.train(build, train, epochs=5, seed=800, verbose=False)
    assert r.loss == pytest.approx(1.49686, abs=2e-4)


def test_label_free_training_and_observer_match_reference(staid_phases):
    train = [p for p in staid_phases if p.group != 8]
    test = [p for p in staid_phases if p.group == 8]
    r = psrnn.train(build, train, epochs=5, seed=800, latent_init=True,
                    weights=psrnn.LossWeights(w_soc=0.0), verbose=False)
    assert r.loss == pytest.approx(1.19565, abs=2e-4)
    met, _ = psrnn.evaluate(r.model, test)
    assert met.RMSE_T_K.iloc[0] == pytest.approx(3.86, abs=0.01)
    assert met.RMSE_SoC.iloc[0] == pytest.approx(0.072, abs=0.001)
