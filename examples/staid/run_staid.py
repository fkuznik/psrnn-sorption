"""STAID example: the full-scale open zeolite 13X store of INSA Lyon (one module, 40 kg of zeolite).

Uses the two demonstration phases shipped in ``data/`` (charge 4D and discharge 8H, lower module, 1 min
step, resampled at 5 min). The other 26 measured phases of the companion article are available from the
author on reasonable request.

1. untrained model (calibrated Toth isotherm, network at zero) on both phases;
2. training on the charge without state-of-charge labels, test on the discharge with the observer.

With a single training phase the run checks the workflow: it does not reproduce the accuracy of the
article, which trains on 21 to 25 phases per fold.

Run: python run_staid.py [--epochs 80] [--out results]
"""
import argparse
from pathlib import Path

import pandas as pd

import psrnn
from psrnn import presets

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--out', default='results')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(Path(__file__).parent / 'data' / 'phases_mesurees.csv')
d = d[~d.exclue_comparaison]
phases = psrnn.phases_from_dataframe(
    d, 'module', dict(time='t_s', flow='Q_m3h', T_in='Tin_C', pv_in='pv_in_Pa', T_out='T_out_mes_C',
                      pv_out='pv_out_mes_Pa', soc='SoC'),
    group_col='essai', kind_col='phase', kind_map={'D': 'charge', 'H': 'discharge'},
    water0_cols=['m_w_z1_kg', 'm_w_z2_kg', 'm_w_z3_kg', 'm_w_z4_kg'], water0_total_col='m_w_kg',
    every=5, step=60.0)
bounds = presets.staid_bounds()


def build():
    return psrnn.SorptionBedRNN(presets.staid_bed(n_slices=6), presets.staid_isotherm())


met0, _ = psrnn.evaluate(build(), phases, use_label=True)
print('Untrained model, initial state from the labels:')
print(met0[['phase', 'kind', 'RMSE_T_K', 'RMSE_pv_pct', 'RMSE_SoC']].round(3).to_string(index=False))

train = [p for p in phases if p.group != 8]
test = [p for p in phases if p.group == 8]
res = psrnn.train(build, train, epochs=args.epochs, latent_init=True, weights=psrnn.LossWeights(w_soc=0.0))
met, pred = psrnn.evaluate(res.model, test, bounds=bounds)
print('\nTrained on test 4 without SoC labels, tested on test 8 (observer on the whole phase):')
print(met[['phase', 'soc0', 'RMSE_T_K', 'RMSE_pv_pct', 'RMSE_SoC', 'err_E_pct']].round(3).to_string(index=False))
print('learnt parameters:', {k: round(v, 3) for k, v in res.model.physical_parameters().items()})
met.to_csv(out / 'metrics.csv', index=False)
pred.to_csv(out / 'predictions.csv', index=False, float_format='%.5g')
