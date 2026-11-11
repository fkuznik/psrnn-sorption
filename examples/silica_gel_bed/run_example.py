"""Example away from STAID: a laboratory packed bed of 10 kg of silica gel with a Dubinin-Astakhov isotherm.

The "measurements" are simulated with a reference model whose parameters the training does not know
(isotherm, transfer units, heat of adsorption), plus sensor noise. The model is then trained without any
state-of-charge label, from the outlet temperature and humidity of four phases, and tested on two unseen
phases whose initial state is estimated by the observer. Because the truth is known, the script reports
how well the state of charge is recovered.

Run: python run_example.py [--epochs 60] [--out results]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import psrnn
from psrnn.synthetic import inlet_profile, simulate

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--out', default='results')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ the store
TRUE_DA = dict(q0=0.38, E=3600.0, n=0.82)            # silica gel and water, kg/kg, J/mol, -
CHARGE, DISCHARGE = (353.15, 1000.0), (293.15, 1640.0)   # air at 80 degC / 1000 Pa and 20 degC / 70 % RH
iso = psrnn.DubininAstakhov(**TRUE_DA)
with torch.no_grad():                                # water content at the two reference states (kg/kg)
    q_lo = float(iso.loading(torch.tensor(CHARGE[1]), torch.tensor(CHARGE[0])))
    q_hi = float(iso.loading(torch.tensor(DISCHARGE[1]), torch.tensor(DISCHARGE[0])))
mass = 10.0


def bed(**kw):
    return psrnn.BedConfig(adsorbent_mass=mass, water_min=q_lo * mass, water_max=q_hi * mass, cross_section=0.05,
                           n_slices=4, cp_solid=920.0, heat_of_adsorption=(2.3e6, 3.0e6), wall_capacity=5e3,
                           loss_bed=0.5, loss_duct=0.3, duct_time_constant=60.0, velocity_ref=0.17,
                           loading_ref=0.15, **kw)


# reference model ("truth"): physical parameters unknown to the training
truth = psrnn.SorptionBedRNN(bed(ntu_mass=8.0, ntu_heat=15.0), psrnn.DubininAstakhov(**TRUE_DA), use_network=False)
with torch.no_grad():
    truth.dH_z.fill_(0.8)                             # heat of adsorption 2.78 MJ/kg

# ------------------------------------------------------------------ six simulated phases (three tests)
rng = np.random.default_rng(3)
phases, soc0 = [], []
for test in (1, 2, 3):
    for kind, T_in, pv_in, s0 in (('charge', CHARGE[0], CHARGE[1], rng.uniform(0.0, 0.3)),
                                  ('discharge', DISCHARGE[0], DISCHARGE[1], rng.uniform(0.7, 1.0))):
        phases.append(inlet_profile(f'{test}{kind[0].upper()}', kind, hours=8, dt=120.0, flow=30.0,
                                    T_in=T_in, pv_in=pv_in, seed=10 * test + len(phases), group=test))
        soc0.append(s0)
measured = simulate(truth, phases, soc0, noise_T=0.3, noise_pv=20.0, Ts0=293.15, seed=1)
for p in measured:
    p.soc_true, p.soc = p.soc, None                  # hide the state of charge from the training

# ------------------------------------------------------------------ training without SoC labels
train_set = [p for p in measured if p.group != 3]
test_set = [p for p in measured if p.group == 3]
guess = dict(q0=0.34, E=3300.0, n=0.9)               # literature-like values, about 10 % off
widths = dict(q0=0.2, E=0.2, n=0.2)


def build():
    return psrnn.SorptionBedRNN(bed(), psrnn.DubininAstakhov(**guess, prior_widths=widths))


res = psrnn.train(build, train_set, epochs=args.epochs, lr=1e-2, latent_init=True,
                  weights=psrnn.LossWeights(w_soc=0.0, w_prior=0.1), log_every=10)
model = res.model
torch.save(model.state_dict(), out / 'model.pt')

# ------------------------------------------------------------------ test on unseen phases
rows = []
for p in test_set:
    pred = psrnn.predict(model, p, window_h=2.0)     # initial state observed on the first 2 h
    pred['soc_true'] = p.soc_true
    pred.to_csv(out / f'prediction_{p.name}.csv', index=False, float_format='%.5g')
    rmse_T = float(np.sqrt(np.mean((pred.T_out_pred_C + 273.15 - p.T_out) ** 2)))
    rmse_soc = float(np.sqrt(np.mean((pred.soc_pred - p.soc_true) ** 2)))
    rows.append((p.name, p.kind, pred.soc0.iloc[0], p.soc_true[0], rmse_T, rmse_soc))
    print(f'{p.name} ({p.kind}): SoC0 observed {pred.soc0.iloc[0]:.2f} / true {p.soc_true[0]:.2f}, '
          f'RMSE T {rmse_T:.2f} K, RMSE SoC {rmse_soc:.3f}')
print('learnt parameters:', {k: round(v, 3) for k, v in model.physical_parameters().items()})
print('true isotherm:', TRUE_DA)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(test_set), figsize=(4.5 * len(test_set), 5.5), sharex=True)
    for j, p in enumerate(test_set):
        pr = psrnn.predict(model, p, window_h=2.0); h = p.time / 3600
        ax[0, j].plot(h, p.T_out - 273.15, '.', ms=2, color='0.5', label='measured')
        ax[0, j].plot(h, pr.T_out_pred_C, color='C0', label='model')
        ax[0, j].set_title(f'{p.name} ({p.kind})'); ax[0, j].set_ylabel('Outlet temperature (°C)')
        ax[1, j].plot(h, p.soc_true, color='0.3', label='true')
        ax[1, j].plot(h, pr.soc_pred, color='C1', label='estimated')
        ax[1, j].set_xlabel('Time (h)'); ax[1, j].set_ylabel('State of charge (-)')
    ax[0, 0].legend(); ax[1, 0].legend()
    fig.tight_layout(); fig.savefig(out / 'silica_gel_test.png', dpi=150)
    print('figure:', out / 'silica_gel_test.png')
except ImportError:
    pass
