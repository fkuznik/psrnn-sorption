# API reference

All public names are importable from `psrnn`. Units: K, Pa, m3/h of air, s, kg.

## Configuration and data

`BedConfig(adsorbent_mass, water_min, water_max, cross_section, n_slices=6, cp_solid=900.0, heat_of_adsorption=(2.6e6, 4.6e6), ntu_mass=20.0, ntu_heat=40.0, loss_bed=1.5, loss_duct=1.5, wall_capacity=2e4, duct_time_constant=300.0, ambient_temperature=293.15, p_atm=101325.0, rho_air=1.2, min_flow=1.0, velocity_ref=0.13, loading_ref=0.2, temperature_ref=330.0, temperature_scale=60.0)`
: description of the bed and initial values of the physical parameters.

`Phase(name, time, flow, T_in, pv_in, T_out=None, pv_out=None, soc=None, water0=None, Ts0=None, scale=1.0, group=None, kind=None, step=None)`
: one phase at a constant step. `water0`: initial water, total or profile (kg). `scale`: fraction of the bed in use. `group`: test identifier. `kind`: `'charge'` or `'discharge'`. Method `subsample(every)`.

`phases_from_dataframe(df, phase_col, columns, celsius=True, group_col=None, kind_col=None, kind_map=None, water0_cols=None, water0_total_col=None, scale=None, every=1, step=None)`
: list of `Phase` from a long table.

`make_batch(phases, n_slices)` → `Batch`
: phases padded to a common length, with masks of the measured values.

`water_profile(water0, n_slices)`
: spreads a total or a profile over the slices, keeping the total.

## Isotherms

`Toth(qm, lb20, Q, t20, c, prior_widths=None)`, `Langmuir(qm, lb20, Q, prior_widths=None)`, `DubininAstakhov(q0, E, n, prior_widths=None)`
: methods `loading(p, T)`, `pressure(q, T)`, `q_max()`, `prior()`, `summary()`. `lb20` is log10 of the affinity at 20 °C (1/Pa), `Q` and `E` in J/mol.

## Model

`SorptionBedRNN(bed, isotherm, hidden=16, substeps=1, use_network=True)`
: `forward(batch, water0=None, traces=False)` returns a dict with `T_out`, `pv_out`, `soc` (B, L) and, with `traces`, `water` and `Ts` (B, L, K). Other methods: `water_from_soc(soc, scale)`, `soc_from_water(water, scale, bounds=None)`, `water_bounds(charge, discharge)`, `physical_parameters()`, `heat_of_adsorption()`.

## Training and estimation

`LossWeights(w_T=1, w_pv=1, w_soc=1, w_prior=0.1, s_T=5.0, s_pv=150.0, s_soc=0.05)`
: weights and scales of the loss terms.

`train(model, phases, epochs=150, lr=1e-2, seed=0, latent_init=False, weights=None, latent_lr=0.02, log_every=10, verbose=True)` → `TrainResult(model, loss, history, initial_soc)`
: `model` is a `SorptionBedRNN` or a function returning one.

`leave_one_group_out(phases)`
: yields `(group, train, test)`.

`observe_soc(model, phase, window_h=None)`, `observe_water_fraction(...)`
: initial SoC (or water fraction) of a phase from its outlet measurements.

## Prediction and metrics

`predict(model, phase, soc0=None, window_h=None, use_label=False, bounds=None)` → `DataFrame`
: initial state given, from the labels, or observed.

`evaluate(model, phases, **predict_kw)` → `(metrics, predictions)`

`phase_metrics(phase, prediction)`, `exchanged_energy(time, flow, T_in, T_out, kind)`, `heat_power(flow, T_in, T_out)`

`conformal_quantile(scores, alpha=0.1)`, `conformal_interval(cal_residuals, prediction, alpha=0.1, cal_scale=None, scale=None)`, `adaptive_scale(x, time, s0, half_window=1800.0, slope=0.1)`

## Synthetic data and presets

`psrnn.synthetic.inlet_profile(name, kind, hours, dt, flow, T_in, pv_in, seed=0, steps=3, spread=0.15, group=None)`, `psrnn.synthetic.simulate(model, phases, soc0, noise_T=0.3, noise_pv=20.0, Ts0=None, seed=0)`

`psrnn.presets.staid_bed(n_slices=6)`, `staid_isotherm()`, `staid_bounds()`
