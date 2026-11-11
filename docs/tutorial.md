# Tutorial: from measurements of a store to its state of charge

This tutorial follows `examples/silica_gel_bed/run_example.py`. It takes about three minutes on a laptop.

## 1. Describe the bed

```python
import psrnn

bed = psrnn.BedConfig(adsorbent_mass=10.0, water_min=0.29, water_max=2.79, cross_section=0.05,
                      n_slices=4, cp_solid=920.0, heat_of_adsorption=(2.3e6, 3.0e6),
                      wall_capacity=5e3, loss_bed=0.5, loss_duct=0.3, duct_time_constant=60.0,
                      velocity_ref=0.17, loading_ref=0.15)
```

`water_min` and `water_max` fix the meaning of the SoC: the water of the bed in equilibrium with the
charge air (here 80 °C, 1000 Pa) and with the discharge air (20 °C, 70 % relative humidity). Compute
them from the isotherm of the adsorbent, `isotherm.loading(p, T) * adsorbent_mass`. The other values
initialise the trainable parameters; rough estimates are enough. `velocity_ref` and `loading_ref`
only scale the inputs of the transfer network: take typical values of the store.

## 2. Choose the isotherm

```python
isotherm = psrnn.DubininAstakhov(q0=0.34, E=3300.0, n=0.9, prior_widths=dict(q0=0.2, E=0.2, n=0.2))
```

The values are the starting point and the centre of the prior; `prior_widths` sets how far training
may move them (relative widths for Dubinin-Astakhov). A calibrated isotherm deserves narrow widths, a
literature one wider widths. To use another isotherm, subclass `psrnn.Isotherm` and implement
`loading(p, T)`, `pressure(q, T)`, `q_max()` and `prior()`.

## 3. Load the phases

A phase is one charge or one discharge at a constant time step. From a table with one row per step:

```python
phases = psrnn.phases_from_dataframe(
    df, phase_col='phase',
    columns=dict(time='t_s', flow='Q_m3h', T_in='T_in_C', pv_in='pv_in_Pa',
                 T_out='T_out_C', pv_out='pv_out_Pa'),
    group_col='test', kind_col='kind', kind_map={'D': 'charge', 'H': 'discharge'},
    every=5, step=60.0)
```

Temperatures are read in °C and converted to K. Missing outlet humidity (NaN) is allowed: the loss
ignores it. `every=5` resamples 1 min data at 5 min; `step` gives the nominal logging step when the
time stamps jitter. If the SoC or the initial water profile is known (from a reference model, or a
weighing), give the `soc` column and `water0_cols`.

## 4. Train without SoC labels

```python
build = lambda: psrnn.SorptionBedRNN(bed, isotherm)
res = psrnn.train(build, train_phases, epochs=60, lr=1e-2, latent_init=True,
                  weights=psrnn.LossWeights(w_soc=0.0))
```

Passing a function that builds the model makes the run reproducible for a given `seed`. Training
keeps the best state. `res.initial_soc` gives the initial SoC learnt for each training phase and
`res.model.physical_parameters()` the learnt parameters: check that they are physically sensible (for
example the heat of adsorption against the literature) before trusting the SoC.

For an ensemble, train with several seeds: each seed perturbs the starting point.

## 5. Estimate the state of charge of a new phase

```python
pred = psrnn.predict(res.model, new_phase, window_h=2.0)
```

The observer fits the initial state on the first two hours, then the model runs over the whole phase.
`pred` contains the predicted outlet temperature and humidity, the SoC and the water of the bed.
`psrnn.evaluate` does the same for a list of phases and returns the accuracy metrics. For an online
use, call the observer once at the start of a phase, then run the model step by step with the
measured inlet.

## 6. Cross-validate and attach intervals

```python
for group, train_set, test_set in psrnn.leave_one_group_out(phases):
    res = psrnn.train(build, train_set, latent_init=True, weights=psrnn.LossWeights(w_soc=0.0))
    met, pred = psrnn.evaluate(res.model, test_set)
```

Leaving out whole tests gives honest errors, because the phases of a test share their sensors and
their initial bed state. The out-of-fold residuals of the other tests calibrate the prediction
intervals of a test with `psrnn.conformal_interval`.

## What the example gives

With four simulated training phases and sensor noise of 0.3 K and 20 Pa, the model recovers the outlet
temperature of the two test phases within 0.5 to 0.8 K and their SoC within 0.05 to 0.06; most of the
SoC error is an offset of the initial state given by the observer.
