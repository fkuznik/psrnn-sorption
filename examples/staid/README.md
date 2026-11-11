# STAID full-scale zeolite 13X store

One module of the STAID prototype (INSA Lyon, CETHIL): 40 kg of zeolite 13X beads in a 0.72 m diameter
cassette, Toth isotherm calibrated on the prototype (`psrnn.presets`).

`data/phases_mesurees.csv` holds two measured phases (charge 4D and discharge 8H, lower module, 1 min
step). Columns: module, test (`essai`), phase (`D` charge, `H` discharge), arrangement, time (s), flow
rate per module (m3/h), inlet temperature (°C), inlet and outlet vapour pressure (Pa), measured outlet
temperature (°C). The columns `m_w_*`, `x_eau`, `SoC`, `E_rest_kWh` and `Ts_moy_C` are reference
trajectories computed with the calibrated axisymmetric reactor model of the prototype; they serve as
labels and as the reference of the SoC metrics. `data/bornes.json` gives the water of the bed at the two
reference states and the zeolite mass. The other 26 measured phases are available from F. Kuznik on
reasonable request.

```bash
python run_staid.py --epochs 80
```

With a single training phase the run checks the workflow; it does not reproduce the accuracy obtained
with 21 to 25 training phases per fold.
