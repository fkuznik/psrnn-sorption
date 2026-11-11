# Laboratory silica gel bed

A packed bed of 10 kg of silica gel crossed by 30 m3/h of air, charged at 80 °C and discharged at 20 °C,
70 % relative humidity. The measurements are simulated with a reference model (Dubinin-Astakhov isotherm
q0 = 0.38 kg/kg, E = 3600 J/mol, n = 0.82) plus sensor noise; the training starts from values about 10 %
off and never sees the state of charge.

```bash
python run_example.py --epochs 60
```

The script trains on tests 1 and 2 (four phases), observes the initial state of the two phases of test 3
on their first 2 h, and writes the predictions and `silica_gel_test.png` to `results/`. Expected output:
outlet temperature within 0.5 to 0.8 K and SoC within 0.05 to 0.06 on the test phases (about 3 min on a
laptop CPU).
