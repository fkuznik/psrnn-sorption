# The model

## State and balances

The bed holds a mass $m_z$ of dry adsorbent, split into $K$ slices of equal mass along the air path.
The state of slice $k$ is its water content $m_k$ (kg) and its solid temperature $T_{s,k}$ (K). The
air enters slice 1 at the measured inlet conditions; the air leaving slice $k$ enters slice $k+1$.

Over a step $\Delta t$, slice $k$ exchanges water and heat with the air through two capacities

$$G_k = \dot m_a \Delta t \,(1 - e^{-\mathrm{NTU}_m / K}), \qquad E_k = \dot m_a c_a \Delta t \,(1 - e^{-\mathrm{NTU}_h / K}),$$

where $\dot m_a$ is the dry-air flow rate. The effectiveness form keeps the exchange stable at any time
step: the air can approach the equilibrium of the slice but cannot overshoot it. The water taken up by
the slice is

$$\Delta m_k = G_k \,\big(w_k - w_{eq}(m_k + \Delta m_k, T_{s,k} + \Delta T_{s,k})\big),$$

with $w$ the humidity ratio of the air and $w_{eq}$ the equilibrium humidity ratio given by the
isotherm. The energy balance of the slice is

$$C_k \Delta T_{s,k} = E_k (T_k - T_{s,k}) + \Delta H \,\Delta m_k - U_k \big(1 + \beta (T_{s,k} - T_a)/100\big)(T_{s,k} - T_a)\Delta t,$$

with $C_k$ the heat capacity of the adsorbent, of its water and of its share of the wall, $\Delta H$
the heat of adsorption and $U_k$ the heat loss coefficient. The two equations are solved together by
linearising $w_{eq}$ around the current state (semi-implicit step), then the state is clipped so that
the slice takes no more water than the air brings and stays below the capacity of the isotherm. The air
leaves the slice with

$$T_{k+1} = T_k - (1 - e^{-\mathrm{NTU}_h/K})(T_k - T_{s,k}), \qquad w_{k+1} = w_k - \Delta m_k / (\dot m_a \Delta t),$$

so that the water removed from the air equals the water gained by the bed at every step (this is
checked by `tests/test_model.py`). The outlet duct adds a heat loss that grows with the temperature
difference to ambient and a first-order lag of time constant $\tau$.

## Learnt closures

| Closure | Parameters | Constraint |
|---|---|---|
| isotherm | Toth: $q_m, b_{20}, Q, t_{20}, c$; Langmuir: $q_m, b_{20}, Q$; Dubinin-Astakhov: $q_0, E, n$ | quadratic prior towards the reference values |
| transfer units | $\mathrm{NTU}_m$, $\mathrm{NTU}_h$ of the bed | positive (logarithm) |
| transfer modulation | network 4-16-2 (superficial velocity, loading, solid and air temperature), shared by the slices | output initialised at zero, factor $e^{g}$ with $g \in [-3, 3]$ |
| heat of adsorption | $\Delta H$ | within the bounds of `BedConfig.heat_of_adsorption` |
| losses | $U$, $\beta$ (bed), $U_d$, $\gamma$ (duct) | positive |
| wall and duct | wall capacity, $\tau$ | positive |

With the network at zero, the model is a purely physical one-dimensional bed model; training can start
from it and only adds the corrections that the measurements support. `use_network=False` keeps the
physical model only.

## State of charge

The SoC is defined on the water of the whole bed:

$$\mathrm{SoC} = 1 - \frac{\sum_k m_k - m_{min}}{m_{max} - m_{min}},$$

with $m_{min}$ and $m_{max}$ the water of the bed at the charged and discharged reference states
(`BedConfig.water_min`, `water_max`). `SorptionBedRNN.water_bounds` gives the same bounds on the
learnt isotherm, and `predict(..., bounds=...)` adds the SoC computed on them.

## Training and observer

The loss compares the predicted outlet temperature and vapour pressure with the measurements, and the
SoC with its labels when they exist (`LossWeights`). With `latent_init=True`, the initial water fraction
of each training phase is a free parameter, started from the observer and learnt with the model. The
observer (`observe_soc`) searches the uniform initial water fraction that best reproduces the outlet
signals over a chosen window (grid of 45 values, then refinement).

## Prediction intervals

`conformal_interval` turns the residuals of phases unseen by the model (for example the out-of-fold
predictions of a leave-one-test-out cross-validation) into prediction intervals of chosen coverage,
of constant width or scaled by `adaptive_scale`, which widens them in the fronts.
