"""Ready-made configurations.

``staid``: one module of the full-scale STAID open zeolite 13X store (INSA Lyon, CETHIL), with the Toth
isotherm calibrated on the prototype, as used in the companion article of the code.
"""
from .config import BedConfig
from .isotherms import Toth

STAID_TOTH = dict(qm=0.436399, lb20=5.55205, Q=86382.8, t20=0.144617, c=0.0549505)
STAID_CHARGE_STATE = (453.15, 750.0)      # air at 180 degC, 750 Pa
STAID_DISCHARGE_TEMPERATURE = 293.15      # air at 20 degC, 70 % relative humidity


def staid_bed(n_slices=6):
    """One STAID module: 40.2 kg of zeolite 13X, bed cross-section 0.407 m2."""
    return BedConfig(adsorbent_mass=40.22646030103742, water_min=3.186521724090474,
                     water_max=12.200534435036275, cross_section=0.407, n_slices=n_slices)


def staid_isotherm():
    return Toth(**STAID_TOTH)


def staid_bounds():
    """(charge, discharge) reference states (T in K, pv in Pa) of the STAID state of charge."""
    from .physics import psat
    import torch
    T = STAID_DISCHARGE_TEMPERATURE
    return STAID_CHARGE_STATE, (T, 0.7 * float(psat(torch.tensor(T, dtype=torch.float64))))
