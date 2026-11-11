"""Moist-air properties and physical constants used by the bed model.

All temperatures are in kelvin, pressures in pascal, flow rates in m3/h of air.
"""
import torch

T0K = 273.15          # K
P_ATM = 101325.0      # Pa
CP_AIR = 1005.0       # J/(kg K), dry air
CP_WATER = 4180.0     # J/(kg K), adsorbed water
R_GAS = 8.314         # J/(mol K)
EPS = 0.622           # ratio of the molar masses of water and dry air


def psat(T):
    """Saturation vapour pressure of water (Pa) at temperature T (K), Magnus formula."""
    Tc = torch.as_tensor(T) - T0K
    return 610.94 * torch.exp(17.625 * Tc / (Tc + 243.04))


def humidity_ratio(pv, p=P_ATM):
    """Humidity ratio (kg water / kg dry air) from the vapour pressure pv (Pa)."""
    return EPS * pv / (p - pv)


def vapour_pressure(w, p=P_ATM):
    """Vapour pressure (Pa) from the humidity ratio w (kg/kg)."""
    return p * w / (EPS + w)
