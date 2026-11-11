"""Description of an open packed-bed sorption heat store."""
from dataclasses import dataclass


@dataclass
class BedConfig:
    """Geometry, reference states and initial values of the trainable physical parameters.

    The state of charge is defined on the water content of the whole bed:
    SoC = 1 - (m - water_min) / (water_max - water_min), so that SoC = 1 when the bed
    holds ``water_min`` (charged, dry) and SoC = 0 when it holds ``water_max``.
    """
    adsorbent_mass: float              # kg of dry adsorbent in the bed
    water_min: float                   # kg of water at the charged reference state (SoC = 1)
    water_max: float                   # kg of water at the discharged reference state (SoC = 0)
    cross_section: float               # m2, used to compute the superficial air velocity
    n_slices: int = 6                  # number of slices along the air path
    cp_solid: float = 900.0            # J/(kg K), dry adsorbent
    heat_of_adsorption: tuple = (2.6e6, 4.6e6)   # J/kg water, bounds of the trainable value
    ntu_mass: float = 20.0             # initial number of mass transfer units of the whole bed
    ntu_heat: float = 40.0             # initial number of heat transfer units of the whole bed
    loss_bed: float = 1.5              # W/K, initial heat loss coefficient of the bed near ambient
    loss_duct: float = 1.5             # W/K, initial heat loss coefficient of the outlet duct
    wall_capacity: float = 2e4         # J/K, initial heat capacity of the wall and internals
    duct_time_constant: float = 300.0  # s, initial time constant of the outlet duct
    ambient_temperature: float = 293.15   # K
    p_atm: float = 101325.0            # Pa
    rho_air: float = 1.2               # kg/m3 of dry air at the flow meter
    min_flow: float = 1.0              # m3/h, floor applied to the flow rate (stopped fan)
    # scales of the inputs of the transfer network
    velocity_ref: float = 0.13         # m/s
    loading_ref: float = 0.2           # kg/kg
    temperature_ref: float = 330.0     # K
    temperature_scale: float = 60.0    # K
