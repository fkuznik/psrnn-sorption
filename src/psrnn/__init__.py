"""psrnn: physics-structured recurrent networks for open sorption heat stores.

Estimate the state of charge and the heat power of a packed-bed sorption store from its inlet and
outlet air measurements, with a recurrent model whose structure carries the water and energy balances.
"""
from .config import BedConfig
from .data import Batch, Phase, make_batch, phases_from_dataframe, water_profile
from .isotherms import DubininAstakhov, Isotherm, Langmuir, Toth
from .metrics import (adaptive_scale, conformal_interval, conformal_quantile, evaluate, exchanged_energy,
                      heat_power, phase_metrics, predict)
from .model import SorptionBedRNN
from .observer import observe_soc, observe_water_fraction
from .training import LossWeights, TrainResult, leave_one_group_out, train

__version__ = '0.1.0'

__all__ = ['BedConfig', 'Batch', 'Phase', 'make_batch', 'phases_from_dataframe', 'water_profile',
           'Isotherm', 'Toth', 'Langmuir', 'DubininAstakhov', 'SorptionBedRNN', 'observe_soc',
           'observe_water_fraction', 'LossWeights', 'TrainResult', 'train', 'leave_one_group_out',
           'predict', 'evaluate', 'phase_metrics', 'exchanged_energy', 'heat_power', 'conformal_quantile',
           'conformal_interval', 'adaptive_scale', '__version__']
