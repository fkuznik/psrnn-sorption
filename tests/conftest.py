from pathlib import Path

import pandas as pd
import pytest
import torch

import psrnn
from psrnn import presets
from psrnn.synthetic import inlet_profile, simulate

ROOT = Path(__file__).resolve().parents[1]
STAID_DATA = ROOT / 'examples' / 'staid' / 'data'


def load_staid():
    d = pd.read_csv(STAID_DATA / 'phases_mesurees.csv')
    d = d[~d.exclue_comparaison]
    return psrnn.phases_from_dataframe(
        d, 'module', dict(time='t_s', flow='Q_m3h', T_in='Tin_C', pv_in='pv_in_Pa', T_out='T_out_mes_C',
                          pv_out='pv_out_mes_Pa', soc='SoC'),
        group_col='essai', kind_col='phase', kind_map={'D': 'charge', 'H': 'discharge'},
        water0_cols=['m_w_z1_kg', 'm_w_z2_kg', 'm_w_z3_kg', 'm_w_z4_kg'], water0_total_col='m_w_kg',
        every=5, step=60.0)


@pytest.fixture(scope='session')
def staid_phases():
    return load_staid()


def small_bed(**kw):
    return psrnn.BedConfig(adsorbent_mass=2.0, water_min=0.06, water_max=0.56, cross_section=0.01,
                           n_slices=3, heat_of_adsorption=(2.3e6, 3.0e6), wall_capacity=1e3,
                           duct_time_constant=30.0, velocity_ref=0.17, loading_ref=0.15, **kw)


@pytest.fixture(scope='session')
def synthetic():
    """A small silica gel bed, its reference model and two simulated phases."""
    truth = psrnn.SorptionBedRNN(small_bed(ntu_mass=8.0, ntu_heat=15.0),
                                 psrnn.DubininAstakhov(0.38, 3600.0, 0.82), use_network=False)
    ph = [inlet_profile('C', 'charge', 3, 120.0, 6.0, 353.15, 1000.0, seed=1, group=1),
          inlet_profile('D', 'discharge', 3, 120.0, 6.0, 293.15, 1640.0, seed=2, group=2)]
    return truth, simulate(truth, ph, [0.2, 0.9], noise_T=0.0, noise_pv=0.0, Ts0=293.15)
