"""Measured or simulated operating phases of a store, and padding of phases into batches."""
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import torch

from .physics import T0K


@dataclass
class Phase:
    """One charge or discharge phase at a constant time step.

    Temperatures in K, vapour pressures in Pa, flow rate in m3/h of air, time in s.
    Optional fields may be None (no measurement) or contain NaN (missing values).
    """
    name: str
    time: np.ndarray
    flow: np.ndarray
    T_in: np.ndarray
    pv_in: np.ndarray
    T_out: Optional[np.ndarray] = None      # measured outlet temperature
    pv_out: Optional[np.ndarray] = None     # measured outlet vapour pressure
    soc: Optional[np.ndarray] = None        # state-of-charge labels, if any
    water0: Optional[np.ndarray] = None     # initial water (kg): total, or profile along the air path
    Ts0: Optional[float] = None             # initial solid temperature (K); default: first outlet or inlet temperature
    scale: float = 1.0                      # fraction of the nominal bed in use (0.5 for a half bed)
    group: Optional[object] = None          # test identifier, used for leave-one-test-out splits
    kind: Optional[str] = None              # 'charge' or 'discharge'
    step: Optional[float] = None            # nominal time step (s); default: mean of the time differences
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in ('time', 'flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc'):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, np.asarray(v, dtype=float))
        n = len(self.time)
        for k in ('flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc'):
            v = getattr(self, k)
            if v is not None and len(v) != n:
                raise ValueError(f'{self.name}: {k} has {len(v)} values, time has {n}')

    @property
    def dt(self):
        if self.step is not None:
            return float(self.step)
        d = np.diff(self.time)
        if len(d) == 0:
            raise ValueError(f'{self.name}: at least two time steps are needed')
        if np.ptp(d) > 1e-3 * np.mean(d) + 1.0:
            raise ValueError(f'{self.name}: the time step must be constant (resample the phase first)')
        return float(np.mean(d))

    def subsample(self, every):
        """Keep one row in ``every`` (for example 1 min data -> 5 min with every=5)."""
        kw = {k: (getattr(self, k)[::every] if getattr(self, k) is not None else None)
              for k in ('time', 'flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc')}
        if self.step is not None:
            kw['step'] = self.step * every
        return replace(self, **kw)

    def initial_temperature(self):
        if self.Ts0 is not None:
            return float(self.Ts0)
        return float(self.T_out[0]) if self.T_out is not None else float(self.T_in[0])


def water_profile(water0, n_slices):
    """Spread an initial water content (scalar total or profile) over ``n_slices`` slices, keeping the total."""
    z = np.atleast_1d(np.asarray(water0, dtype=float))
    total = z.sum()
    if len(z) == 1:
        return np.full(n_slices, total / n_slices)
    if n_slices >= len(z):
        prof = np.repeat(z, int(np.ceil(n_slices / len(z))))[:n_slices]
    elif len(z) % n_slices == 0:
        prof = z.reshape(n_slices, -1).sum(1)
    else:
        x = np.linspace(0, 1, len(z)); xs = np.linspace(0, 1, n_slices)
        prof = np.interp(xs, x, z)
    return prof / prof.sum() * total


@dataclass
class Batch:
    """Phases padded to a common length (B phases, L steps). ``mask`` marks the real steps."""
    flow: torch.Tensor
    T_in: torch.Tensor
    pv_in: torch.Tensor
    T_out: torch.Tensor
    pv_out: torch.Tensor
    soc: torch.Tensor
    mask: torch.Tensor          # 1 on real steps
    mask_T: torch.Tensor        # 1 where T_out is measured
    mask_pv: torch.Tensor       # 1 where pv_out is measured
    mask_soc: torch.Tensor      # 1 where a SoC label exists
    water0: Optional[torch.Tensor]   # (B, K) or None when an initial water content is missing
    Ts0: torch.Tensor
    scale: torch.Tensor
    dt: float
    names: list

    def __len__(self):
        return self.flow.shape[0]

    def window(self, n):
        """First ``n`` steps of the batch (used by the observer)."""
        kw = {k: getattr(self, k)[:, :n] for k in ('flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc',
                                                   'mask', 'mask_T', 'mask_pv', 'mask_soc')}
        return replace(self, **kw)

    def repeat(self, G):
        """Batch of one phase repeated G times (used by the observer)."""
        kw = {k: getattr(self, k).expand(G, -1) for k in ('flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc',
                                                          'mask', 'mask_T', 'mask_pv', 'mask_soc')}
        kw.update(Ts0=self.Ts0.expand(G), scale=self.scale.expand(G),
                  water0=None if self.water0 is None else self.water0.expand(G, -1))
        return replace(self, **kw)


def make_batch(phases, n_slices):
    """Pad a list of phases (same time step) into a Batch; inputs are held at their last value after the end."""
    phases = list(phases)
    dts = [p.dt for p in phases]
    if max(dts) - min(dts) > 1e-3 * dts[0] + 1.0:
        raise ValueError('all the phases of a batch must have the same time step')
    B, L = len(phases), max(len(p.time) for p in phases)
    arr = {k: np.zeros((B, L)) for k in ('flow', 'T_in', 'pv_in', 'T_out', 'pv_out', 'soc',
                                         'mask', 'mask_T', 'mask_pv', 'mask_soc')}
    has_w0 = all(p.water0 is not None for p in phases)
    w0 = np.zeros((B, n_slices))
    for i, p in enumerate(phases):
        n = len(p.time)
        for k in ('flow', 'T_in', 'pv_in'):
            arr[k][i, :n] = getattr(p, k); arr[k][i, n:] = getattr(p, k)[-1]
        arr['mask'][i, :n] = 1
        for k, m in (('T_out', 'mask_T'), ('pv_out', 'mask_pv'), ('soc', 'mask_soc')):
            v = getattr(p, k)
            if v is not None:
                ok = np.isfinite(v)
                arr[m][i, :n] = ok
                arr[k][i, :n] = np.nan_to_num(v)
                if k != 'pv_out':
                    arr[k][i, n:] = arr[k][i, n - 1]
        if has_w0:
            w0[i] = water_profile(p.water0, n_slices)
    f64 = torch.float64
    t = {k: torch.tensor(v, dtype=f64) for k, v in arr.items()}
    return Batch(**t, water0=torch.tensor(w0, dtype=f64) if has_w0 else None,
                 Ts0=torch.tensor([p.initial_temperature() for p in phases], dtype=f64),
                 scale=torch.tensor([float(p.scale) for p in phases], dtype=f64), dt=float(np.mean(dts)),
                 names=[p.name for p in phases])


def phases_from_dataframe(df, phase_col, columns, celsius=True, group_col=None, kind_col=None,
                          kind_map=None, water0_cols=None, water0_total_col=None, scale=None, every=1, step=None):
    """Build phases from a long table (one row per time step, one column identifying the phase).

    ``columns`` maps Phase fields to column names, for example
    ``dict(time='t_s', flow='Q_m3h', T_in='Tin_C', pv_in='pv_in_Pa', T_out='T_out_C', pv_out='pv_out_Pa', soc='SoC')``.
    Temperatures are converted from degC when ``celsius`` is True. ``water0_cols`` lists the columns
    whose first row gives the initial water profile (kg); ``water0_total_col`` rescales it to a total.
    ``scale`` maps a phase name to its bed fraction. ``every`` keeps one row in ``every``; ``step`` is the
    nominal time step (s) of the table before subsampling, when the logged time stamps jitter.
    """
    out = []
    for name, g in df.groupby(phase_col, sort=False):
        g = g.sort_values(columns['time'])
        kw = {}
        for field_, col in columns.items():
            if col is None:
                continue
            v = g[col].to_numpy(dtype=float)
            if field_ in ('T_in', 'T_out') and celsius:
                v = v + T0K
            kw[field_] = v
        if water0_cols:
            w0 = g[list(water0_cols)].iloc[0].to_numpy(dtype=float)
            if water0_total_col:
                w0 = w0 / w0.sum() * float(g[water0_total_col].iloc[0])
            kw['water0'] = w0
        kind = None
        if kind_col is not None:
            kind = g[kind_col].iloc[0]
            kind = (kind_map or {}).get(kind, kind)
        p = Phase(name=str(name), group=g[group_col].iloc[0] if group_col else None, kind=kind,
                  scale=(scale or {}).get(name, 1.0), step=step, **kw)
        out.append(p.subsample(every) if every > 1 else p)
    return out
