"""Prediction of phases, accuracy metrics, exchanged energy and conformal prediction intervals."""
import numpy as np
import pandas as pd
import torch

from .data import make_batch
from .observer import observe_water_fraction
from .physics import CP_AIR, T0K


def exchanged_energy(time, flow, T_in, T_out, kind, rho_air=1.2):
    """Heat exchanged with the air over a phase (kWh): released in discharge, absorbed in charge."""
    dT = (np.asarray(T_out) - np.asarray(T_in)) if kind == 'discharge' else (np.asarray(T_in) - np.asarray(T_out))
    return float(np.trapezoid(rho_air * np.asarray(flow) / 3600 * CP_AIR * dT, time) / 3.6e6)


def heat_power(flow, T_in, T_out, rho_air=1.2):
    """Heat power given to the air (W, positive when the air is heated)."""
    return rho_air * np.asarray(flow) / 3600 * CP_AIR * (np.asarray(T_out) - np.asarray(T_in))


def predict(model, phase, soc0=None, window_h=None, use_label=False, bounds=None):
    """Simulate one phase and return a table with measured and predicted outlet, SoC and water.

    The initial state comes, in this order, from ``soc0``, from the phase labels (``use_label=True``:
    ``phase.water0``), or from the observer over the first ``window_h`` hours (whole phase if None).
    ``bounds`` = (charge, discharge) reference states, each (T in K, pv in Pa), adds the SoC computed
    on the water bounds of the learnt isotherm (column ``soc_model_bounds``).
    """
    batch = make_batch([phase], model.K)
    if soc0 is not None:
        w0, how = model.water_from_soc(torch.tensor([float(soc0)], dtype=torch.float64), batch.scale), 'given'
    elif use_label:
        w0, how = batch.water0, 'label'
        soc0 = float(phase.soc[0]) if phase.soc is not None else np.nan
    else:
        x = observe_water_fraction(model, phase, window_h)
        w0, how, soc0 = model.water_from_soc(torch.tensor([1 - x], dtype=torch.float64), batch.scale), 'observer', 1 - x
    with torch.no_grad():
        r = model(batch, w0, traces=True)
    n = len(phase.time)
    water = r['water'][0, :n].sum(-1).numpy()
    df = pd.DataFrame(dict(phase=phase.name, time_s=phase.time, init=how, soc0=float(soc0),
                           T_out_pred_C=r['T_out'][0, :n].numpy() - T0K, pv_out_pred_Pa=r['pv_out'][0, :n].numpy(),
                           soc_pred=r['soc'][0, :n].numpy(), water_kg=water))
    if bounds is not None:
        lo, hi = model.water_bounds(*bounds)
        df['soc_model_bounds'] = 1 - (water - lo * phase.scale) / ((hi - lo) * phase.scale)
    if phase.T_out is not None:
        df['T_out_meas_C'] = phase.T_out - T0K
    if phase.pv_out is not None:
        df['pv_out_meas_Pa'] = phase.pv_out
    if phase.soc is not None:
        df['soc_label'] = phase.soc
    return df


def phase_metrics(phase, pred, rho_air=1.2):
    """Accuracy of one predicted phase (output of :func:`predict`)."""
    r = dict(phase=phase.name, group=phase.group, kind=phase.kind, init=pred['init'].iloc[0], soc0=pred['soc0'].iloc[0])
    T = pred['T_out_pred_C'].to_numpy() + T0K
    if phase.T_out is not None:
        ok = np.isfinite(phase.T_out)
        r['RMSE_T_K'] = float(np.sqrt(np.mean((T[ok] - phase.T_out[ok]) ** 2)))
        if phase.kind in ('charge', 'discharge'):
            e_m = exchanged_energy(phase.time[ok], phase.flow[ok], phase.T_in[ok], phase.T_out[ok], phase.kind, rho_air)
            e_p = exchanged_energy(phase.time[ok], phase.flow[ok], phase.T_in[ok], T[ok], phase.kind, rho_air)
            r.update(E_meas_kWh=e_m, E_pred_kWh=e_p, err_E_pct=(e_p - e_m) / e_m * 100 if e_m else np.nan)
    if phase.pv_out is not None:
        ok = np.isfinite(phase.pv_out)
        if ok.any():
            ref = float(np.median(phase.pv_in))
            r['RMSE_pv_pct'] = float(np.sqrt(np.mean((pred['pv_out_pred_Pa'].to_numpy()[ok] - phase.pv_out[ok]) ** 2)) / ref * 100)
    if phase.soc is not None:
        s = pred['soc_pred'].to_numpy(); ok = np.isfinite(phase.soc)
        r['RMSE_SoC'] = float(np.sqrt(np.mean((s[ok] - phase.soc[ok]) ** 2)))
        r['err_SoC0'] = float(s[0] - phase.soc[0])
        r['err_SoC_end'] = float(s[-1] - phase.soc[-1])
    return r


def evaluate(model, phases, **predict_kw):
    """Predict each phase and return (metrics table, predictions table)."""
    preds = [predict(model, p, **predict_kw) for p in phases]
    rows = [phase_metrics(p, pr, model.bed.rho_air) for p, pr in zip(phases, preds)]
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------------- conformal intervals
def conformal_quantile(scores, alpha=0.1):
    """Quantile (1 - alpha)(1 + 1/n) of the non-conformity scores (split conformal prediction)."""
    s = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.nan
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    return float(s[min(k, n - 1)])


def adaptive_scale(x, time, s0, half_window=1800.0, slope=0.1):
    """Local scale of the error: s0 + slope |x(t + h) - x(t - h)|; errors concentrate in the fronts."""
    xp = np.interp(time + half_window, time, x)
    xm = np.interp(time - half_window, time, x)
    return s0 + slope * np.abs(xp - xm)


def conformal_interval(cal_residuals, prediction, alpha=0.1, cal_scale=None, scale=None):
    """Prediction interval from calibration residuals (prediction - measurement) of phases unseen by the
    model. Constant width if no scale is given; otherwise width proportional to ``scale``.
    Returns (lower, upper, q)."""
    r = np.abs(np.asarray(cal_residuals, dtype=float))
    if cal_scale is not None:
        r = r / np.asarray(cal_scale, dtype=float)
    q = conformal_quantile(r, alpha)
    half = q if scale is None else q * np.asarray(scale, dtype=float)
    return np.asarray(prediction) - half, np.asarray(prediction) + half, q
