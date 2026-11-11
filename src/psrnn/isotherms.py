"""Sorption isotherms with trainable parameters.

Each isotherm gives the equilibrium loading q (kg water / kg dry adsorbent) at vapour
pressure p (Pa) and temperature T (K), and its inverse, the equilibrium vapour pressure
of a bed slice holding the loading q. The bed model only needs the inverse.

Parameters are stored in transformed form (logarithms) so that they stay physical
during training. ``prior()`` returns a quadratic penalty that pulls them back towards
their initial (calibrated or literature) values; its widths are set by ``prior_widths``.
"""
import math
import torch
import torch.nn as nn

from .physics import R_GAS, psat

T_REF = 293.15


def _param(x):
    return nn.Parameter(torch.tensor(float(x), dtype=torch.float64))


class Isotherm(nn.Module):
    """Base class. Subclasses implement ``loading``, ``pressure``, ``q_max``, ``prior`` and ``summary``."""

    def loading(self, p, T):
        raise NotImplementedError

    def pressure(self, q, T):
        raise NotImplementedError

    def q_max(self):
        raise NotImplementedError

    def prior(self):
        return torch.zeros(())

    def summary(self):
        return {}


class Toth(Isotherm):
    """Toth isotherm with temperature-dependent heterogeneity:

    q = qm b p / (1 + (b p)^t)^(1/t),  b = b20 exp(Q/R (1/T - 1/T20)),  t = t20 + c (1 - T20/T).

    ``lb20`` is log10(b20) with b20 in 1/Pa, ``Q`` in J/mol.
    """

    def __init__(self, qm, lb20, Q, t20, c, prior_widths=None):
        super().__init__()
        self.ref = dict(qm=qm, lb20=lb20, Q=Q, t20=t20, c=c)
        self.widths = dict(qm=0.05, lb20=0.2, Q=5000.0, t20=0.1, c=0.02)   # qm and t20: relative; others: absolute
        self.widths.update(prior_widths or {})
        self.l_qm = _param(math.log(qm))
        self.lb20 = _param(lb20)
        self.Q4 = _param(Q / 1e4)
        self.l_t20 = _param(math.log(t20))
        self.c = _param(c)

    def params(self):
        return torch.exp(self.l_qm), 10 ** self.lb20, self.Q4 * 1e4, torch.exp(self.l_t20), self.c

    def _b_t(self, T):
        qm, b20, Qa, t20, c = self.params()
        b = b20 * torch.exp(Qa / R_GAS * (1 / T - 1 / T_REF))
        t = t20 + c * (1 - T_REF / T)
        return qm, b, t

    def loading(self, p, T):
        qm, b, t = self._b_t(T)
        bp = b * p
        return qm * bp / (1 + bp ** t) ** (1 / t)

    def pressure(self, q, T):
        qm, b, t = self._b_t(T)
        t = t.clamp(min=0.03)
        y = (q / qm).clamp(1e-6, 0.999)
        yt = y ** t
        return (yt / (1 - yt)) ** (1 / t) / b

    def q_max(self):
        return torch.exp(self.l_qm)

    def prior(self):
        r, w = self.ref, self.widths
        return (((torch.exp(self.l_qm) / r['qm'] - 1) / w['qm']) ** 2
                + ((self.lb20 - r['lb20']) / w['lb20']) ** 2
                + ((self.Q4 - r['Q'] / 1e4) / (w['Q'] / 1e4)) ** 2
                + ((torch.exp(self.l_t20) / r['t20'] - 1) / w['t20']) ** 2
                + ((self.c - r['c']) / w['c']) ** 2)

    def summary(self):
        qm, _, Qa, t20, c = (float(x.detach()) for x in self.params())
        return dict(qm=qm, lb20=float(self.lb20.detach()), Q_J_mol=Qa, t20=t20, c=c)


class Langmuir(Isotherm):
    """Langmuir isotherm q = qm b p / (1 + b p), b = b20 exp(Q/R (1/T - 1/T20)); ``lb20`` = log10(b20 in 1/Pa)."""

    def __init__(self, qm, lb20, Q, prior_widths=None):
        super().__init__()
        self.ref = dict(qm=qm, lb20=lb20, Q=Q)
        self.widths = dict(qm=0.05, lb20=0.2, Q=5000.0)
        self.widths.update(prior_widths or {})
        self.l_qm = _param(math.log(qm))
        self.lb20 = _param(lb20)
        self.Q4 = _param(Q / 1e4)

    def _b(self, T):
        return 10 ** self.lb20 * torch.exp(self.Q4 * 1e4 / R_GAS * (1 / T - 1 / T_REF))

    def loading(self, p, T):
        bp = self._b(T) * p
        return torch.exp(self.l_qm) * bp / (1 + bp)

    def pressure(self, q, T):
        y = (q / torch.exp(self.l_qm)).clamp(1e-6, 0.999)
        return y / (1 - y) / self._b(T)

    def q_max(self):
        return torch.exp(self.l_qm)

    def prior(self):
        r, w = self.ref, self.widths
        return (((torch.exp(self.l_qm) / r['qm'] - 1) / w['qm']) ** 2
                + ((self.lb20 - r['lb20']) / w['lb20']) ** 2
                + ((self.Q4 - r['Q'] / 1e4) / (w['Q'] / 1e4)) ** 2)

    def summary(self):
        return dict(qm=float(torch.exp(self.l_qm).detach()), lb20=float(self.lb20.detach()), Q_J_mol=float(self.Q4.detach()) * 1e4)


class DubininAstakhov(Isotherm):
    """Dubinin-Astakhov isotherm, usual for silica gels and activated carbons:

    q = q0 exp(-(A/E)^n),  A = R T ln(psat(T)/p)  (adsorption potential, J/mol).
    """

    def __init__(self, q0, E, n, prior_widths=None):
        super().__init__()
        self.ref = dict(q0=q0, E=E, n=n)
        self.widths = dict(q0=0.05, E=0.1, n=0.1)        # relative widths
        self.widths.update(prior_widths or {})
        self.l_q0 = _param(math.log(q0))
        self.l_E = _param(math.log(E))
        self.l_n = _param(math.log(n))

    def loading(self, p, T):
        A = R_GAS * T * torch.log(psat(T) / p)
        return torch.exp(self.l_q0) * torch.exp(-(A.clamp(min=0) / torch.exp(self.l_E)) ** torch.exp(self.l_n))

    def pressure(self, q, T):
        y = (q / torch.exp(self.l_q0)).clamp(1e-6, 0.999)
        A = torch.exp(self.l_E) * (-torch.log(y)) ** (1 / torch.exp(self.l_n))
        return psat(T) * torch.exp(-A / (R_GAS * T))

    def q_max(self):
        return torch.exp(self.l_q0)

    def prior(self):
        r, w = self.ref, self.widths
        return sum(((torch.exp(getattr(self, 'l_' + k)) / r[k] - 1) / w[k]) ** 2 for k in ('q0', 'E', 'n'))

    def summary(self):
        return {k: float(torch.exp(getattr(self, 'l_' + k)).detach()) for k in ('q0', 'E', 'n')}
