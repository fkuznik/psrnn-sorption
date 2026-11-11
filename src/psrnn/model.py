"""Physics-structured recurrent model of an open packed-bed sorption heat store.

The bed is split into K slices along the air path. The state of slice k is its water
content m_k (kg) and its solid temperature Ts_k (K). At each step the air crosses the
slices in order; the outlet of slice k is the inlet of slice k+1. Water and energy are
balanced exactly in each slice (semi-implicit coupled step); air-solid exchanges take
the effectiveness form 1 - exp(-NTU), so that the air never overshoots equilibrium.
A small network, shared by all slices and initialised at zero, modulates the two NTU.
The equilibrium vapour pressure comes from an isotherm with trainable parameters.
The outlet duct adds a first-order lag and a temperature-dependent heat loss.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BedConfig
from .isotherms import Isotherm
from .physics import CP_AIR, CP_WATER, EPS, psat


class SorptionBedRNN(nn.Module):
    """Recurrent bed model. ``forward`` maps a :class:`psrnn.data.Batch` to outlet temperature,
    outlet vapour pressure, state of charge and (optionally) the water of each slice."""

    def __init__(self, bed: BedConfig, isotherm: Isotherm, hidden=16, substeps=1, use_network=True):
        super().__init__()
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)            # the semi-implicit step needs double precision
        try:
            self._build(bed, isotherm, hidden, substeps, use_network)
        finally:
            torch.set_default_dtype(old)
        self.double()

    def _build(self, bed, isotherm, hidden, substeps, use_network):
        self.bed, self.isotherm = bed, isotherm
        self.K, self.substeps, self.use_network = int(bed.n_slices), int(substeps), bool(use_network)
        self.l_ntum = nn.Parameter(torch.tensor(math.log(bed.ntu_mass)))
        self.l_ntuh = nn.Parameter(torch.tensor(math.log(bed.ntu_heat)))
        self.dH_z = nn.Parameter(torch.tensor(0.0))
        self.l_U = nn.Parameter(torch.tensor(math.log(bed.loss_bed)))
        self.beta_z = nn.Parameter(torch.tensor(-1.0))     # bed losses: U (1 + beta (Ts - Tamb) / 100)
        self.l_Ud = nn.Parameter(torch.tensor(math.log(bed.loss_duct)))
        self.gam_z = nn.Parameter(torch.tensor(-1.0))      # duct losses: Ud (1 + gamma |T - Tamb| / 100)
        self.l_Cp = nn.Parameter(torch.tensor(math.log(bed.wall_capacity)))
        self.l_tau = nn.Parameter(torch.tensor(math.log(bed.duct_time_constant)))
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)                   # start from the purely physical model

    # ------------------------------------------------------------------ physical parameters
    def heat_of_adsorption(self):
        lo, hi = self.bed.heat_of_adsorption
        return lo + (hi - lo) * torch.sigmoid(self.dH_z)

    def physical_parameters(self):
        """Current values of the trainable physical parameters (floats)."""
        d = dict(NTU_m=torch.exp(self.l_ntum), NTU_h=torch.exp(self.l_ntuh),
                 dH_MJ_kg=self.heat_of_adsorption() / 1e6, U_W_K=torch.exp(self.l_U),
                 beta=F.softplus(self.beta_z), Ud_W_K=torch.exp(self.l_Ud), gamma_d=F.softplus(self.gam_z),
                 Cwall_kJ_K=torch.exp(self.l_Cp) / 1e3, tau_s=torch.exp(self.l_tau))
        d = {k: float(v.detach()) for k, v in d.items()}
        d.update(self.isotherm.summary())
        return d

    def prior(self):
        return self.isotherm.prior()

    def w_eq(self, q, T):
        """Equilibrium humidity ratio over a slice of loading q at temperature T, capped below saturation."""
        p_atm = self.bed.p_atm
        pv = torch.minimum(self.isotherm.pressure(q, T), 0.98 * psat(T)).clamp(max=0.6 * p_atm)
        return EPS * pv / (p_atm - pv)

    def water_bounds(self, charge, discharge):
        """Water of the whole nominal bed (kg) at equilibrium with the charge air and the discharge air,
        each given as (T in K, vapour pressure in Pa), on the current (learnt) isotherm."""
        with torch.no_grad():
            mz = self.bed.adsorbent_mass
            f = lambda x: torch.tensor(float(x), dtype=torch.float64)
            lo = self.isotherm.loading(f(charge[1]), f(charge[0]))
            hi = self.isotherm.loading(f(discharge[1]), f(discharge[0]))
            return float(lo) * mz, float(hi) * mz

    def soc_from_water(self, water, scale, bounds=None):
        lo, hi = bounds if bounds is not None else (self.bed.water_min, self.bed.water_max)
        return 1 - (water - lo * scale) / ((hi - lo) * scale)

    def water_from_soc(self, soc, scale):
        """Uniform initial water profile (B, K) from the state of charge (B,)."""
        b = self.bed
        x = 1 - torch.as_tensor(soc)
        return ((b.water_min + x * (b.water_max - b.water_min)) * scale / self.K)[:, None].expand(-1, self.K)

    # ------------------------------------------------------------------ simulation
    def forward(self, batch, water0=None, traces=False):
        """Simulate the batch from the initial water ``water0`` (B, K); default: ``batch.water0``.

        Returns a dict with ``T_out`` (K), ``pv_out`` (Pa), ``soc`` (-), each (B, L), and with
        ``traces=True`` also ``water`` (B, L, K, kg) and ``Ts`` (B, L, K, K).
        """
        b = self.bed
        water0 = batch.water0 if water0 is None else water0
        if water0 is None:
            raise ValueError('no initial water content: give water0, or SoC labels, or use the observer')
        B, L = batch.flow.shape
        Kz, fac = self.K, batch.scale
        mz = b.adsorbent_mass * fac / Kz
        m = [water0[:, k].clone() for k in range(Kz)]
        Ts = [batch.Ts0.clone() for _ in range(Kz)]
        Td = batch.Ts0.clone()
        Tamb, p_atm = b.ambient_temperature, b.p_atm
        dH = self.heat_of_adsorption()
        Uk = torch.exp(self.l_U) * fac / Kz
        beta = F.softplus(self.beta_z)
        Ud = torch.exp(self.l_Ud)
        gam = F.softplus(self.gam_z)
        Cp = torch.exp(self.l_Cp) * fac / Kz
        tau = torch.exp(self.l_tau)
        ntum0, ntuh0 = torch.exp(self.l_ntum) / Kz, torch.exp(self.l_ntuh) / Kz
        qmax = self.isotherm.q_max() * 0.999
        dt = batch.dt / self.substeps
        outT, outpv, outsoc, outm, outTs = [], [], [], [], []
        for t in range(L):
            Q, Tin, pvin = batch.flow[:, t], batch.T_in[:, t], batch.pv_in[:, t]
            ma = b.rho_air * Q.clamp(min=b.min_flow) / 3600.0         # kg/s of dry air
            u = Q / 3600.0 / b.cross_section                           # m/s
            for _ in range(self.substeps):
                T_k, w_k = Tin, EPS * pvin / (p_atm - pvin)
                for k in range(Kz):
                    q = m[k] / mz
                    if self.use_network:
                        x = torch.stack([u / b.velocity_ref - 1, q / b.loading_ref - 1,
                                         (Ts[k] - b.temperature_ref) / b.temperature_scale,
                                         (T_k - b.temperature_ref) / b.temperature_scale], -1)
                        g = self.mlp(x).clamp(-3, 3)
                        ntum, ntuh = ntum0 * torch.exp(g[:, 0]), ntuh0 * torch.exp(g[:, 1])
                    else:
                        ntum, ntuh = ntum0, ntuh0
                    # semi-implicit coupled water-energy step: w_eq(m + dm, Ts + dTs) linearised
                    w0 = self.w_eq(q, Ts[k])
                    a_m = (self.w_eq(q + 1e-3, Ts[k]) - w0) / (1e-3 * mz)       # d w_eq / d m (1/kg)
                    b_T = (self.w_eq(q, Ts[k] + 0.5) - w0) / 0.5                # d w_eq / d Ts (1/K)
                    G = ma * dt * (1 - torch.exp(-ntum))
                    E = ma * CP_AIR * dt * (1 - torch.exp(-ntuh))
                    C = mz * b.cp_solid + m[k] * CP_WATER + Cp
                    A0 = (E * (T_k - Ts[k]) - Uk * (1 + beta * (Ts[k] - Tamb) / 100) * dt * (Ts[k] - Tamb)) / (C + E)
                    A1 = dH / (C + E)
                    dm = G * (w_k - w0 - b_T * A0) / (1 + G * a_m + G * b_T * A1)
                    dm = torch.minimum(dm, 0.999 * ma * dt * w_k)               # no more water than the air brings
                    m_new = torch.minimum(torch.maximum(m[k] + dm, 1e-4 * mz), qmax * mz)
                    dm = m_new - m[k]
                    Ts[k] = Ts[k] + A0 + A1 * dm
                    m[k] = m_new
                    T_k = T_k - (1 - torch.exp(-ntuh)) * (T_k - Ts[k])
                    w_k = (w_k - dm / (ma * dt)).clamp(min=1e-7)
                Udt = Ud * (1 + gam * (T_k - Tamb).abs() / 100)
                Tdin = Tamb + (T_k - Tamb) * torch.exp(-Udt / (ma * CP_AIR))
                Td = Td + (Tdin - Td) * (1 - torch.exp(-dt / tau))
            mt = torch.stack(m, 1).sum(1)
            outT.append(Td)
            outpv.append(p_atm * w_k / (EPS + w_k))
            outsoc.append(self.soc_from_water(mt, fac))
            if traces:
                outm.append(torch.stack(m, 1)); outTs.append(torch.stack(Ts, 1))
        r = dict(T_out=torch.stack(outT, 1), pv_out=torch.stack(outpv, 1), soc=torch.stack(outsoc, 1))
        if traces:
            r.update(water=torch.stack(outm, 1), Ts=torch.stack(outTs, 1))
        return r
