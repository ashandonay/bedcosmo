"""
Cosmology utilities and mixin class for bedcosmo experiments.

This module provides shared cosmology functions and a CosmologyMixin class that
can be inherited by experiment classes to compute standard cosmological distances.
"""

import math
import torch
from astropy import constants

from bedcosmo.util import profile_method

# Physical constants for cosmology calculations
T_CMB = 2.7255  # CMB temperature in Kelvin
KB_eV_per_K = 8.617333262e-5  # Boltzmann constant in eV/K
Tnu0_eV = (4.0 / 11.0) ** (1.0 / 3.0) * T_CMB * KB_eV_per_K  # Neutrino temperature today in eV


def _interp1(xg, yg, x):
    """
    Linear interpolation.

    Args:
        xg: Grid x values (1D tensor or array)
        yg: Grid y values (tensor)
        x: Query x values (tensor)

    Returns:
        Interpolated y values at query points
    """
    xg = torch.as_tensor(xg, dtype=torch.float64, device=yg.device)
    yg = torch.as_tensor(yg, dtype=torch.float64, device=yg.device)
    x = torch.as_tensor(x, dtype=torch.float64, device=yg.device)

    x = x.clamp(xg[0], xg[-1])
    idx = torch.searchsorted(xg, x, right=True) - 1
    idx = idx.clamp(0, xg.numel() - 2)
    idx1 = idx + 1

    x0, x1 = xg[idx], xg[idx1]
    y0, y1 = yg[idx], yg[idx1]
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _infer_plate_shape(dev, dtype, *args):
    """
    Infer broadcast plate shape from input tensors.

    Look at inputs (tensors or scalars), treat any final-dim==1 as 'placeholder',
    and broadcast the *leading* dims to a common plate shape.

    Args:
        dev: Device to use
        dtype: Data type to use
        *args: Input tensors or scalars

    Returns:
        Tuple representing the common plate shape
    """
    lead_shapes = []
    for a in args:
        ta = torch.as_tensor(a, device=dev, dtype=dtype) if a is not None else None
        if ta is None or ta.ndim == 0:
            lead_shapes.append(())  # scalar: no plate dims
        else:
            if ta.shape[-1] == 1:
                lead_shapes.append(ta.shape[:-1])
            else:
                lead_shapes.append(ta.shape)
    plate = ()
    for s in lead_shapes:
        plate = torch.broadcast_shapes(plate, s)
    return plate


def _cumsimpson(x, y, dim=-1):
    """
    Cumulative composite Simpson's rule along `dim`.

    Requires an even number of intervals; if not, the last interval
    falls back to trapezoid. Returns same shape as y.

    Args:
        x: Grid points (1D tensor)
        y: Function values (tensor)
        dim: Dimension along which to integrate

    Returns:
        Cumulative integral values with same shape as y
    """
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    N = y.shape[dim]
    if N < 2:
        return torch.zeros_like(y)

    # Δx segments
    dx = x.diff()
    # Move dim to last for easier slicing
    yT = y.transpose(dim, -1)  # (..., N)

    out = torch.zeros_like(yT)
    out[..., 0] = 0.0

    # For simplicity and speed, do blocks of 2 intervals where possible
    two = (N - 1) // 2 * 2  # largest even number ≤ N-1

    # Simpson over [2k, 2k+2]
    i0 = torch.arange(0, two, 2, device=y.device)
    i1 = i0 + 1
    i2 = i0 + 2
    dx0 = dx[i0]
    dx1 = dx[i1]
    h = dx0 + dx1
    contrib = h / 6.0 * (yT[..., i0] + 4.0 * yT[..., i1] + yT[..., i2])  # (..., #pairs)

    # cumulative sum in pairs
    acc_pairs = contrib.cumsum(dim=-1)
    # scatter accumulated values into out at even indices
    out[..., 2 : two + 1 : 2] = acc_pairs

    # fill odd indices (use Simpson partials)
    trap01 = 0.5 * dx0 * (yT[..., i0] + yT[..., i1])
    out[..., 1:two:2] = (acc_pairs - contrib) + trap01

    # last interval if N-1 is odd: trapezoid from N-2→N-1
    if (N - 1) % 2 == 1:
        last = N - 1
        out[..., last] = out[..., last - 1] + 0.5 * dx[last - 1] * (yT[..., last - 1] + yT[..., last])

    # transpose back
    return out.transpose(-1, dim)


class _NeutrinoTableCache:
    """
    Cache for neutrino density evolution table.

    Builds a table of f_nu(a) = rho_nu(a)/rho_nu(1) on an a-grid for
    efficient computation of massive neutrino contributions to E(z).
    """

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.key = None
        self.a = None
        self.lna = None
        self.fnu = None  # (n_massive, n_ag)
        self._built = False

    def _build_table(self, mnu_eV, n_massive, n_ag=1200, qmax=40.0, nq=2000, a_min=0.28):
        """
        Build f_nu(a) on an a-grid in (a_min, 1].

        f_nu(a) = rho_nu(a)/rho_nu(1) = [ I(a) / I(1) ] * a^{-4}
        with I(a) = ∫ dq q^2 sqrt(q^2 + (a*y0)^2) / (1+e^q),
        y0 = (m_per / Tnu0), m_per = mnu_eV / n_massive.
        """
        # constants
        T_cmb = torch.tensor(2.7255, device=self.device, dtype=self.dtype)  # K
        kB_eV = torch.tensor(8.61733262e-5, device=self.device, dtype=self.dtype)  # eV/K
        Tnu0 = (torch.tensor(4.0 / 11.0, device=self.device, dtype=self.dtype)) ** (1.0 / 3.0) * T_cmb * kB_eV

        m_per = torch.tensor(float(mnu_eV) / int(n_massive), device=self.device, dtype=self.dtype)
        y0 = m_per / Tnu0

        # ln a grid
        self.lna = torch.linspace(
            torch.log(torch.as_tensor(a_min, device=self.device, dtype=self.dtype)),
            torch.tensor(0.0, device=self.device, dtype=self.dtype),
            int(n_ag),
            device=self.device,
            dtype=self.dtype,
        )
        self.a = torch.exp(self.lna)  # (G,)

        # momentum grid
        q = torch.linspace(0.0, float(qmax), int(nq), device=self.device, dtype=self.dtype)  # (nq,)
        fq = 1.0 / (torch.exp(q) + 1.0)
        q2 = q * q

        # I(a) integral: E(q,a) = sqrt(q^2 + (a*y0)^2)
        Ey = torch.sqrt(q2[:, None] + (self.a[None, :] * y0) ** 2)  # (nq, G)
        Ia = torch.trapezoid(q2[:, None] * Ey * fq[:, None], q, dim=0)  # (G,)

        # I(1) at a=1:
        E1 = torch.sqrt(q2 + y0 * y0)
        I1 = torch.trapezoid(q2 * E1 * fq, q, dim=0)  # scalar

        # f_nu(a) = (Ia/I1) * a^{-4}
        fnu_1d = (Ia / I1) * (self.a ** (-4))
        self.fnu = fnu_1d.unsqueeze(0).expand(int(n_massive), -1)  # (n_massive, G)

        self._built = True

    def get_table(self, mnu_eV, n_massive, n_ag=1200, qmax=40.0, nq=2000, a_min=0.28):
        """Get the neutrino table, building it if not already built."""
        if not self._built:
            self._build_table(mnu_eV, n_massive, n_ag, qmax, nq, a_min)
        return self.a, self.fnu, self.lna


class CosmologyMixin:
    """
    Mixin class providing standard cosmology distance calculations.

    This mixin can be inherited by experiment classes to provide
    D_H/r_d, D_M/r_d, and D_V/r_d calculations with proper handling
    of radiation, massive neutrinos, dark energy (CPL), and curvature.

    Required attributes on the inheriting class:
        - device: torch device
        - c: speed of light in km/s (typically from astropy.constants)
        - hrdrag_multiplier: multiplier for hrdrag parameter (default 100.0)
    """

    # Physical constants
    T_CMB = 2.7255
    KB_eV_per_K = 8.617333262e-5
    Tnu0_eV = (4.0 / 11.0) ** (1.0 / 3.0) * T_CMB * KB_eV_per_K

    def _E_of_z(self, z, Om, Ok, w0, wa, Or, Onu0, Ode0, n_massive, cache):
        """
        Compute E(z) = H(z)/H0 including radiation and massive neutrinos.

        Args:
            z: (plate, Nz) redshift array
            Om, Ok, w0, wa: (plate, 1) cosmological parameters
            Or: (plate, 1) radiation density today
            Onu0: (plate, 1) massive neutrino density today
            Ode0: (plate, 1) dark energy density today
            n_massive: Number of massive neutrino species
            cache: _NeutrinoTableCache object (or None if n_massive==0)

        Returns:
            E(z) with shape (plate, Nz)
        """
        zp1 = 1.0 + z
        # CPL dark energy
        fde = zp1 ** (3.0 * (1.0 + w0 + wa)) * torch.exp(-3.0 * wa * z / zp1)

        # massive ν(a) term
        if n_massive > 0 and cache is not None:
            ln_a = -torch.log1p(z)
            fnu_at_a = _interp1(cache.lna, cache.fnu[0], ln_a)
            Onu_a = Onu0 * fnu_at_a
        else:
            Onu_a = torch.zeros_like(z, dtype=z.dtype, device=z.device)

        # split cb from total matter today
        Ocb = (Om - Onu0).clamp_min(0.0)

        E2 = Or * zp1**4 + Ocb * zp1**3 + Ok * zp1**2 + Ode0 * fde + Onu_a
        return torch.sqrt(torch.clamp_min(E2, torch.finfo(z.dtype).tiny * 1e6))

    @profile_method
    def D_H_func(
        self,
        z_eff,
        Om,
        Ok=None,
        w0=None,
        wa=None,
        hrdrag=None,
        h=0.6736,
        Neff=3.044,
        mnu=0.06,
        n_massive=1,
        T_cmb=2.7255,
        include_radiation=True,
    ):
        """
        Compute D_H/r_d (Hubble distance over sound horizon).

        High-accuracy computation including radiation and massive neutrino contributions.

        Args:
            z_eff: Redshift(s) to evaluate at
            Om: Matter density parameter
            Ok: Curvature density parameter (default 0)
            w0: Dark energy equation of state w0 (default -1)
            wa: Dark energy equation of state wa (default 0)
            hrdrag: h * r_drag in km/s (100 * r_d if using hrdrag_multiplier=100)
            h: Hubble parameter H0/100 km/s/Mpc (default 0.6736)
            Neff: Effective number of neutrino species (default 3.044)
            mnu: Sum of neutrino masses in eV (default 0.06)
            n_massive: Number of massive neutrino species (default 1)
            T_cmb: CMB temperature in K (default 2.7255)
            include_radiation: Whether to include radiation contribution (default True)

        Returns:
            D_H/r_d with shape (plate, Nz)
        """
        DTYPE = torch.float64
        dev = self.device

        plate = _infer_plate_shape(dev, DTYPE, Om, Ok, w0, wa, hrdrag)

        def to_plate1(x, default=None):
            if x is None:
                x = default
            t = torch.as_tensor(x, device=dev, dtype=DTYPE)
            if t.ndim == len(plate) + 1 and t.shape[-1] == 1 and list(t.shape[:-1]) == list(plate):
                return t
            return t.view(*([1] * len(plate)), 1).expand(plate + (1,))

        Om = to_plate1(Om)
        Ok = to_plate1(0.0 if Ok is None else Ok)
        w0 = to_plate1(-1.0 if w0 is None else w0)
        wa = to_plate1(0.0 if wa is None else wa)
        hrdrag = to_plate1(hrdrag)

        h_t = None if h is None else torch.as_tensor(h, device=dev, dtype=DTYPE)
        Tcmb = torch.as_tensor(T_cmb, device=dev, dtype=DTYPE)

        # z -> (plate, Nz)
        z_eff = torch.as_tensor(z_eff, device=dev, dtype=DTYPE)
        if z_eff.ndim == 0:
            z_eff = z_eff[None]

        if z_eff.ndim == 1:
            Nz = z_eff.shape[0]
            z = z_eff.reshape(*([1] * len(plate)), Nz).expand(plate + (Nz,))
        else:
            assert z_eff.shape[-1] > 0, "z must have a last dimension Nz > 0"
            if tuple(z_eff.shape[:-1]) != tuple(plate):
                z = torch.broadcast_to(z_eff, plate + (z_eff.shape[-1],))
            else:
                z = z_eff

        # radiation today
        if include_radiation:
            assert h_t is not None, "Pass h=H0/100 when include_radiation=True"
            Ogam0 = 2.469e-5 * (Tcmb / 2.7255) ** 4 / (h_t * h_t)
            N_massless = max(0.0, float(Neff) - int(n_massive))
            Onur0 = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * N_massless * Ogam0
            Or = torch.as_tensor(Ogam0 + Onur0, device=dev, dtype=DTYPE).view(*([1] * len(plate)), 1).expand(plate + (1,))
        else:
            Or = torch.zeros(plate + (1,), device=dev, dtype=DTYPE)

        # massive ν today
        if n_massive > 0:
            assert h_t is not None, "Need h to compute Ων0"
            Onu0 = torch.as_tensor((float(mnu) / 93.14) / (float(h_t) * float(h_t)), device=dev, dtype=DTYPE)
            Onu0 = Onu0.view(*([1] * len(plate)), 1).expand(plate + (1,))
        else:
            Onu0 = torch.zeros(plate + (1,), device=dev, dtype=DTYPE)

        # Dark-energy density today
        Ode0 = 1.0 - Om - Ok - Or

        # neutrino table cache (if needed)
        if n_massive > 0:
            if not hasattr(self, "_nu_cache"):
                self._nu_cache = _NeutrinoTableCache(dev, DTYPE)
            self._nu_cache.get_table(mnu, n_massive)
        else:
            self._nu_cache = None

        # common E(z)
        E = self._E_of_z(z, Om, Ok, w0, wa, Or, Onu0, Ode0, n_massive, self._nu_cache)

        # Get hrdrag_multiplier (default to 100.0 if not set)
        hrdrag_multiplier = getattr(self, "hrdrag_multiplier", 100.0)
        prefac = torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (hrdrag_multiplier * hrdrag)
        return prefac / E

    @profile_method
    def D_M_func(
        self,
        z_eff,
        Om,
        Ok=None,
        w0=None,
        wa=None,
        hrdrag=None,
        h=0.6736,
        Neff=3.044,
        mnu=0.06,
        n_massive=1,
        T_cmb=2.7255,
        include_radiation=True,
        n_int=1025,
    ):
        """
        Compute D_M/r_d (transverse comoving distance over sound horizon).

        Uses Simpson's rule integration in ln(a) coordinates with
        sophisticated curvature handling.

        Args:
            z_eff: Redshift(s) to evaluate at
            Om: Matter density parameter
            Ok: Curvature density parameter (default 0)
            w0: Dark energy equation of state w0 (default -1)
            wa: Dark energy equation of state wa (default 0)
            hrdrag: h * r_drag in km/s
            h: Hubble parameter H0/100 km/s/Mpc (default 0.6736)
            Neff: Effective number of neutrino species (default 3.044)
            mnu: Sum of neutrino masses in eV (default 0.06)
            n_massive: Number of massive neutrino species (default 1)
            T_cmb: CMB temperature in K (default 2.7255)
            include_radiation: Whether to include radiation contribution (default True)
            n_int: Number of integration points (default 1025)

        Returns:
            D_M/r_d with shape (plate, Nz)
        """
        DTYPE = torch.float64
        dev = self.device

        plate = _infer_plate_shape(dev, DTYPE, Om, Ok, w0, wa, hrdrag)

        def to_plate1(x, default=None):
            if x is None:
                x = default
            t = torch.as_tensor(x, device=dev, dtype=DTYPE)
            if t.ndim == len(plate) + 1 and t.shape[-1] == 1 and list(t.shape[:-1]) == list(plate):
                return t
            return t.view(*([1] * len(plate)), 1).expand(plate + (1,))

        Om = to_plate1(Om)
        Ok = to_plate1(0.0 if Ok is None else Ok)
        w0 = to_plate1(-1.0 if w0 is None else w0)
        wa = to_plate1(0.0 if wa is None else wa)
        hrdrag = to_plate1(hrdrag)

        # targets in z / ln a
        z_eff = torch.as_tensor(z_eff, device=dev, dtype=DTYPE)
        if z_eff.ndim == 0:
            z_eff = z_eff[None]

        if z_eff.ndim == 1:
            Nz = z_eff.shape[0]
            z_eval = z_eff.reshape(*([1] * len(plate)), Nz).expand(plate + (Nz,))
        else:
            assert z_eff.shape[-1] > 0, "z must have a last dimension Nz > 0"
            if tuple(z_eff.shape[:-1]) != tuple(plate):
                z_eval = torch.broadcast_to(z_eff, plate + (z_eff.shape[-1],))
            else:
                z_eval = z_eff

        ln_a_ev = -torch.log1p(z_eval)

        # base Simpson grid in ln a (odd length) & merge exact eval nodes
        zmax = float(z_eval.max())
        a_min = 1.0 / (1.0 + zmax)
        ln_a_base = torch.linspace(math.log(a_min), 0.0, int(n_int) | 1, device=dev, dtype=DTYPE)
        ln_a_all = torch.unique(torch.cat([ln_a_base, ln_a_ev.reshape(-1)])).sort().values
        if ln_a_all[-1].abs() > 0:
            ln_a_all = torch.unique(torch.cat([ln_a_all, torch.zeros(1, device=dev, dtype=DTYPE)])).sort().values
        if (ln_a_all.numel() % 2) == 0:
            mid = 0.5 * (ln_a_all[-2] + ln_a_all[-1])
            ln_a_all = torch.unique(torch.cat([ln_a_all, mid[None]])).sort().values

        a_all = torch.exp(ln_a_all)
        z_all = 1.0 / a_all - 1.0
        zB = z_all.reshape(*([1] * len(plate)), -1).expand(plate + (ln_a_all.numel(),))

        DH_over_rd_all = self.D_H_func(
            zB,
            Om,
            Ok,
            w0,
            wa,
            hrdrag,
            h=h,
            Neff=Neff,
            mnu=mnu,
            n_massive=n_massive,
            T_cmb=T_cmb,
            include_radiation=include_radiation,
        )

        hrdrag_multiplier = getattr(self, "hrdrag_multiplier", 100.0)
        pref = torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (hrdrag_multiplier * hrdrag)
        pref = pref.expand_as(DH_over_rd_all)
        E_all = (pref / DH_over_rd_all).clamp_min(torch.finfo(DTYPE).tiny * 1e6)

        # integrate: ∫ d ln a / (a E(a)) from a(z)→1
        aB = a_all.reshape(*([1] * len(plate)), -1).expand_as(E_all)
        integ = 1.0 / (aB * E_all)
        cum = _cumsimpson(ln_a_all, integ, dim=-1)
        cum_1 = cum[..., -1:]

        # exact pick at ln a(z) (we merged nodes above)
        idx = torch.searchsorted(ln_a_all, ln_a_ev)
        DC_over_DH0 = cum_1 - torch.gather(cum, -1, idx)

        # curvature mapping (with flat-limit series)
        absOk = torch.abs(Ok)
        sqrtOk = torch.sqrt(absOk)
        x = sqrtOk * DC_over_DH0

        Skx = torch.where(Ok > 0, torch.sinh(x), torch.where(Ok < 0, torch.sin(x), DC_over_DH0))
        den = torch.where(Ok == 0, torch.ones_like(Ok), sqrtOk)

        tiny = (absOk < 1e-8).expand_as(DC_over_DH0)
        series = DC_over_DH0 * (1.0 + (Ok * DC_over_DH0**2) / 6.0 + (Ok**2 * DC_over_DH0**4) / 120.0)
        geom = torch.where(tiny, series, torch.where(Ok != 0, Skx / den, Skx))

        # (c/H0)/r_d
        prefac = torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (hrdrag_multiplier * hrdrag)
        return prefac * geom

    @profile_method
    def D_V_func(
        self,
        z_eff,
        Om,
        Ok=None,
        w0=None,
        wa=None,
        hrdrag=None,
        h=0.6736,
        Neff=3.044,
        mnu=0.06,
        n_massive=1,
        T_cmb=2.7255,
        include_radiation=True,
        n_int=1025,
    ):
        """
        Compute D_V/r_d (volume-averaged distance over sound horizon).

        D_V/r_d = [ z * (D_M/r_d)^2 * (D_H/r_d) ]^{1/3}

        Args:
            z_eff: Redshift(s) to evaluate at
            Om: Matter density parameter
            Ok: Curvature density parameter (default 0)
            w0: Dark energy equation of state w0 (default -1)
            wa: Dark energy equation of state wa (default 0)
            hrdrag: h * r_drag in km/s
            h: Hubble parameter H0/100 km/s/Mpc (default 0.6736)
            Neff: Effective number of neutrino species (default 3.044)
            mnu: Sum of neutrino masses in eV (default 0.06)
            n_massive: Number of massive neutrino species (default 1)
            T_cmb: CMB temperature in K (default 2.7255)
            include_radiation: Whether to include radiation contribution (default True)
            n_int: Number of integration points (default 1025)

        Returns:
            D_V/r_d with shape (plate, Nz)
        """
        DM = self.D_M_func(
            z_eff,
            Om,
            Ok,
            w0,
            wa,
            hrdrag,
            h=h,
            Neff=Neff,
            mnu=mnu,
            n_massive=n_massive,
            T_cmb=T_cmb,
            include_radiation=include_radiation,
            n_int=n_int,
        )
        DH = self.D_H_func(
            z_eff,
            Om,
            Ok,
            w0,
            wa,
            hrdrag,
            h=h,
            Neff=Neff,
            mnu=mnu,
            n_massive=n_massive,
            T_cmb=T_cmb,
            include_radiation=include_radiation,
        )

        DTYPE, dev = DM.dtype, DM.device
        z_t = torch.as_tensor(z_eff, dtype=DTYPE, device=dev)
        if z_t.ndim == 0:
            z_t = z_t[None]
        if z_t.ndim == 1:
            zB = z_t.reshape(*([1] * (DM.ndim - 1)), -1).expand_as(DM)
        else:
            zB = torch.broadcast_to(z_t, DM.shape)

        return (zB * (DM**2) * DH).pow(1.0 / 3.0)
