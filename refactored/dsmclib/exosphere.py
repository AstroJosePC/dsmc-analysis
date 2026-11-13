"""Collisionless (Liouville / Chamberlain) exosphere from a Maxwellian source sphere.

Reference solution for 1-D DSMC runs without collisions (or above the exobase), fed by the solver's
lower-boundary injection: a half-Maxwellian outgoing flux n0 * vbar / 4 at radius r0 and temperature T,
with every particle that comes back down to r0 absorbed (LBC_ALL_RETURN).

By Liouville's theorem f(r, v) = f_M(E) (Maxwellian of the source in total energy) on every phase-space
point whose trajectory connects to the source sphere going outward, and 0 elsewhere:

* outward (v_r > 0): periapsis below r0 (the orbit passes through r0);
* inward  (v_r < 0): additionally bound (E < 0), and the particle must have come back:
    - ``ubc="reflect"`` / ``"ballistic"``: every bound orbit returns (1-D reflection at r_top is the zero-delay
      limit of the ballistic return), except ballistic orbits with apoapsis beyond ``r_hill``;
    - ``ubc="free_escape"``: only orbits whose apoapsis lies below r_top return.

This is Chamberlain's ballistic + escaping population (no satellites), truncated for free escape.

    from dsmclib import exosphere
    prof = exosphere.profile(r, n0=2.3e13, T=270, mass=2.016*AMU, r0=R+100e3, r_top=R+2500e3, ubc="reflect")
    exosphere.escape_rate(n0=2.3e13, T=270, mass=..., r0=..., r_top=..., ubc="free_escape")
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .constants import K_BOLTZMANN as K_B

G = 6.674e-11  # same value as the solver (MOD_Constants grav)

MARS_MASS = 6.4171e23
MARS_HILL = (6.4171e23 / 1.98892e30 / 3.0) ** (1.0 / 3.0) * 1.52366231 * 1.495978707e11   # same as the solver


def _grid(vmax: float, n_r: int, n_t: int):
    # midpoint grid in (v_r, v_t); weights include the 2*pi*v_t azimuthal Jacobian
    vr = (np.arange(n_r) + 0.5) / n_r * 2 * vmax - vmax
    vt = (np.arange(n_t) + 0.5) / n_t * vmax
    dvr, dvt = 2 * vmax / n_r, vmax / n_t
    VR, VT = np.meshgrid(vr, vt, indexing="ij")
    return VR, VT, 2 * np.pi * VT * dvr * dvt


def _accessible(r, VR, VT, mu, r0, r_top, ubc, r_hill):
    v2 = VR**2 + VT**2
    eps = 0.5 * v2 - mu / r
    h = r * VT
    # radial speed squared at r0 >= 0  <=>  periapsis <= r0 (orbit passes through the source sphere)
    vr0_sq = 2 * (eps + mu / r0) - (h / r0) ** 2
    from_source = vr0_sq >= 0
    bound = eps < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(bound, -mu / (2 * eps), np.inf)
        e = np.sqrt(np.maximum(0.0, 1 + 2 * eps * h**2 / mu**2))
        r_apo = np.where(bound, a * (1 + e), np.inf)
    out = from_source & (VR > 0)
    if ubc in ("reflect", "ballistic"):
        back = bound & (r_apo <= r_hill) if ubc == "ballistic" else bound
    elif ubc == "free_escape":
        back = bound & (r_apo <= r_top)
    else:
        raise ValueError(f"unknown ubc {ubc!r}")
    return out | (from_source & (VR < 0) & back)


def profile(r, n0: float, T: float, mass: float, r0: float, r_top: float, ubc: str = "reflect",
            planet_mass: float = MARS_MASS, r_hill: float = MARS_HILL, n_v: int = 1200,
            vmax_inject: float | None = None) -> xr.Dataset:
    """Number density, kinetic temperatures and radial flow velocity of the collisionless exosphere at radii r [m].

    ``vmax_inject`` mimics the solver's truncation of the injected speed at ``max_MB_vel_dis`` (applied to the
    speed at r0; the truncated fraction is negligible when it is several thermal speeds above escape).
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    mu = G * planet_mass
    beta = mass / (2 * K_B * T)
    norm = n0 * (beta / np.pi) ** 1.5
    out = {k: np.zeros(r.size) for k in ("n", "vr_mean", "T_kin", "T_rad", "T_perp")}
    for i, ri in enumerate(r):
        # f decays like exp(-beta (v^2 - 2mu/r + 2mu/r0)); 7 thermal speeds above v=0 is ample
        vmax = np.sqrt(7.0**2 / beta)
        VR, VT, W = _grid(vmax, 2 * n_v, n_v)
        acc = _accessible(ri, VR, VT, mu, r0, r_top, ubc, r_hill)
        v2 = VR**2 + VT**2
        e_src = v2 - 2 * mu / ri + 2 * mu / r0          # speed^2 the particle had at r0
        f = norm * np.exp(-beta * e_src) * acc
        if vmax_inject is not None:
            f = f * (e_src <= vmax_inject**2)
        w = f * W
        n = w.sum()
        vr_m = (w * VR).sum() / n
        var_r = (w * (VR - vr_m) ** 2).sum() / n
        var_t = (w * VT**2).sum() / n / 2.0               # per tangential component
        out["n"][i] = n
        out["vr_mean"][i] = vr_m
        out["T_rad"][i] = mass * var_r / K_B
        out["T_perp"][i] = mass * var_t / K_B
        out["T_kin"][i] = (out["T_rad"][i] + 2 * out["T_perp"][i]) / 3
    ds = xr.Dataset({
        "num_dens": ("radius", out["n"], {"units": "m-3"}),
        "vel_rad": ("radius", out["vr_mean"], {"units": "m s-1"}),
        "temp_trans": ("radius", out["T_kin"], {"units": "K"}),
        "temp_rad": ("radius", out["T_rad"], {"units": "K"}),
        "temp_perp": ("radius", out["T_perp"], {"units": "K"}),
    }, coords={"radius": ("radius", r, {"units": "m"})})
    ds.attrs.update(n0=n0, T=T, mass=mass, r0=r0, r_top=r_top, ubc=ubc, source="dsmclib.exosphere.profile")
    return ds


def jeans_rate(n0: float, T: float, mass: float, r0: float, planet_mass: float = MARS_MASS) -> float:
    """Jeans escape rate [s-1] from the source sphere: 4 pi r0^2 * n0 v0 / (2 sqrt(pi)) (1 + lam) exp(-lam)."""
    lam = G * planet_mass * mass / (K_B * T * r0)
    v0 = np.sqrt(2 * K_B * T / mass)
    return 4 * np.pi * r0**2 * n0 * v0 / (2 * np.sqrt(np.pi)) * (1 + lam) * np.exp(-lam)


def escape_rate(n0: float, T: float, mass: float, r0: float, r_top: float, ubc: str = "reflect",
                planet_mass: float = MARS_MASS, r_hill: float = MARS_HILL, n_v: int = 1500) -> float:
    """Rate [s-1] at which computational particles are counted as escaping at r_top for a collisionless exosphere.

    reflect: unbound outward flux (= Jeans at r0); ballistic: plus bound orbits with apoapsis beyond r_hill;
    free_escape: every outward crossing of r_top (unbound + all bound orbits reaching r_top).
    """
    mu = G * planet_mass
    beta = mass / (2 * K_B * T)
    norm = n0 * (beta / np.pi) ** 1.5
    vmax = np.sqrt(7.0**2 / beta + 2 * mu / r0)
    VR, VT, W = _grid(vmax, 2 * n_v, n_v)
    v2 = VR**2 + VT**2
    eps = 0.5 * v2 - mu / r_top
    h = r_top * VT
    from_source = 2 * (eps + mu / r0) - (h / r0) ** 2 >= 0
    up = from_source & (VR > 0)
    bound = eps < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(bound, -mu / (2 * eps), np.inf)
        e = np.sqrt(np.maximum(0.0, 1 + 2 * eps * h**2 / mu**2))
        r_apo = np.where(bound, a * (1 + e), np.inf)
    if ubc == "reflect":
        counted = up & ~bound
    elif ubc == "ballistic":
        counted = up & (~bound | (r_apo > r_hill))
    elif ubc == "free_escape":
        counted = up
    else:
        raise ValueError(f"unknown ubc {ubc!r}")
    f = norm * np.exp(-beta * (v2 - 2 * mu / r_top + 2 * mu / r0))
    flux = (f * counted * VR * W).sum()
    return 4 * np.pi * r_top**2 * flux


def held_count(n0: float, T: float, mass: float, r0: float, r_top: float, planet_mass: float = MARS_MASS,
               r_hill: float = MARS_HILL, n_v: int = 1500, t=None):
    """Number of molecules held above r_top by the Kepler-hold ballistic UBC:
    sum over bound outward crossings of (crossing rate x time of flight back to r_top).

    t=None gives the asymptotic steady value. Near-escape orbits have times of flight up to ~1e7 s
    (bounded by r_hill), so a run reaches it only after months. Passing t (seconds since an empty
    start with collisionless transport; scalar or 1-D array) gives the transient count instead:
    sum of rate x clip(t - t_transit, 0, tof), with t_transit = Kepler time from r0 out to r_top."""
    mu = G * planet_mass
    beta = mass / (2 * K_B * T)
    norm = n0 * (beta / np.pi) ** 1.5
    vmax = np.sqrt(7.0**2 / beta + 2 * mu / r0)
    VR, VT, W = _grid(vmax, 2 * n_v, n_v)
    v2 = VR**2 + VT**2
    eps = 0.5 * v2 - mu / r_top
    h = r_top * VT
    sel = (2 * (eps + mu / r0) - (h / r0) ** 2 >= 0) & (VR > 0) & (eps < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -mu / (2 * eps)
        ec = 1 - r_top / a
        es = r_top * VR / np.sqrt(mu * a)
        e = np.hypot(ec, es)
        sel &= a * (1 + e) <= r_hill
        n_mean = np.sqrt(mu / a**3)
        E_top = np.arctan2(es, ec)                      # outbound crossing, in [0, pi]
        tof = 2 * (np.pi - (E_top - es)) / n_mean       # 2 x (time from crossing to apoapsis)
        ec0 = 1 - r0 / a
        es0 = np.sqrt(np.maximum(e**2 - ec0**2, 0.0))   # outbound at r0
        t_transit = ((E_top - es) - (np.arctan2(es0, ec0) - es0)) / n_mean
    f = norm * np.exp(-beta * (v2 - 2 * mu / r_top + 2 * mu / r0))
    rate = np.where(sel, f * VR * W, 0.0) * 4 * np.pi * r_top**2
    tof = np.where(sel, tof, 0.0)
    if t is None:
        return float(np.nansum(rate * tof))
    t_transit = np.where(sel, t_transit, 0.0)
    out = np.array([np.nansum(rate * np.clip(ti - t_transit, 0.0, tof)) for ti in np.atleast_1d(t)])
    return float(out[0]) if np.ndim(t) == 0 else out


def ballistic_rates(n0: float, T: float, mass: float, r0: float, r_top: float, planet_mass: float = MARS_MASS,
                    r_hill: float = MARS_HILL, n_v: int = 1500, t=None) -> dict:
    """Collisionless Liouville-Kepler ballistic entry rate Phi_ball_in(t) and return rate
    Phi_ball_ret(t) [s-1] across r_top for bound orbits with r_apo <= r_hill.

    Each velocity-grid element crossing r_top outbound on a bound orbit (r_apo <= r_hill) starts
    arriving at r_top at t_transit (Kepler time from r0 to r_top) and begins returning inbound
    across r_top at t_transit + tof (where tof is the Kepler flight time above r_top).
    Thus at time t since an empty start:
        Phi_ball_in(t)  = sum_{t >= t_transit} rate
        Phi_ball_ret(t) = sum_{t >= t_transit + tof} rate
    and d(held_count)/dt = Phi_ball_in(t) - Phi_ball_ret(t).
    """
    mu = G * planet_mass
    beta = mass / (2 * K_B * T)
    norm = n0 * (beta / np.pi) ** 1.5
    vmax = np.sqrt(7.0**2 / beta + 2 * mu / r0)
    VR, VT, W = _grid(vmax, 2 * n_v, n_v)
    v2 = VR**2 + VT**2
    eps = 0.5 * v2 - mu / r_top
    h = r_top * VT
    sel = (2 * (eps + mu / r0) - (h / r0) ** 2 >= 0) & (VR > 0) & (eps < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -mu / (2 * eps)
        ec = 1 - r_top / a
        es = r_top * VR / np.sqrt(mu * a)
        e = np.hypot(ec, es)
        sel &= a * (1 + e) <= r_hill
        n_mean = np.sqrt(mu / a**3)
        E_top = np.arctan2(es, ec)
        tof = 2 * (np.pi - (E_top - es)) / n_mean
        ec0 = 1 - r0 / a
        es0 = np.sqrt(np.maximum(e**2 - ec0**2, 0.0))
        t_transit = ((E_top - es) - (np.arctan2(es0, ec0) - es0)) / n_mean
    f = norm * np.exp(-beta * (v2 - 2 * mu / r_top + 2 * mu / r0))
    rate = np.where(sel, f * VR * W, 0.0) * 4 * np.pi * r_top**2
    asym = float(np.nansum(rate))
    if t is None:
        return {"ballistic_in_rate": asym, "ballistic_return_rate": asym, "asymptotic_rate": asym}
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    t_tr = np.where(sel, t_transit, np.inf)
    t_ret = np.where(sel, t_transit + tof, np.inf)
    r_in = np.array([np.nansum(rate[ti >= t_tr]) for ti in t_arr])
    r_ret = np.array([np.nansum(rate[ti >= t_ret]) for ti in t_arr])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r_ret > 0, r_in / r_ret, np.nan)
    if np.ndim(t) == 0:
        return {"ballistic_in_rate": float(r_in[0]), "ballistic_return_rate": float(r_ret[0]),
                "ratio_in_over_return": float(ratio[0]), "asymptotic_rate": asym}
    return {"ballistic_in_rate": r_in, "ballistic_return_rate": r_ret,
            "ratio_in_over_return": ratio, "asymptotic_rate": asym}
