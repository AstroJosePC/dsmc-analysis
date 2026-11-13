"""Escape rates: DSMC count at the top boundary, Jeans estimate, lower-boundary balance.

Under UBC_FREE_ESCAPE every particle that crosses the top boundary is removed and counted
(escape_rate, physical s-1 through the whole top sphere), whatever its speed. The direct DSMC
rate is resolution-limited: one escaping computational particle in a window of length dt_avg registers as
W / dt_avg. With zero counts we report the Poisson 95% upper limit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from scipy import stats

from .constants import G_NEWTON, K_BOLTZMANN
from .convergence import batch_means, inventory
from .ncrun import NcRun
from .profiles import exobase, interp_at_altitude

POISSON_UL95_ZERO = -np.log(0.05)  # 2.996: 95% upper limit on a Poisson mean when 0 events are seen


def _window(run: NcRun, t0: float, t1: Optional[float]):
    t = run.time
    t1 = float(t[-1]) if t1 is None else float(t1)
    sel = (t > t0) & (t <= t1)
    return np.where(sel)[0], float(t0), t1


def poisson_upper_limit(k: int, conf: float = 0.95) -> float:
    """One-sided upper limit on a Poisson mean given k observed events."""
    return float(stats.chi2.ppf(conf, 2 * (k + 1)) / 2.0)


def dsmc_escape(run: NcRun, t0: float, t1: Optional[float] = None) -> xr.Dataset:
    """Mean escape rate over (t0, t1] per species, number of escaping computational particles k, and the
    95% upper limit W * UL(k) / (t1 - t0) (equal to the rate bound when k = 0)."""
    idx, t0, t1 = _window(run, t0, t1)
    esc = run.escape.isel(time=idx)
    dt_avg = run.macro["avg_time_interval"].isel(time=idx)
    k = (esc["num_part_escape"].sum("boundary_cell") * dt_avg).sum("time")
    k = np.rint(k).astype(int)
    duration = float(dt_avg.sum())
    rate = esc["escape_rate"].sum("boundary_cell").mean("time")
    ul = xr.DataArray([poisson_upper_limit(int(v)) for v in k.values], dims="species",
                      coords={"species": k["species"]}) * run.weight / duration
    resolution = run.weight / float(dt_avg.mean())
    area = float(esc["upper_area"].sum())
    return xr.Dataset({
        "escape_rate": rate.assign_attrs(units="s-1", long_name="DSMC escape rate, time mean"),
        "escape_npart": k.assign_attrs(units="1", long_name="Escaping computational particles in the window"),
        "escape_rate_ul95": ul.assign_attrs(units="s-1", long_name="95% upper limit on the escape rate"),
        "escape_rate_resolution": resolution.assign_attrs(
            units="s-1", long_name="Rate of one escaping computational particle per output window"),
        "escape_flux_ul95": (ul / area).assign_attrs(units="m-2 s-1", long_name="95% upper limit on the escape flux at the top boundary"),
    }, attrs={"t0": t0, "t1": t1, "duration": duration, "upper_area": area,
              "top_radius": float(np.sqrt(area / (4 * np.pi)))})


def jeans_escape(n, temperature, mass, radius, planet_mass):
    """Jeans escape flux and rate through a sphere of radius r.

    lambda = G M m / (k T r);  v0 = sqrt(2 k T / m);
    flux = n v0 / (2 sqrt(pi)) (1 + lambda) exp(-lambda);  rate = 4 pi r^2 flux.
    Works with numpy arrays and xarray objects alike.
    """
    lam = G_NEWTON * planet_mass * mass / (K_BOLTZMANN * temperature * radius)
    v0 = np.sqrt(2.0 * K_BOLTZMANN * temperature / mass)
    flux = n * v0 / (2.0 * np.sqrt(np.pi)) * (1.0 + lam) * np.exp(-lam)
    return {"lambda": lam, "flux": flux, "rate": 4.0 * np.pi * radius ** 2 * flux}


def radial_travel_time(v0, r0: float, r1: float, planet_mass: float, n_quad: int = 400) -> np.ndarray:
    """Time for a particle launched radially outward at speed v0 (>= escape speed) from r0 to
    reach r1 in a point-mass gravity field (no collisions). Gauss-Legendre in s = sqrt(r - r0),
    which removes the 1/sqrt singularity of the slow branch."""
    v0 = np.atleast_1d(np.asarray(v0, dtype=float))
    mu = G_NEWTON * planet_mass
    x, w = np.polynomial.legendre.leggauss(n_quad)
    s1 = np.sqrt(r1 - r0)
    s = 0.5 * s1 * (x + 1.0)
    r = r0 + s ** 2
    vr = np.sqrt(np.maximum(v0[:, None] ** 2 - 2 * mu * (1 / r0 - 1 / r[None, :]), 1e-30))
    return (0.5 * s1 * w[None, :] * 2 * s[None, :] / vr).sum(axis=1)


def transit_expected_count(rate: float, weight: float, temperature: float, mass: float, r_exo: float,
                           r_top: float, planet_mass: float, t_on: float, t_obs0: float, t_obs1: float,
                           n_speed: int = 200) -> dict:
    """Expected number of escaping computational particles that reach the top boundary inside the counting
    window (t_obs0, t_obs1], for a constant escape rate `rate` [s-1] switched on at t_on at the
    exobase, with the flux-weighted Maxwellian speed distribution above escape speed (Jeans) and
    radial trajectories. Radial launches are the fastest way up, so this is an upper bound on
    the count for a given rate; a rate that is still rising (filling) lowers it further.
    """
    mu = G_NEWTON * planet_mass
    v_esc = np.sqrt(2 * mu / r_exo)
    vth = np.sqrt(2 * K_BOLTZMANN * temperature / mass)
    # Escaping flux density in speed ~ v^3 exp(-v^2/vth^2), v > v_esc; sample on u = v^2/vth^2
    u_esc = (v_esc / vth) ** 2
    du = 40.0 / n_speed
    u = u_esc + du * (np.arange(n_speed) + 0.5)  # midpoints; the tail beyond u_esc + 40 is negligible
    pdf = u * np.exp(-(u - u_esc))  # v^3 exp(-v^2/vth^2) dv ∝ u exp(-u) du (shifted to avoid underflow)
    pdf /= pdf.sum()
    tau = radial_travel_time(vth * np.sqrt(u), r_exo, r_top, planet_mass)
    # Launch times t_l in (t_on, t_obs1]; arrival t_l + tau counted if in (t_obs0, t_obs1]
    lo = np.clip(t_obs0 - tau, t_on, None)
    hi = np.clip(t_obs1 - tau, t_on, None)
    eff_time = float((pdf * np.clip(hi - lo, 0, None)).sum())
    expected = rate * eff_time / weight
    naive = rate * (t_obs1 - t_obs0) / weight
    return {"expected_npart": expected, "naive_expected_npart": naive, "effective_time": eff_time,
            "median_travel_time": float(np.interp(0.5, np.cumsum(pdf), tau)),
            "min_travel_time": float(tau.min()), "v_esc_exobase": float(v_esc),
            "p_zero": float(np.exp(-expected))}


def jeans_at_lower_boundary(run: NcRun) -> xr.Dataset:
    """Jeans quantities from the Info lower-boundary setup (n0, T0 at the bottom radius).
    Reproduces the solver's Surface_Cells.ascii Jeans columns; used as a check."""
    s = run.info_surface
    r0 = run.radial_edges()[0]
    j = jeans_escape(s["num_dens"].isel(surface_cell=0), float(s["T0"].values.ravel()[0]),
                     run.mass, r0, run.planet_mass)
    return xr.Dataset({"jeans_parameter": j["lambda"], "jeans_flux": j["flux"], "jeans_rate": j["rate"]},
                      attrs={"radius": r0})


def jeans_at_exobase(run: NcRun, prof: xr.Dataset, temperature_field: str = "temp_trans") -> xr.Dataset:
    """Jeans escape evaluated at each species' DSMC exobase (Kn = 1) with the DSMC time-mean
    density and temperature interpolated there. NaN when the exobase is outside the domain."""
    z_exo = exobase(prof)
    n_exo = interp_at_altitude(prof, "num_dens", z_exo, log=True)
    t_exo = interp_at_altitude(prof, temperature_field, z_exo)
    r_exo = run.planet_radius + z_exo * 1e3
    j = jeans_escape(n_exo, t_exo, run.mass, r_exo, run.planet_mass)
    return xr.Dataset({
        "exobase_altitude": z_exo,
        "n_exobase": n_exo.assign_attrs(units="m-3"),
        "T_exobase": t_exo.assign_attrs(units="K"),
        "jeans_parameter": j["lambda"].assign_attrs(units="1"),
        "jeans_flux": j["flux"].assign_attrs(units="m-2 s-1"),
        "jeans_rate": j["rate"].assign_attrs(units="s-1", long_name="Jeans escape rate at the exobase"),
    })


def lower_boundary_balance(run: NcRun, t0: float, t1: Optional[float] = None, n_batches: int = 10) -> xr.Dataset:
    """Net outflow implied by mass balance: <inject - return> - dN/dt over (t0, t1].

    In steady state this equals the loss through the top (escape); while the domain is still
    filling, dN/dt absorbs most of the net injection. Errors from batch means (net source) and
    from the batch-means slope fit (dN/dt), added in quadrature.
    """
    idx, t0, t1 = _window(run, t0, t1)
    t = run.time[idx]
    net = (run.surface["inject_rate"] - run.surface["return_rate"]).sum("boundary_cell").isel(time=idx)
    n_phys = inventory(run).N_phys.isel(time=idx)
    rows = {k: [] for k in ("net_source", "net_source_sem", "dN_dt", "dN_dt_se", "outflow", "outflow_se")}
    for sp in run.species:
        y = net.sel(species=sp).values
        b = batch_means(y, n_batches)
        rows["net_source"].append(y.mean())
        rows["net_source_sem"].append(b.std(ddof=1) / np.sqrt(n_batches))
        fit = stats.linregress(batch_means(t, n_batches), batch_means(n_phys.sel(species=sp).values, n_batches))
        rows["dN_dt"].append(fit.slope)
        rows["dN_dt_se"].append(fit.stderr)
    rows["outflow"] = list(np.array(rows["net_source"]) - np.array(rows["dN_dt"]))
    rows["outflow_se"] = list(np.hypot(rows["net_source_sem"], rows["dN_dt_se"]))
    coords = {"species": run.species}
    ds = xr.Dataset({k: ("species", np.array(v, dtype=float), {"units": "s-1"}) for k, v in rows.items()},
                    coords=coords, attrs={"t0": t0, "t1": t1})
    ds["outflow"].attrs["long_name"] = "Net outflow implied by mass balance (inject - return - dN/dt)"
    return ds
