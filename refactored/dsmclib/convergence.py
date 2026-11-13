"""Approach to steady state: domain inventories, boundary rates and a drift test.

    from dsmclib.ncrun import open_run_nc
    from dsmclib import convergence
    run = open_run_nc("OH2_B_r1")
    inv = convergence.inventory(run)                  # N_phys, N_part per species vs time
    convergence.steady_window(inv.N_phys.sel(species="O"), t_end=run.wave_start_time)

All outputs are window means (accumulators reset after each output), so consecutive values
are close to independent apart from the physical memory of the system; block (batch) means
are used for error bars throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import xarray as xr
from scipy import stats

from .ncrun import NcRun


def inventory(run: NcRun) -> xr.Dataset:
    """Number of molecules in the domain per species and time.

    N_phys = sum_r n(r) V(r) [molecules]; N_part = sum_r num_part(r) [computational particles].
    For a constant per-species weight W, N_phys = W * N_part to rounding.
    """
    vol = run.cell_volume()
    n_phys = (run.macro["num_dens"] * vol).sum("radial")
    n_part = run.macro["num_part"].sum("radial")
    return xr.Dataset({
        "N_phys": n_phys.assign_attrs(units="1", long_name="Molecules in the domain"),
        "N_part": n_part.assign_attrs(units="1", long_name="Computational particles in the domain (mean per sample)"),
    })


def boundary_rates(run: NcRun) -> xr.Dataset:
    """Physical rates through the domain boundaries [s-1], summed over boundary cells.

    inject - return - escape is the net source; d(N_phys)/dt is its counterpart from the
    inventory (centred differences on the output times).
    """
    out = {}
    if run.surface is not None:
        out["inject_rate"] = run.surface["inject_rate"].sum("boundary_cell")
        out["return_rate"] = run.surface["return_rate"].sum("boundary_cell")
    if run.escape is not None:
        out["escape_rate"] = run.escape["escape_rate"].sum("boundary_cell")
        out["escape_npart"] = (run.escape["num_part_escape"].sum("boundary_cell")
                                   * run.macro["avg_time_interval"]).assign_attrs(
            units="1", long_name="Escaping computational particles per output window")
    ds = xr.Dataset(out)
    if {"inject_rate", "return_rate", "escape_rate"} <= set(ds):
        ds["net_source"] = ds.inject_rate - ds.return_rate - ds.escape_rate
    inv = inventory(run).N_phys
    ds["dN_dt"] = xr.DataArray(np.gradient(inv.values, run.time, axis=inv.get_axis_num("time")),
                               dims=inv.dims, coords=inv.coords)
    for v in ds:
        ds[v].attrs.setdefault("units", "s-1")
    return ds


def batch_means(y: np.ndarray, n_batches: int) -> np.ndarray:
    """Means of n_batches contiguous blocks along axis 0 (trailing remainder dropped)."""
    y = np.asarray(y)
    m = len(y) // n_batches
    if m < 1:
        raise ValueError(f"{len(y)} samples cannot make {n_batches} batches")
    return y[: m * n_batches].reshape((n_batches, m) + y.shape[1:]).mean(axis=1)


def mean_and_sem(y: np.ndarray, n_batches: int = 10):
    """Mean over axis 0 and its batch-means standard error."""
    b = batch_means(y, n_batches)
    return np.asarray(y).mean(axis=0), b.std(axis=0, ddof=1) / np.sqrt(n_batches)


def drift_test(t: np.ndarray, y: np.ndarray, n_batches: int = 10, conf: float = 0.95) -> dict:
    """Linear drift of y over t, estimated on batch means (robust to autocorrelation).

    Returns slope [y/s], its standard error, the relative drift over the segment
    (slope * duration / mean) and the upper bound of |relative drift| at `conf`.
    """
    tb, yb = batch_means(t, n_batches), batch_means(y, n_batches)
    fit = stats.linregress(tb, yb)
    tcrit = stats.t.ppf(0.5 + conf / 2, n_batches - 2)
    duration = t[-1] - t[0]
    mean = float(np.mean(y))
    rel = fit.slope * duration / mean if mean else np.nan
    rel_ub = (abs(fit.slope) + tcrit * fit.stderr) * duration / abs(mean) if mean else np.nan
    return {"slope": float(fit.slope), "slope_se": float(fit.stderr), "duration": float(duration),
            "mean": mean, "rel_drift": float(rel), "rel_drift_ub": float(rel_ub),
            "slope_t": float(fit.slope / fit.stderr) if fit.stderr > 0 else np.inf}


@dataclass
class SteadyResult:
    steady: bool
    t_start: Optional[float]  # earliest start of a steady segment ending at t_end (None if none)
    t_end: float
    rel_tol: float
    rel_drift: float  # of the tested segment (the steady one, or the last-fraction one if none)
    rel_drift_ub: float
    slope: float
    slope_se: float
    mean: float
    drift_per_ks: float  # relative change per 1000 s on the tested segment

    def as_dict(self) -> dict:
        return asdict(self)


def steady_window(series: xr.DataArray, t_end: Optional[float] = None, rel_tol: float = 0.05,
                  min_frac: float = 0.2, n_starts: int = 40, n_batches: int = 10,
                  conf: float = 0.95) -> SteadyResult:
    """Earliest t_start such that the series on [t_start, t_end] has no drift above rel_tol.

    Criterion: the `conf` upper bound of |slope| * (t_end - t_start) / mean is below rel_tol,
    i.e. the data rule out a relative change larger than rel_tol over the segment. Candidate
    starts are spaced evenly between the first output and t_end - min_frac * (t_end - t0).
    t_end defaults to the last output; pass the forcing start to test only the pre-forcing record.
    """
    t = series["time"].values
    y = series.values
    t_end = float(t[-1]) if t_end is None else float(t_end)
    keep = t <= t_end
    t, y = t[keep], y[keep]
    span = t[-1] - t[0]
    last = None
    for t0 in np.linspace(t[0], t[-1] - min_frac * span, n_starts):
        sel = t >= t0
        if sel.sum() < 3 * n_batches:
            break
        d = drift_test(t[sel], y[sel], n_batches, conf)
        last = (t0, d)
        if d["rel_drift_ub"] < rel_tol:
            return SteadyResult(True, float(t0), t_end, rel_tol, d["rel_drift"], d["rel_drift_ub"],
                                d["slope"], d["slope_se"], d["mean"], 1e3 * d["slope"] / d["mean"])
    t0, d = last
    return SteadyResult(False, None, t_end, rel_tol, d["rel_drift"], d["rel_drift_ub"],
                        d["slope"], d["slope_se"], d["mean"], 1e3 * d["slope"] / d["mean"])


def filling_timescale(series: xr.DataArray, t0: float, t1: Optional[float] = None) -> dict:
    """Fit N(t) = N_inf - A exp(-t / tau) on [t0, t1]; returns N_inf, tau and their 1-sigma errors.

    Used for a species that has not converged, to extrapolate its steady inventory. The fit
    is unreliable when tau is much longer than the fitted segment; `tau_over_span` flags it.
    """
    from scipy.optimize import curve_fit

    t = series["time"].values
    y = series.values
    t1 = t[-1] if t1 is None else t1
    sel = (t >= t0) & (t <= t1)
    tt = t[sel] - t0
    scale = float(np.mean(y[sel]))  # fit in units of the mean: keeps the covariance well conditioned
    yy = y[sel] / scale
    span = tt[-1]

    def model(x, n_inf, a, tau):
        return n_inf - a * np.exp(-x / tau)

    p0 = (yy[-1] * 1.5, yy[-1] * 1.5 - yy[0], span)
    nan = {"N_inf": np.nan, "N_inf_err": np.nan, "tau": np.nan, "tau_err": np.nan,
           "tau_over_span": np.nan, "fraction_filled": np.nan, "t0": float(t0), "t1": float(t1)}
    try:
        popt, pcov = curve_fit(model, tt, yy, p0=p0, bounds=([0, -np.inf, 1.0], [np.inf, np.inf, 1e9]),
                               maxfev=20000)
    except RuntimeError:
        return nan
    # Inflate the covariance for serially correlated residuals (AR(1): n / n_eff = (1 + rho) / (1 - rho))
    resid = yy - model(tt, *popt)
    rho = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0.0, 0.99)) if resid.std() > 0 else 0.0
    perr = np.sqrt(np.diag(pcov) * (1 + rho) / (1 - rho))
    return {"N_inf": float(popt[0] * scale), "N_inf_err": float(perr[0] * scale), "tau": float(popt[2]),
            "tau_err": float(perr[2]), "tau_over_span": float(popt[2] / span),
            "fraction_filled": float(y[sel][-1] / (popt[0] * scale)), "resid_lag1": rho,
            "t0": float(t0), "t1": float(t1)}
