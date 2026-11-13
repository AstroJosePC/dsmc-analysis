"""Response of a 1-D, multi-species run to the lower-boundary flux wave.

The forcing modulates the injected flux at angular frequency omega from the wave start step.
For every species and altitude we fit, over the whole number of forcing periods that fit
inside the run,

    y(t) = a + b (t - t_mid) + A cos(omega t) + B sin(omega t)

to y = x / x_bg - 1 (x_bg: time mean over a pre-forcing window), or to y = x - x_bg for
`absolute_fields` (vel_rad: its background mean is ~0, so a relative change is meaningless;
amplitudes are then in m s-1). The linear term absorbs a
slow drift (e.g. a species still filling the domain), which would otherwise leak into a plain
lock-in over a finite record. Amplitude = sqrt(A^2 + B^2); phase = atan2(-B, A), i.e. y ~
amp cos(omega t + phase). Standard errors use the residual lag-1 autocorrelation to reduce
the effective sample size. The plain lock-in projection (attenuation.lockin_projection) of
the detrended series is returned as a cross-check.

Noise floor (null test): the same fit at omega over the last whole forcing periods that end at
the forcing start inside the background window, where no forcing is present. Its amplitude is
what shot noise alone produces at omega, and null_amp / null_amp_se calibrates the AR(1)
errors (for pure noise it follows a Rayleigh distribution: 95% below 2.45).

The registry/Info "End Step of Wave Injection" can lie beyond the last step (OH2_A: 10
configured periods, ~4.2 simulated); the period count used here comes from the outputs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .attenuation import lockin_projection
from .ncrun import NcRun
from .profiles import time_mean_profiles


def forcing_window(run: NcRun) -> Optional[dict]:
    """Start, end and number of whole forcing periods inside the run (None if no forcing)."""
    t_start, omega = run.wave_start_time, run.wave_omega
    if t_start is None or not omega:
        return None
    period = 2 * np.pi / omega
    n_periods = int(np.floor((run.t_end - t_start) / period + 1e-9))
    return {"t_start": t_start, "period": period, "omega": omega, "n_periods": n_periods,
            "t_end": t_start + n_periods * period, "periods_in_run": (run.t_end - t_start) / period}


def _harmonic_fit(t: np.ndarray, y: np.ndarray, omega: float):
    """Least-squares fit of a + b t + A cos + B sin to y (time on axis 0, any trailing shape)."""
    tc = t - t.mean()
    X = np.column_stack([np.ones_like(t), tc, np.cos(omega * t), np.sin(omega * t)])
    shape = y.shape[1:]
    Y = y.reshape(len(t), -1)
    good = np.all(np.isfinite(Y), axis=0)
    coef = np.full((4, Y.shape[1]), np.nan)
    se = np.full((4, Y.shape[1]), np.nan)
    if good.any():
        c, *_ = np.linalg.lstsq(X, Y[:, good], rcond=None)
        resid = Y[:, good] - X @ c
        n, p = X.shape
        rho = np.array([np.corrcoef(r[:-1], r[1:])[0, 1] if np.std(r) > 0 else 0.0 for r in resid.T])
        rho = np.clip(np.nan_to_num(rho), 0.0, 0.95)
        inflation = (1 + rho) / (1 - rho)  # n / n_eff for AR(1) residuals
        s2 = (resid ** 2).sum(axis=0) / max(n - p, 1)  # OLS residual variance
        cov_unit = np.diag(np.linalg.inv(X.T @ X))
        coef[:, good] = c
        se[:, good] = np.sqrt(np.outer(cov_unit, s2 * inflation))
    return coef.reshape((4,) + shape), se.reshape((4,) + shape)


def _amp_phase(coef, se):
    A, B = coef[2], coef[3]
    amp = np.hypot(A, B)
    phase = np.arctan2(-B, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp_se = np.sqrt((A * se[2]) ** 2 + (B * se[3]) ** 2) / amp
        phase_se = np.sqrt((B * se[2]) ** 2 + (A * se[3]) ** 2) / amp ** 2
    return amp, amp_se, phase, phase_se


def _perturbation(x, bg, absolute: bool):
    return x - bg if absolute else x / bg - 1.0


def null_window(run: NcRun, background: tuple[float, float]) -> Optional[tuple[float, float, int]]:
    """(t0, t1, n) for the last n whole forcing periods inside the background window that end at
    min(background end, forcing start); n is capped at the number of periods in the forcing fit."""
    fw = forcing_window(run)
    if fw is None:
        return None
    t1 = min(float(background[1]), fw["t_start"])
    n = min(fw["n_periods"], int(np.floor((t1 - float(background[0])) / fw["period"] + 1e-9)))
    return (t1 - n * fw["period"], t1, n) if n >= 1 else None


def forcing_response(run: NcRun, background: tuple[float, float],
                     fields: Sequence[str] = ("num_dens", "temp_trans", "vel_rad"),
                     absolute_fields: Sequence[str] = ("vel_rad",), min_part: float = 10.0) -> xr.Dataset:
    """Perturbation amplitude and phase vs altitude for each species and field (relative, or
    absolute for `absolute_fields`), the pre-forcing noise floor of each, plus the response of
    the domain inventory and the forcing itself (lower-boundary inject rate).

    background: (t0, t1) of the pre-forcing steady window used for x_bg.
    min_part: cells whose background holds fewer computational particles (mean per sample) than this are
    masked (NaN): their relative fluctuations are shot noise, not a resolved response.
    """
    fw = forcing_window(run)
    if fw is None or fw["n_periods"] < 1:
        raise ValueError(f"{run.name}: no complete forcing period inside the run")
    omega = fw["omega"]
    bg = time_mean_profiles(run, *background, fields=tuple(dict.fromkeys(tuple(fields) + ("num_part",))))
    resolved = bg["num_part"] >= min_part
    t = run.time
    sel = (t > fw["t_start"]) & (t <= fw["t_end"] + 1e-9)
    tt = t[sel]
    m = run.macro.isel(time=np.where(sel)[0])
    nw = null_window(run, background)
    if nw is not None:
        nsel = (t > nw[0]) & (t <= nw[1] + 1e-9)
        m_null, t_null = run.macro.isel(time=np.where(nsel)[0]), t[nsel]
    out = {}
    for f in fields:
        absolute = f in absolute_fields
        unit = m[f].attrs.get("units", "") if absolute else "1"
        kind = "Absolute" if absolute else "Relative"
        rel = _perturbation(m[f], bg[f], absolute).where(resolved).transpose("time", "species", "radial")
        coef, se = _harmonic_fit(tt, rel.values, omega)
        amp, amp_se, ph, ph_se = _amp_phase(coef, se)
        trend = coef[1] * (tt[-1] - tt[0])
        detr = rel.values - (coef[0][None] + np.einsum("t,...->t...", tt - tt.mean(), coef[1]))
        lock = np.where(np.isfinite(amp), np.abs(lockin_projection(np.nan_to_num(detr), tt, omega)), np.nan)
        dims = ("species", "radial")
        out[f"{f}_amp"] = (dims, amp, {"units": unit, "long_name": f"{kind} {f} perturbation amplitude at omega"})
        out[f"{f}_amp_se"] = (dims, amp_se, {"units": unit})
        out[f"{f}_phase"] = (dims, ph, {"units": "rad", "long_name": "Phase of amp*cos(omega t + phase)"})
        out[f"{f}_phase_se"] = (dims, ph_se, {"units": "rad"})
        out[f"{f}_trend"] = (dims, trend, {"units": unit, "long_name": "Linear drift over the fit window"})
        out[f"{f}_lockin_amp"] = (dims, lock, {"units": unit, "long_name": "Lock-in amplitude of the detrended series"})
        if nw is not None:
            rel0 = _perturbation(m_null[f], bg[f], absolute).where(resolved).transpose("time", "species", "radial")
            c0, s0 = _harmonic_fit(t_null, rel0.values, omega)
            a0, a0_se, _, _ = _amp_phase(c0, s0)
            out[f"{f}_null_amp"] = (dims, a0, {"units": unit, "long_name": (
                f"Noise floor: {kind.lower()} {f} amplitude at omega over the {nw[2]} periods before forcing")})
            out[f"{f}_null_amp_se"] = (dims, a0_se, {"units": unit})
    ds = xr.Dataset(out, coords={"species": m["species"], "radial": m["radial"], "altitude": m["altitude"]})
    ds["background_num_part"] = bg["num_part"].assign_attrs(long_name="Background computational particles per cell (mean per sample)")
    ds["resolved"] = resolved.assign_attrs(long_name=f"Background holds >= {min_part:g} computational particles")
    for f in fields:
        ds[f"{f}_background"] = bg[f].assign_attrs(long_name=f"Pre-forcing time mean of {f}")
    # Domain inventory and the forcing (inject rate) as relative series
    from .convergence import inventory

    def series_fit(name, da):
        base = da.sel(time=slice(*background)).mean("time")
        rel = (da.isel(time=np.where(sel)[0]) / base - 1.0).transpose("time", "species")
        c, s = _harmonic_fit(tt, rel.values, omega)
        amp, amp_se, ph, ph_se = _amp_phase(c, s)
        ds[f"{name}_amp"] = ("species", amp, {"units": "1"})
        ds[f"{name}_amp_se"] = ("species", amp_se, {"units": "1"})
        ds[f"{name}_phase"] = ("species", ph, {"units": "rad"})
        ds[f"{name}_phase_se"] = ("species", ph_se, {"units": "rad"})

    series_fit("inventory", inventory(run).N_phys)
    if run.surface is not None:
        series_fit("inject_rate", run.surface["inject_rate"].sum("boundary_cell"))
        series_fit("return_rate", run.surface["return_rate"].sum("boundary_cell"))
        # Phase of the response relative to the forcing (inject-rate modulation)
        for f in fields:
            lag = (ds[f"{f}_phase"] - ds["inject_rate_phase"] + np.pi) % (2 * np.pi) - np.pi
            ds[f"{f}_phase_rel_forcing"] = lag.assign_attrs(units="rad", long_name="Phase relative to the inject-rate forcing")
    ds.attrs.update({k: float(v) for k, v in fw.items()})
    ds.attrs.update(background_t0=float(background[0]), background_t1=float(background[1]), run=run.name,
                    n_outputs=int(sel.sum()), min_part=float(min_part),
                    absolute_fields=" ".join(f for f in fields if f in absolute_fields))
    if nw is not None:
        ds.attrs.update(null_t0=float(nw[0]), null_t1=float(nw[1]), null_n_periods=int(nw[2]))
    return ds


def perturbation_series(run: NcRun, background: tuple[float, float], alt_km: Sequence[float],
                        field: str = "num_dens", absolute: bool = False, fit: bool = False):
    """x / x_bg - 1 (or x - x_bg if `absolute`) vs time at the cells nearest the requested
    altitudes (for plotting). With fit=True also returns the harmonic-plus-trend fit over the
    forcing window (the same model as forcing_response) as a second DataArray."""
    bg = time_mean_profiles(run, *background, fields=(field,))
    rel = _perturbation(run.macro[field], bg[field], absolute)
    alt = run.macro["altitude"].values / 1e3
    idx = [int(np.argmin(np.abs(alt - a))) for a in alt_km]
    rel = rel.isel(radial=idx).assign_coords(altitude_km=("radial", alt[idx]))
    if not fit:
        return rel
    fw = forcing_window(run)
    t = run.time
    sel = np.where((t > fw["t_start"]) & (t <= fw["t_end"] + 1e-9))[0]
    y = rel.isel(time=sel).transpose("time", ...)
    tt = t[sel]
    coef, _ = _harmonic_fit(tt, y.values, fw["omega"])
    X = np.column_stack([np.ones_like(tt), tt - tt.mean(), np.cos(fw["omega"] * tt), np.sin(fw["omega"] * tt)])
    curve = np.einsum("tk,k...->t...", X, coef)
    return rel, y.copy(data=curve)
