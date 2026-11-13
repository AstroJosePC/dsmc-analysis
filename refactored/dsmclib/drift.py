"""Drifting-Maxwellian lower-boundary forcing (MarsAtmos VEL_WAVE_LINEAR, perturbation type 5).

    from dsmclib import drift
    from dsmclib.ncrun import open_run_nc
    run = open_run_nc(path)                       # loads the diagnostics_* groups too
    factor = drift.correct_restart_rates(run)     # analysis-tier fix of the reload_inputs restart interval
    sch = drift.schedule(run)                     # per-output steps, elapsed time, period / phase index, forcing flag
    s = drift.interval_s(run, sch, k, "H2")       # drift ratio s = w'/v_th at every step of output k

Theory. y = v_n / v_th with v_th = sqrt(2 k T0 / m). The injected (flux-weighted) normal-velocity
distribution for a drift w' = s v_th is

    f(y | s) = y exp(-(y - s)^2) / Z(s),   y > 0,
    Z(s) = [exp(-s^2) + s sqrt(pi) (1 + erf s)] / 2,    Phi(s) / Phi(0) = m(s) = 2 Z(s).

Moments use I_n(s) = int_0^inf y^n exp(-(y - s)^2) dy with I_0 = sqrt(pi)/2 (1 + erf s),
I_1 = s I_0 + exp(-s^2)/2 = Z(s) and I_{n+1} = s I_n + (n/2) I_{n-1}; E[y^k | s] = I_{k+1} / I_1.
The two tangential components are undrifted Maxwellians, so E[v_t^2] = 2 k T0 / m = v_th^2.

Within an output interval the solver samples s at every step; a histogram or moment accumulated
over steps i is the flux-weighted mixture with weights m(s_i) (dsmclib.drift.mixture_*), which is
exact for any interval length and phase bin.

Restart quirk (MarsAtmos feature/velwave-1d, 9088aab3). A restart with reload_inputs = 1
re-reads avg_step but keeps avg_time_interval = avg_step * dt from the restart file, so every
quantity the solver divides by avg_time_interval (Rates: num_part_*, *_rate, *_energy; Macro:
therm_conduct_*, avg_time_interval itself) is off by stored / (avg_steps * dt). Counts, Macro
densities/temperatures/velocities and the diagnostics groups are unaffected. correct_restart_rates
rescales the affected variables in memory (the run file stays unmodified, docs/CONVENTIONS.md §1.2).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import xarray as xr
from scipy.special import erf

SQRT_PI = math.sqrt(math.pi)
N_PHASE = 16


# ------------------------------------------------------------------ theory
def m_flux(s):
    """Flux ratio Phi(s) / Phi(0) = exp(-s^2) + s sqrt(pi) (1 + erf s)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-s * s) + s * SQRT_PI * (1.0 + erf(s))


def Z(s):
    return 0.5 * m_flux(s)


def pdf_y(y, s):
    """f(y | s) = y exp(-(y-s)^2) / Z(s) for y > 0 (0 otherwise)."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y * np.exp(-(y - s) ** 2) / Z(s), 0.0)


def cdf_y(y, s):
    """CDF of f(y | s); broadcasting over y and s."""
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    s = np.asarray(s, dtype=float)
    num = 0.5 * (np.exp(-s * s) - np.exp(-(y - s) ** 2)) + 0.5 * s * SQRT_PI * (erf(y - s) + erf(s))
    return num / Z(s)


def _I(s, nmax: int) -> np.ndarray:
    """I_0 .. I_nmax at s (array, shape (nmax+1,) + s.shape)."""
    s = np.asarray(s, dtype=float)
    out = np.empty((nmax + 1,) + s.shape)
    out[0] = 0.5 * SQRT_PI * (1.0 + erf(s))
    if nmax >= 1:
        out[1] = s * out[0] + 0.5 * np.exp(-s * s)
    for n in range(1, nmax):
        out[n + 1] = s * out[n] + 0.5 * n * out[n - 1]
    return out


def raw_moments(s, kmax: int = 4) -> np.ndarray:
    """E[y^k | s] for k = 0..kmax (shape (kmax+1,) + s.shape)."""
    I = _I(s, kmax + 1)
    return I[1:] / I[1]


def _weights(s) -> np.ndarray:
    w = m_flux(s)
    return w / w.sum()


def mixture_raw_moments(s_steps, kmax: int = 4) -> np.ndarray:
    """E[y^k] (k = 0..kmax) of the flux-weighted mixture over the steps' drift ratios."""
    s_steps = np.atleast_1d(np.asarray(s_steps, dtype=float))
    return (raw_moments(s_steps, kmax) * _weights(s_steps)).sum(axis=1)


def central_from_raw(raw: np.ndarray) -> dict:
    """mean, variance, skewness, excess kurtosis from raw moments E[y^0..4] (leading axis)."""
    m1, m2, m3, m4 = raw[1], raw[2], raw[3], raw[4]
    var = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"mean": m1, "var": var, "skew": mu3 / var ** 1.5, "exkurt": mu4 / var ** 2 - 3.0,
                "mu4": mu4}


def sample_moment_se(raw: np.ndarray, n: float) -> dict:
    """Asymptotic standard errors of the sample mean, variance, skewness and excess kurtosis of n draws,
    from the population raw moments E[y^0..8] (delta method on the raw sample moments m1..m4,
    Cov(m_j, m_k) = (E[y^(j+k)] - E[y^j] E[y^k]) / n)."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape[0] < 9:
        raise ValueError("need raw moments up to order 8")
    cov = np.array([[raw[j + k] - raw[j] * raw[k] for k in range(1, 5)] for j in range(1, 5)]) / n

    def stats4(m):
        r = np.concatenate([[1.0], m])
        c = central_from_raw(r)
        return np.array([c["mean"], c["var"], c["skew"], c["exkurt"]])

    m0 = raw[1:5]
    f0 = stats4(m0)
    J = np.zeros((4, 4))
    for k in range(4):
        h = 1e-6 * max(abs(m0[k]), 1.0)
        dm = np.zeros(4)
        dm[k] = h
        J[:, k] = (stats4(m0 + dm) - stats4(m0 - dm)) / (2 * h)
    var = np.einsum("ij,jk,ik->i", J, cov, J)
    se = np.sqrt(np.maximum(var, 0))
    return {"mean": se[0], "var": se[1], "skew": se[2], "exkurt": se[3], "value": f0}


def mixture_bin_prob(edges, s_steps) -> np.ndarray:
    """Probability of each y bin [edges[i], edges[i+1]) under the flux-weighted mixture (sums to
    the mixture mass inside the histogram range, ~1 for edges up to y = 8)."""
    s_steps = np.atleast_1d(np.asarray(s_steps, dtype=float))
    c = (cdf_y(np.asarray(edges)[:, None], s_steps[None, :]) * _weights(s_steps)[None, :]).sum(axis=1)
    return np.diff(c)


def mixture_cdf(y, s_steps) -> np.ndarray:
    s_steps = np.atleast_1d(np.asarray(s_steps, dtype=float))
    return (cdf_y(np.asarray(y)[:, None], s_steps[None, :]) * _weights(s_steps)[None, :]).sum(axis=1)


def cycle_mean_flux_ratio(s_amp: float, n: int = 4096) -> float:
    """<m(s_amp sin phi)> over one period: cycle-mean injected flux relative to s = 0."""
    phi = (np.arange(n) + 0.5) * 2 * np.pi / n
    return float(m_flux(s_amp * np.sin(phi)).mean())


# ------------------------------------------------------------------ run-file helpers
def _num(v) -> float:
    return float(str(v).split()[0])


def wave_amplitude_per_species(run) -> np.ndarray:
    """Velocity-wave amplitude w' [m/s] per species (Info 'Velocity-Wave Amplitude (per species)')."""
    raw = run.attrs.get("param_Velocity_Wave_Amplitude__per_species")
    if raw is None:
        raw = run.attrs.get("param_Velocity_Wave_Amplitude__m_s", "0")
    vals = [float(x) for x in str(raw).split()]
    n = len(run.species)
    return np.array(vals if len(vals) == n else vals[:1] * n, dtype=float)


def correct_restart_rates(run) -> float:
    """Rescale the Rates / therm_conduct variables of a reload_inputs restart in memory (see module
    docstring). Returns the factor applied (stored avg_time_interval / (avg_steps * dt)); 1.0 when the
    stored interval is already right. Idempotent: sets run.attrs['restart_rate_factor']."""
    if "restart_rate_factor" in run.attrs:
        return float(run.attrs["restart_rate_factor"])
    true_dt = float(run.attrs["avg_steps"]) * run.dt
    stored = run.macro["avg_time_interval"].values
    factor = float(np.median(stored) / true_dt)
    if not np.allclose(stored, stored[0], rtol=1e-9):
        raise ValueError(f"{run.name}: avg_time_interval varies within the run; cannot apply one factor")
    if abs(factor - 1.0) > 1e-9:
        for ds, names in ((run.surface, ("num_part_inject", "num_part_return", "num_part_reemit", "num_part_surf",
                                        "inject_rate", "return_rate", "reemit_rate", "inject_energy",
                                        "return_energy")),
                          (run.escape, ("num_part_escape", "num_part_ball", "escape_rate", "escape_energy")),
                          (run.macro, ("therm_conduct_up", "therm_conduct_down"))):
            if ds is None:
                continue
            for v in names:
                if v in ds:
                    ds[v] = (ds[v] * factor).assign_attrs(ds[v].attrs)
        run.macro["avg_time_interval"] = run.macro["avg_time_interval"] * 0 + true_dt
    run.attrs["restart_rate_factor"] = factor
    return factor


def schedule(run) -> xr.Dataset:
    """Per-output bookkeeping of a forced (or spin-up) run.

    Output k (1-based since the run's first step S0 = n_steps - n_out avg + 1) accumulates the steps
    S0 + (k-1) avg .. S0 + k avg - 1. For forced drift_mb runs S0 equals the wave start step and
    16 avg dt = P, so output k sits in forcing period (k-1) // 16 at phase bin (k-1) % 16.
    """
    avg = int(run.attrs["avg_steps"])
    n_out = run.macro.sizes["time"]
    s0 = int(run.attrs["n_steps"]) - n_out * avg + 1
    step0 = run.attrs.get("wave_start_step")
    stepf = run.attrs.get("wave_end_step")
    omega = run.wave_omega
    k = np.arange(n_out)
    first = s0 + k * avg
    out = {"step_first": ("time", first), "step_last": ("time", first + avg - 1)}
    attrs = {"avg_steps": avg, "dt": run.dt, "run_first_step": s0, "n_outputs": n_out}
    if step0 is not None and omega:
        step0, stepf = int(step0), int(stepf)
        period = 2 * np.pi / omega
        n_force = int(round((stepf - step0 - 1) * run.dt / period))
        pidx = (first - step0) // (N_PHASE * avg)
        out |= {
            "elapsed": ("time", (first + avg - 1 - step0 + 1) * run.dt, {"units": "s",
                                                                          "long_name": "Time since forcing start at the output"}),
            "period_index": ("time", pidx),
            "phase_index": ("time", ((first - step0) // avg) % N_PHASE),
            # stepf = step0 + n_force P/dt + 1: the first ring-down output still holds one step < stepf, where
            # sin(2 pi n_force) = 0, so forcing is defined by whole periods.
            "forcing": ("time", (pidx < n_force).astype(np.int8)),
        }
        attrs |= {"wave_start_step": step0, "wave_end_step": stepf, "omega": omega, "period": period,
                  "phase_locked": int(abs(N_PHASE * avg * run.dt - period) < 1e-6 * period and s0 == step0),
                  "n_force_periods": n_force}
    return xr.Dataset(out, coords={"time": run.macro["time"]}, attrs=attrs)


def step_drift(run, steps, species: Optional[str] = None) -> np.ndarray:
    """Drift velocity w'(step) [m/s] for each species (or one), exactly as the solver applies it:
    w' AMP sin(omega (step - step0) dt) for step0 <= step < stepf, else 0."""
    steps = np.asarray(steps)
    w = wave_amplitude_per_species(run) * float(run.attrs.get("wave_amplitude", 1.0))
    step0, stepf = int(run.attrs["wave_start_step"]), int(run.attrs["wave_end_step"])
    on = (steps >= step0) & (steps < stepf)
    ph = np.sin(run.wave_omega * (steps - step0) * run.dt)
    vals = np.where(on[None, :], w[:, None] * ph[None, :], 0.0)
    if species is None:
        return vals
    return vals[run.species.index(species)]


def interval_s(run, sch: xr.Dataset, k: int, species: str, v_th: float) -> np.ndarray:
    """s = w'/v_th at every step of output k (0-based)."""
    steps = np.arange(int(sch.step_first[k]), int(sch.step_last[k]) + 1)
    return step_drift(run, steps, species) / v_th


def phase_s(run, sch: xr.Dataset, phase: int, species: str, v_th: float, periods=None) -> np.ndarray:
    """Drift ratios of every step in phase bin `phase` over the given forcing periods (default all)."""
    sel = (sch.phase_index.values == phase) & (sch.forcing.values == 1)
    if periods is not None:
        sel &= np.isin(sch.period_index.values, list(periods))
    return np.concatenate([interval_s(run, sch, k, species, v_th) for k in np.where(sel)[0]])
