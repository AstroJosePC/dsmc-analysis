"""Symmetric lock-in reduction and plotting for horizontal source attenuation.

This ports the Figure 4c attenuation panel from
``lockin_symmetric_rederivation.py`` into the refactored analysis package. The
reduction keeps the symmetric pair geometry and lock-in estimator, while making
the background method explicit so the S1-S6 staging recipe can be used.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2dist


AZI_CELLS = (128, 256, 384, 512, 640, 1024)
LABELS_REMAP = {128: "S1", 256: "S2", 384: "S3", 512: "S4", 640: "S5", 1024: "S6"}
VIRIDIS_COLORS = [plt.get_cmap("viridis")(i / (len(AZI_CELLS) - 1)) for i in range(len(AZI_CELLS))]

PERIOD_DETECTION_ALTITUDE_KM = 230.0
N_CYCLES_START = 3
N_CYCLES_END = 5
QUIESCENT_WINDOW_S = 1000.0
CACHE_MAX_DISTANCE_KM = 800.0
DEFAULT_SNR_CUT = 3.0
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260908  # root lockin_attenuation_analysis.DEFAULT_SEED
MIN_FIT_POINTS = 4       # Table S3: fewer points above the floor do not constrain L_att
K0_FIT_START = 0


def _dominant_frequency(time: np.ndarray, signal: np.ndarray) -> float:
    """FFT dominant frequency after removing a linear trend, matching legacy detection."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.size < 3 or signal.size != time.size:
        raise ValueError("Need at least three matching time and signal samples.")
    dt = float(np.mean(np.diff(time)))
    detrended = signal - np.mean(signal)
    coeffs = np.polyfit(time, detrended, 1)
    detrended -= np.polyval(coeffs, time)
    frequencies = np.fft.rfftfreq(time.size, dt)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    positive = frequencies > 0
    if not positive.any():
        raise ValueError("Could not detect a positive forcing frequency.")
    return float(frequencies[positive][np.argmax(power[positive])])


def _oscillating_columns(run) -> np.ndarray:
    """Return forced columns using the same surface-rate criterion as the root loader."""
    if run.surface_data is None and run.escape_data is None:
        growth_peak_files = sorted(Path(run.data_dir).glob("growth_peak_data_*azi_macro_grids_data.pkl"))
        if growth_peak_files:
            with growth_peak_files[0].open("rb") as stream:
                growth_data = pickle.load(stream)
            if "azi_osci" in growth_data:
                return np.asarray(growth_data["azi_osci"], dtype=int)
        raise ValueError("No surface/escape data or archived growth-peak file supplies oscillating columns.")
    if run.surface_data is not None:
        rates = np.asarray(run.surface_data["species_data"]["inject_rate"])
        if rates.ndim == 3:
            baseline = rates[0, 0, :]
            std_over_time = np.std(rates[0] - baseline[None, :], axis=0)
        else:
            baseline = rates[0, :]
            std_over_time = np.std(rates - baseline[None, :], axis=0)
    else:
        rates = np.asarray(run.escape_data["species_data"]["escape_rate"])
        if rates.ndim == 3:
            baseline = rates[0, :, :]
            std_over_time = np.std(rates - baseline[:, None, :], axis=0)
        else:
            baseline = rates[0, :]
            std_over_time = np.std(rates - baseline[None, :], axis=0)
    columns = np.where(~np.isclose(std_over_time, 0.0, atol=1e-12))[0]
    if columns.size == 0:
        raise ValueError("No oscillating azimuthal columns found.")
    return columns


def lockin_projection(series: np.ndarray, times: np.ndarray, omega: float) -> np.ndarray:
    """Complex lock-in amplitude at angular frequency ``omega`` (time on axis 0)."""
    kernel = np.exp(-1j * omega * np.asarray(times))
    shape = (-1,) + (1,) * (series.ndim - 1)
    return 2.0 * np.sum(series * kernel.reshape(shape), axis=0) / len(times)


def compute_symmetric_lockin_reduction(
    run,
    *,
    background_method: str = "azimean_preforcing",
    cache_max_distance_km: float = CACHE_MAX_DISTANCE_KM,
) -> dict:
    """Reduce a loaded ``DSMCRun`` to symmetric lock-in amplitudes by signed distance.

    Pairing is ``j_left-k`` with ``j_right+k`` and signed distance is
    ``(k-1/2)*(R+z)*dphi``. Thus ``k=0`` ends at the physical source edge, while
    negative distances lie inside the source. The background defaults to the
    refactored S1-S6 common pre-forcing azimuthal mean.
    """
    if run.params is None or run.density_data is None:
        raise ValueError("Call run.load_parameters() and run.load_data() before reduction.")
    params = run.params
    density = np.asarray(run.density_data)
    radial = np.asarray(params.radial_cells, dtype=float)
    times = np.asarray(params.time, dtype=float)
    if density.ndim != 3 or density.shape[:2] != (times.size, radial.size):
        raise ValueError("Expected density shape (time, altitude, azimuth) aligned with params.")

    background = np.asarray(run.background(method=background_method)["density"], dtype=float)
    if background.shape != radial.shape:
        raise ValueError("Background density must align with the native radial grid.")

    osc_cols = _oscillating_columns(run)
    j_left, j_right = int(osc_cols[0]), int(osc_cols[-1])
    edges = np.deg2rad(np.asarray(params.azi_edges, dtype=float))
    dphi_all = np.diff(edges)
    dphi = float(np.mean(dphi_all))
    if dphi <= 0 or np.std(dphi_all) / dphi >= 1e-6:
        raise ValueError("The attenuation distance axis assumes a uniform azimuthal grid.")

    n_time, n_alt, n_phi = density.shape
    n_wedge = j_right - j_left + 1
    cell_min_km = (float(params.mars_radius) + float(radial.min())) * dphi
    k_out = int(np.ceil(cache_max_distance_km / cell_min_km)) + 1
    k = np.arange(-(n_wedge // 2 - 1), min(k_out, j_left, n_phi - 1 - j_right) + 1)

    wave_start = float(params.wave_start_time)
    wave_end = float(params.wave_end_time)
    wave_indices = np.where((times >= wave_start) & (times <= wave_end))[0]
    period_idx = int(np.argmin(np.abs(radial - PERIOD_DETECTION_ALTITUDE_KM)))
    center_col = (j_left + j_right) // 2
    center_pert = (density[wave_indices, period_idx, center_col] - background[period_idx]) / background[period_idx]
    frequency = _dominant_frequency(times[wave_indices], center_pert)
    omega = 2.0 * np.pi * frequency
    period = 1.0 / frequency

    cycles = []
    for cycle in range(N_CYCLES_START, N_CYCLES_END):
        start = wave_start + cycle * period
        idx = np.where((times >= start) & (times < min(start + period, wave_end)))[0]
        if idx.size == 0:
            raise ValueError(f"No samples in forcing cycle {cycle + 1}.")
        cycles.append(idx)
    quiet_all = np.where((times >= max(float(times[0]), wave_start - QUIESCENT_WINDOW_S)) & (times < wave_start))[0]
    quiet = quiet_all[times[quiet_all] >= wave_start - period]
    if quiet.size < 2:
        raise ValueError("The pre-forcing window does not contain one usable lock-in period.")

    z_cycles = np.zeros((len(cycles), n_alt, k.size), dtype=complex)
    z_quiet = np.zeros((n_alt, k.size), dtype=complex)
    inverse_background = 1.0 / background
    for i, offset in enumerate(k):
        left_col, right_col = j_left - offset, j_right + offset
        left = (density[:, :, left_col] - background[None, :]) * inverse_background[None, :]
        right = (density[:, :, right_col] - background[None, :]) * inverse_background[None, :]
        for ci, indices in enumerate(cycles):
            z_left = lockin_projection(left[indices], times[indices], omega)
            z_right = lockin_projection(right[indices], times[indices], omega)
            z_cycles[ci, :, i] = 0.5 * (z_left + z_right)
        quiet_pair = 0.5 * (left[quiet] + right[quiet])
        z_quiet[:, i] = lockin_projection(quiet_pair, times[quiet], omega)

    z_mean = z_cycles.mean(axis=0)
    exterior = k >= 1
    return {
        "n_azi": int(density.shape[2]), "radial_cells": radial, "k": k,
        "s_km": (k[None, :] - 0.5) * (float(params.mars_radius) + radial[:, None]) * dphi,
        "j_left": j_left, "j_right": j_right, "n_phi": n_phi, "n_wedge": n_wedge,
        "dphi": dphi, "mars_radius_km": float(params.mars_radius),
        "period_s": period, "frequency_hz": frequency, "omega": omega,
        "Z_sym": z_mean, "Z_sym_cycles": z_cycles, "Z_quiet": z_quiet,
        "sigma_c": np.sqrt(np.mean(np.abs(z_quiet) ** 2, axis=1)),
        "sigma_c_exterior": np.sqrt(np.mean(np.abs(z_quiet[:, exterior]) ** 2, axis=1)),
        "sigma_c_interior": np.sqrt(np.mean(np.abs(z_quiet[:, ~exterior]) ** 2, axis=1)),
        "background_method": background_method,
        "cycle_start_s": float(times[cycles[0][0]]), "cycle_end_s": float(times[cycles[-1][-1]]),
        "quiet_start_s": float(times[quiet[0]]), "quiet_end_s": float(times[quiet[-1]]),
    }


def fit_range(amplitude, distance_km, sigma_c: float, i0: int, snr_cut: float, max_distance_km: float):
    """Contiguous outward fit range above the lock-in noise threshold."""
    ok = np.isfinite(amplitude) & np.isfinite(distance_km) & (amplitude > snr_cut * sigma_c) & (distance_km <= max_distance_km)
    i_end = i0 - 1
    for i in range(i0, amplitude.size):
        if not ok[i]:
            break
        i_end = i
    return i0, i_end


def _exp_decay(d, a0, length):
    return a0 * np.exp(-d / length)


def fit_exp_at(amplitude, distance_km, sigma_c: float, i0: int, i_end: int) -> dict:
    """Exponential fit over points i0..i_end, with the goodness-of-fit diagnostics of
    root ``lockin_symmetric_rederivation.fit_exp_at`` (L, a0 plus n, R^2, chi2, p, r_0, mask)."""
    m = np.arange(i0, i_end + 1)
    x = np.asarray(distance_km, dtype=float)[m]
    y = np.asarray(amplitude, dtype=float)[m]
    n = x.size
    out = {"n": n, "s0": float(x[0]) if n else np.nan, "s_end": float(x[-1]) if n else np.nan}
    if n < 3:
        return out | {"L": np.nan, "a0": np.nan, "sigma_cov": np.nan, "r2": np.nan,
                      "chi2_dof": np.nan, "p": np.nan, "r0": np.nan, "mask": None}
    popt, pcov = curve_fit(_exp_decay, x, y, p0=(float(y.max()), 120.0),
                           sigma=np.full(n, sigma_c), absolute_sigma=True,
                           bounds=([0.0, 1.0], [np.inf, 1e5]), maxfev=40000)
    fitted = _exp_decay(x, *popt)
    resid = (y - fitted) / sigma_c
    chi2 = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    mask = np.zeros(np.asarray(amplitude).size, bool)
    mask[m] = True
    return out | {
        "L": float(popt[1]), "a0": float(popt[0]),
        "sigma_cov": float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) and pcov[1, 1] > 0 else np.nan,
        "r2": 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else np.nan,
        "chi2_dof": chi2 / (n - 2), "p": float(chi2dist.sf(chi2, n - 2)),
        "r0": float(resid[0]), "mask": mask,
    }


def bootstrap_attenuation(distance_km, amplitude, mask, n_boot: int, rng) -> float:
    """Residual bootstrap of the unweighted exponential fit; returns sigma_L.

    Copied from root ``lockin_attenuation_analysis.bootstrap_attenuation``: residuals are
    inflated by sqrt(n/(n-2)) for the two fitted parameters and resampled with replacement.
    """
    d_fit = np.asarray(distance_km)[mask]
    a_fit = np.asarray(amplitude)[mask]
    if d_fit.size < 3:
        return np.nan
    span = float(max(d_fit.max() - d_fit.min(), 1.0))
    popt, _ = curve_fit(_exp_decay, d_fit, a_fit, p0=(float(a_fit.max()), span / 2.0),
                        bounds=([0.0, 1.0], [np.inf, 1e5]), maxfev=20000)
    fitted = _exp_decay(d_fit, *popt)
    residuals = (a_fit - fitted) * np.sqrt(d_fit.size / max(d_fit.size - 2, 1))
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        synthetic = fitted + rng.choice(residuals, size=residuals.size, replace=True)
        try:
            p_boot, _ = curve_fit(_exp_decay, d_fit, synthetic, p0=popt,
                                  bounds=([0.0, 1.0], [np.inf, 1e5]), maxfev=5000)
            draws[i] = p_boot[1]
        except RuntimeError:
            continue
    return float(np.nanstd(draws, ddof=1))


def start_stability(red: dict, zi: int, snr_cut: float, max_distance_km: float) -> float:
    """L with the fit start moved one pair outward (k0 = +1), or nan if under 3 points."""
    k, A = np.asarray(red["k"]), np.abs(np.asarray(red["Z_sym"]))[zi]
    s, sc = np.asarray(red["s_km"])[zi], float(np.asarray(red["sigma_c"])[zi])
    i0 = int(np.where(k == K0_FIT_START + 1)[0][0])
    _, i_end = fit_range(A, s, sc, i0, snr_cut, max_distance_km)
    return fit_exp_at(A, s, sc, i0, i_end)["L"]


def attenuation_table(red: dict, altitudes_km: Sequence[float] = (150.0, 230.0, 320.0), *,
                      snr_cut: float = DEFAULT_SNR_CUT, max_distance_km: float = 400.0,
                      n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                      min_points: int = MIN_FIT_POINTS) -> list[dict]:
    """Table S3 rows for one reduction: L_att with the bootstrap and cycle-to-cycle terms.

    Same recipe as root ``lockin_symmetric_rederivation.analyze``: fixed start k0 = 0,
    contiguous points above snr_cut*sigma_c, a 2000-draw residual bootstrap (one rng per run,
    seeded with ``seed``, drawn in altitude order), and half the difference of the lengths
    fitted to the cycle-4 and cycle-5 moduli over the same range, combined in quadrature.
    ``L_att_km`` is nan when fewer than ``min_points`` points pass (the table's dashes).
    """
    k = np.asarray(red["k"])
    s_all, radial = np.asarray(red["s_km"]), np.asarray(red["radial_cells"])
    A, A_cyc = np.abs(np.asarray(red["Z_sym"])), np.abs(np.asarray(red["Z_sym_cycles"]))
    i0 = int(np.where(k == K0_FIT_START)[0][0])
    rng = np.random.default_rng(seed)
    rows = []
    for target in altitudes_km:
        zi = int(np.argmin(np.abs(radial - target)))
        s, a, sc = s_all[zi], A[zi], float(np.asarray(red["sigma_c"])[zi])
        _, i_end = fit_range(a, s, sc, i0, snr_cut, max_distance_km)
        fit = fit_exp_at(a, s, sc, i0, i_end)
        sigma_boot = sigma_cycle = np.nan
        L_cyc = [np.nan, np.nan]
        if fit["mask"] is not None:
            sigma_boot = bootstrap_attenuation(s, a, fit["mask"], n_boot, rng)
            L_cyc = [fit_exp_at(A_cyc[c, zi], s, sc, i0, i_end)["L"] for c in range(A_cyc.shape[0])]
            sigma_cycle = 0.5 * abs(L_cyc[0] - L_cyc[1])
        quotable = fit["n"] >= min_points
        L_plus1 = start_stability(red, zi, snr_cut, max_distance_km)
        rows.append({
            "run": str(red.get("alias", LABELS_REMAP.get(int(red["n_azi"]), red["n_azi"]))),
            "n_azi": int(red["n_azi"]), "target_altitude_km": float(target),
            "altitude_km": float(radial[zi]),
            "cell_width_km": float((float(red["mars_radius_km"]) + radial[zi]) * float(red["dphi"])),
            "L_att_km": fit["L"] if quotable else np.nan, "L_fit_km": fit["L"],
            "sigma_L_boot_km": sigma_boot, "sigma_L_cycle_km": sigma_cycle,
            "sigma_L_total_km": float(np.hypot(sigma_boot, sigma_cycle)),
            "L_cycle4_km": L_cyc[0], "L_cycle5_km": L_cyc[1],
            "n_points": fit["n"], "n_exterior": int(np.sum(fit["mask"] & (k >= 1))) if fit["mask"] is not None
            else int(max(i_end - i0, 0)),
            "r_squared": fit["r2"], "chi2_dof": fit["chi2_dof"], "p_chi2": fit["p"], "r_inner": fit["r0"],
            "a0": fit["a0"], "s_end_km": fit["s_end"], "sigma_c": sc,
            "sigma_c_interior": float(np.asarray(red["sigma_c_interior"])[zi]),
            "sigma_c_exterior": float(np.asarray(red["sigma_c_exterior"])[zi]),
            "A_k0": float(a[i0]), "A_edge_interp": float(0.5 * (a[i0] + a[i0 + 1])),
            "L_start_plus1_km": L_plus1, "start_drift_pct": 100.0 * (L_plus1 - fit["L"]) / fit["L"],
        })
    return rows


def plateau_scan(red: dict, zi: int):
    """Outermost interior boundary at which a constant still describes A_c (chi2 p > 0.05)."""
    k, A = np.asarray(red["k"]), np.abs(np.asarray(red["Z_sym"]))[zi]
    s, sc = np.asarray(red["s_km"])[zi], float(np.asarray(red["sigma_c"])[zi])
    best = None
    for ib in range(3, int(np.where(k == 1)[0][0]) + 1):
        y = A[:ib + 1]
        a0 = float(y.mean())
        if chi2dist.sf(float(np.sum((y - a0) ** 2) / sc ** 2), y.size - 1) > 0.05:
            best = {"A0": a0, "s_b": float(s[ib]), "n": y.size}
    return best


def onset_scan(red: dict, zi: int, snr_cut: float, max_distance_km: float):
    """Innermost fit start at which one exponential is adequate (p > 0.05 and |r_0| <= 2)."""
    k, A = np.asarray(red["k"]), np.abs(np.asarray(red["Z_sym"]))[zi]
    s, sc = np.asarray(red["s_km"])[zi], float(np.asarray(red["sigma_c"])[zi])
    for k0 in range(-12, 7):
        hit = np.where(k == k0)[0]
        if hit.size == 0:
            continue
        _, i_end = fit_range(A, s, sc, int(hit[0]), snr_cut, max_distance_km)
        fit = fit_exp_at(A, s, sc, int(hit[0]), i_end)
        if fit["n"] >= 4 and fit["p"] > 0.05 and abs(fit["r0"]) <= 2.0:
            return fit | {"k0": k0}
    return None


def plot_two_branch(red: dict, altitudes_km: Sequence[float] = (150.0, 230.0, 320.0), *,
                    rows: Sequence[dict] | None = None, snr_cut: float = DEFAULT_SNR_CUT,
                    max_distance_km: float = 400.0, x_limits_km: tuple[float, float] = (-700.0, 420.0)):
    """SI Figure S5: interior plateau and exterior exponential with the transition zone shaded.

    Port of root ``lockin_symmetric_rederivation.figure_two_branch``. Left column: A_c on a log
    axis, interior pairs open squares, exterior pairs filled; each branch solid over its accepted
    range and dotted where extrapolated. Right column: exterior residuals in units of sigma_c.
    The label quotes L with the bootstrap + cycle budget from ``rows`` when given.
    """
    import seaborn as sns
    sns.set_theme(context="paper", font_scale=1.25, style="ticks")
    x_lo, x_hi = x_limits_km
    k, s_all, A = np.asarray(red["k"]), np.asarray(red["s_km"]), np.abs(np.asarray(red["Z_sym"]))
    radial, sigma_c = np.asarray(red["radial_cells"]), np.asarray(red["sigma_c"])
    interior, exterior = k <= 0, k >= 1
    i0 = int(np.where(k == K0_FIT_START)[0][0])
    colours = [plt.get_cmap("plasma")(v) for v in (0.10, 0.45, 0.75)]
    fig, axes = plt.subplots(len(altitudes_km), 2, figsize=(12.5, 10.5),
                             gridspec_kw={"width_ratios": [2.3, 1.0]})
    axes = np.atleast_2d(axes)
    for row, (target, colour) in enumerate(zip(altitudes_km[::-1], colours[::-1])):
        ax, axr = axes[row, 0], axes[row, 1]
        zi = int(np.argmin(np.abs(radial - target)))
        sc, s = float(sigma_c[zi]), s_all[zi]
        _, i_end = fit_range(A[zi], s, sc, i0, snr_cut, max_distance_km)
        fit = fit_exp_at(A[zi], s, sc, i0, i_end)
        plat = plateau_scan(red, zi)
        onset = onset_scan(red, zi, snr_cut, max_distance_km)
        s_on = onset["s0"] if onset else s[i0]
        if plat is not None:
            ax.axvspan(plat["s_b"], s_on, color="0.88", zorder=0)
        ax.axvline(0.0, color="0.35", lw=1.2, zorder=1)
        ax.axhline(snr_cut * sc, color="0.45", ls="--", lw=1.2, zorder=1)
        ax.plot(s[interior], A[zi][interior], "s", ms=5.5, mfc="none", mew=1.4, color=colour, zorder=3)
        ax.plot(s[exterior], A[zi][exterior], "o", ms=5.5, color=colour, zorder=3)
        if plat is not None:
            ax.plot([x_lo, plat["s_b"]], [plat["A0"]] * 2, "-", color="k", lw=2.2, zorder=4)
            ax.plot([plat["s_b"], s_on], [plat["A0"]] * 2, ":", color="k", lw=1.5, zorder=4)
        grid = np.linspace(x_lo, s[i_end], 400)
        ax.plot(grid, _exp_decay(grid, fit["a0"], fit["L"]), ":", color="k", lw=1.5, zorder=4)
        shown = grid >= s[i0]
        ax.plot(grid[shown], _exp_decay(grid[shown], fit["a0"], fit["L"]), "-", color="k", lw=2.2, zorder=4)
        ax.plot(s[i0], A[zi][i0], "*", ms=19, color=colour, mec="k", mew=0.8, zorder=6)
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.5 * sc, 3.0 * (plat["A0"] if plat else A[zi].max()))
        ax.set_ylabel(r"$A_c = |Z|$")
        ax.grid(alpha=0.25)
        sig = fit["sigma_cov"]
        if rows is not None:
            match = [r for r in rows if r["n_azi"] == int(red["n_azi"]) and abs(r["altitude_km"] - radial[zi]) < 0.5]
            if match and np.isfinite(match[0]["sigma_L_total_km"]):
                sig = match[0]["sigma_L_total_km"]
        ax.text(0.015, 0.065, f"$r-R_M \\approx {radial[zi]:.0f}$ km   |   "
                f"$L = {fit['L']:.1f} \\pm {sig:.1f}$ km, $n = {fit['n']}$",
                transform=ax.transAxes, fontsize=11)
        ax.text(x_hi, snr_cut * sc, f"  ${snr_cut:g}\\sigma_c$", fontsize=10, color="0.35",
                va="center", ha="left", clip_on=False)
        axr.axhline(0.0, color="0.45", ls=":", lw=1.2)
        for level in (-2, 2):
            axr.axhline(level, color="0.75", ls="--", lw=1.0)
        m = np.arange(i0, i_end + 1)
        axr.plot(s[m], (A[zi][m] - _exp_decay(s[m], fit["a0"], fit["L"])) / sc, "o-", ms=5, lw=1.0, color=colour)
        axr.axvline(0.0, color="0.6", lw=1.0)
        axr.set_ylim(-4, 4)
        axr.set_ylabel(r"residual $/\,\sigma_c$")
        axr.grid(alpha=0.25)
    axes[0, 0].text(0.04, 0.12, "wedge interior", transform=axes[0, 0].transAxes, fontsize=11, color="0.35")
    axes[-1, 0].set_xlabel(r"signed distance from wedge edge, $s$ (km)")
    axes[-1, 1].set_xlabel(r"$s$ (km)")
    handles = [
        Line2D([0], [0], marker="s", color="0.35", mfc="none", mew=1.4, ls="none", ms=6, label="interior pair ($k \\leq 0$)"),
        Line2D([0], [0], marker="o", color="0.35", ls="none", ms=6, label="exterior pair ($k \\geq 1$)"),
        Line2D([0], [0], marker="*", color="0.35", mec="k", ls="none", ms=13, label="fit start, $k=0$"),
        Line2D([0], [0], color="k", lw=2.2, label="branch over its accepted range"),
        Line2D([0], [0], color="k", lw=1.5, ls=":", label="same branch, extrapolated"),
        Line2D([0], [0], color="0.88", lw=8, label="transition zone (neither model)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=10, frameon=False, columnspacing=1.4, handletextpad=0.6)
    fig.subplots_adjust(top=0.91)
    return fig


def lockin_time_windows(run, red: dict, altitude_km: float = PERIOD_DETECTION_ALTITUDE_KM,
                        distances_km: Sequence[float] = (0.0, 355.0), *,
                        background_method: str = "azimean_preforcing") -> list[dict]:
    """Pair-averaged time series in forcing cycles 4-5 at chosen distances from the edge.

    Each distance d is bracketed by the symmetric pairs k and k+1 with
    s_k <= d < s_{k+1}; their four columns are averaged, so the series is centred at
    k*cell exactly (s = 0 for k = 0, 1) and its lock-in amplitude is the mean of the two
    pairs' complex amplitudes. Uses the reduction's period, omega and cycle windows.
    """
    params, density = run.params, np.asarray(run.density_data)
    times, radial = np.asarray(params.time, dtype=float), np.asarray(params.radial_cells, dtype=float)
    background = np.asarray(run.background(method=background_method)["density"], dtype=float)
    zi = int(np.argmin(np.abs(radial - altitude_km)))
    j_left, j_right, omega, period = int(red["j_left"]), int(red["j_right"]), float(red["omega"]), float(red["period_s"])
    cell_km = (float(red["mars_radius_km"]) + radial[zi]) * float(red["dphi"])
    wave_start, wave_end = float(params.wave_start_time), float(params.wave_end_time)
    cycles = []
    for cycle in range(N_CYCLES_START, N_CYCLES_END):
        start = wave_start + cycle * period
        cycles.append(np.where((times >= start) & (times < min(start + period, wave_end)))[0])
    out = []
    for d in distances_km:
        k_lo = int(np.floor(d / cell_km + 0.5))
        cols = [j_left - k_lo, j_right + k_lo, j_left - k_lo - 1, j_right + k_lo + 1]
        delta = (density[:, zi, cols] - background[zi]) / background[zi]
        series = delta.mean(axis=1)
        z_cyc = [lockin_projection(series[idx], times[idx], omega) for idx in cycles]
        out.append({"altitude_km": float(radial[zi]), "s_km": k_lo * cell_km, "k_pairs": (k_lo, k_lo + 1),
                    "times": [times[idx] for idx in cycles], "series": [series[idx] for idx in cycles],
                    "Z": complex(np.mean(z_cyc)), "sigma_c": float(np.asarray(red["sigma_c"])[zi]),
                    "period_s": period, "omega": omega})
    return out


def plot_lockin_time_windows(windows: Sequence[dict]):
    """SI Figure S6: cycle-4/5 pair-averaged series and the cycle-averaged projection."""
    import seaborn as sns
    sns.set_theme(context="talk", font_scale=1.2, style="whitegrid")  # the published panel's look
    fig, axes = plt.subplots(len(windows), 1, figsize=(9, 6), sharex=True)
    for ax, w in zip(np.atleast_1d(axes), windows):
        for i, (t, y) in enumerate(zip(w["times"], w["series"])):
            ax.plot(t, y, "o-", ms=3, alpha=0.45, label="normalized density" if i == 0 else "_nolegend_")
            ax.plot(t, np.real(w["Z"] * np.exp(1j * w["omega"] * t)), "k-", lw=2,
                    label="forcing-frequency projection" if i == 0 else "_nolegend_")
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.14 * (yhi - ylo))
        ax.text(
            0.018, 0.92,
            f"{round(w['altitude_km'], -1):.0f} km, $s = {w['s_km']:.0f}$ km",
            transform=ax.transAxes, fontsize=15,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )
        ax.set_ylabel(r"$(n-n_0)/n_0$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10.5, loc="upper right", frameon=True, framealpha=0.9, handlelength=2.0)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_attenuation_panels(
    reductions: Mapping[int, dict],
    plot_azi: Sequence[int] = (512, 640, 1024),
    altitudes_km: Sequence[float] = (150.0, 230.0, 320.0),
    *,
    snr_cut: float = DEFAULT_SNR_CUT,
    max_distance_km: float = 400.0,
    x_limits_km: tuple[float, float] = (-70.0, 400.0),
):
    """Draw the manuscript Figure 4c signed-distance attenuation stack.

    Negative signed distance is labeled ``source interior`` and zero is marked
    ``source edge``. Returns a Figure; callers choose where to save it.
    """
    panel_azi = [int(n) for n in plot_azi if int(n) in reductions]
    if not panel_azi:
        raise ValueError("No requested azimuthal resolutions are present in reductions.")
    colors = [VIRIDIS_COLORS[AZI_CELLS.index(n)] for n in panel_azi]
    fig, axes = plt.subplots(len(altitudes_km), 1, figsize=(5.0, 10.0), sharex=True)
    axes = np.atleast_1d(axes)
    for row, target in enumerate(altitudes_km[::-1]):
        ax = axes[row]
        ax.axvspan(x_limits_km[0], 0.0, color="0.92", zorder=0)
        ax.axvline(0.0, color="0.35", lw=1.2, zorder=1)
        top = 0.0
        for n_azi, color in zip(panel_azi, colors):
            red = reductions[n_azi]
            k = np.asarray(red["k"])
            distance = np.asarray(red["s_km"])
            amplitude_cube = np.abs(red["Z_sym"])
            zi = int(np.argmin(np.abs(np.asarray(red["radial_cells"]) - target)))
            sigma = float(np.asarray(red["sigma_c"])[zi])
            i0 = int(np.where(k == K0_FIT_START)[0][0])
            visible = (distance[zi] >= x_limits_km[0]) & (distance[zi] <= x_limits_km[1]) & (k >= K0_FIT_START)
            d, a = distance[zi][visible], amplitude_cube[zi][visible]
            detected = a > snr_cut * sigma
            ax.errorbar(d[detected], a[detected], yerr=2 * sigma, marker="o", ms=6,
                        lw=1.8, alpha=0.7, elinewidth=1.2, color=color, zorder=3)
            weak = ~detected
            if weak.any():
                ax.plot(d[weak], a[weak], "o", ms=6, mfc="white", mec=color, mew=1.5, zorder=5)
            if a.size:
                top = max(top, float(np.nanmax(a)))

        ref = reductions.get(1024)
        if ref is not None:
            zi = int(np.argmin(np.abs(np.asarray(ref["radial_cells"]) - target)))
            sigma = float(np.asarray(ref["sigma_c"])[zi])
            k = np.asarray(ref["k"])
            i0 = int(np.where(k == K0_FIT_START)[0][0])
            amplitude = np.abs(np.asarray(ref["Z_sym"])[zi])
            _, i_end = fit_range(amplitude, np.asarray(ref["s_km"])[zi], sigma, i0, snr_cut, max_distance_km)
            fit = fit_exp_at(amplitude, np.asarray(ref["s_km"])[zi], sigma, i0, i_end)
            if np.isfinite(fit["L"]):
                ax.axvline(fit["L"], color="0.45", ls="--", lw=2.0, alpha=0.8, zorder=2)
            actual_altitude = float(np.asarray(ref["radial_cells"])[zi])
        else:
            actual_altitude = float(target)
        ax.set_ylim(0.0, 1.15 * top if top > 0 else 1.0)
        ax.text(0.97, 0.92, rf"$r-R_M \approx$ {round(actual_altitude, -1):.0f} km",
                transform=ax.transAxes, fontsize=15, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=15, which="both", direction="out", top=False, right=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

    axes[0].text(0.035, 0.10, "source interior", transform=axes[0].transAxes,
                 fontsize=11, color="0.35", ha="left", va="bottom")
    axes[0].text(0.0, 0.90, "source edge", transform=axes[0].get_xaxis_transform(),
                 fontsize=10, color="0.35", ha="left", va="top")
    axes[len(axes) // 2].set_ylabel(r"Coherent density amplitude  $A_c = |Z|$", fontsize=17)
    axes[-1].set_xlabel("Signed distance from source edge, $s$ (km)", fontsize=17)
    axes[-1].set_xlim(*x_limits_km)

    handles = [Line2D([0], [0], color=color, marker="o", ms=6, lw=2,
                      label=f"{LABELS_REMAP[n]} ({n} cells)") for n, color in zip(panel_azi, colors)]
    handles += [
        Line2D([0], [0], color="0.35", marker="o", ms=6, mfc="white", mew=1.5, ls="none",
               label=f"below ${snr_cut:g}\\sigma_c$ (excluded)"),
        Line2D([0], [0], color="0.45", ls="--", lw=2, label=r"$L_{\rm att}$ (S6)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=9, frameon=False, columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(top=0.90, hspace=0.10)
    return fig


# --- Route #2: Pre-forcing transport & Stokes-layer attenuation profiles ------

CHAPMAN_ENSKOG_NU_PREFACTOR = 5.0 * np.pi / 32.0   # Bird (1994, Ch. 4) hard-sphere nu / (c_bar * lambda)
VHS_OMEGA_ATOMIC_O = 0.67                          # HARRAH MOD_Species.f90 / MOD_Collisions.f90 O-O VHS exponent
VHS_SIGMA_REF_M2 = 4.5e-20                         # HARRAH O-O VHS reference cross section [m^2] at T_ref = 273 K
VHS_T_REF_K = 273.0
ATOMIC_O_MASS_AMU = 15.999
BOLTZMANN_K_JK = 1.380649e-23
AMU_KG = 1.66053906660e-27
MONATOMIC_GAMMA = 5.0 / 3.0                        # Elastic atomic O collisions (3 translational DOF)
MONATOMIC_PRANDTL = 2.0 / 3.0                      # Chapman-Enskog / Eucken monatomic Prandtl number
STOKES_DIFFUSIVITY_FACTOR = 4.0 / 3.0 + (MONATOMIC_GAMMA - 1.0) / MONATOMIC_PRANDTL  # = 7/3


def vhs_viscosity_factor(omega: float = VHS_OMEGA_ATOMIC_O) -> float:
    """Bird (1994, Eq. 3.65 / 4.64) VHS ratio of (mu / (rho * c_bar * lambda_VHS)) to (5*pi/32).

    For a Variable Hard Sphere gas with viscosity temperature index omega (mu ~ T^omega) and
    HARRAH's VHS mean free path definition lambda_VHS = c_bar / nu_coll,VHS, the Chapman-Enskog
    prefactor 5*pi/32 is multiplied by 24 / ((5 - 2*omega)*(7 - 2*omega)). For omega = 1/2
    (hard sphere) this factor is 1; for atomic O in HARRAH (omega = 0.67) it is 1.1585447,
    giving C_VHS = 15*pi / (2*(5 - 2*omega)*(7 - 2*omega)) = 0.5686993.
    """
    return 24.0 / ((5.0 - 2.0 * omega) * (7.0 - 2.0 * omega))


def compute_preforcing_transport_profiles(
    run,
    *,
    n_snapshots: int | None = None,
    vhs_omega: float = VHS_OMEGA_ATOMIC_O,
    gamma: float = MONATOMIC_GAMMA,
    prandtl: float = MONATOMIC_PRANDTL,
    reduction: Mapping[str, object] | None = None,
    snr_cut: float = DEFAULT_SNR_CUT,
    max_distance_km: float = 400.0,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compute domain-averaged pre-forcing vertical profiles of lambda(z), c_bar(z), nu(z), D(z), and delta(z).

    Uses the standard S1-S6 pre-forcing steady-state window (last ``PREFORCING_SNAPSHOTS`` = 17
    outputs strictly prior to ``wave_start_time``, averaged over all azimuthal cells and snapshots).
    """
    from .background import PREFORCING_SNAPSHOTS, forcing_start_index
    from .macro import load_field

    params = run.load_parameters()
    end_idx = forcing_start_index(params)
    if n_snapshots is None:
        n_snapshots = PREFORCING_SNAPSHOTS
    start_idx = end_idx - int(n_snapshots)
    if start_idx < 0:
        raise ValueError(f"Requested {n_snapshots} pre-forcing snapshots, but forcing starts at index {end_idx}.")
    time_slice = slice(start_idx, end_idx)
    pre_times = np.asarray(params.time[time_slice], dtype=float)
    altitude_km = np.asarray(params.radial_cells, dtype=float)

    def _slice_field(field_name: str) -> np.ndarray:
        arr = load_field(run.data_dir, field_name, run.macro, mmap_mode="r")
        if arr is None:
            raise FileNotFoundError(f"Required macroscopic field {field_name!r} not found in {run.data_dir}")
        return np.asarray(arr[time_slice], dtype=float)

    mfp_2d = _slice_field("mean_free_path")
    c_2d = _slice_field("thermal_vel")
    scale_h_2d = _slice_field("scale_height")
    kn_2d = _slice_field("knudsen")
    density_2d = _slice_field("density")
    temp_trans_2d = _slice_field("temp_trans")
    temp_kin_2d = _slice_field("temperature")
    num_part_2d = _slice_field("number_particles")

    n_azi = int(mfp_2d.shape[2])

    # Direct azimuthal + pre-forcing steady-state average across (time, azimuth)
    mean_free_path_m = mfp_2d.mean(axis=(0, 2))
    thermal_speed_ms = c_2d.mean(axis=(0, 2))
    scale_height_m = scale_h_2d.mean(axis=(0, 2))
    density_m3 = density_2d.mean(axis=(0, 2))
    temp_trans_cell_K = temp_trans_2d.mean(axis=(0, 2))
    temp_kin_cell_K = temp_kin_2d.mean(axis=(0, 2))
    mean_particles_per_cell = num_part_2d.mean(axis=(0, 2))

    # Particle-weighted & occupied-cell diagnostics (relevant above ~570 km where fine cells can have <=1 particle)
    valid_mask = (temp_trans_2d > 0.0) & (mfp_2d > 0.0) & (mfp_2d < 0.99e9)
    valid_cell_fraction = valid_mask.mean(axis=(0, 2))
    part_sum = np.sum(num_part_2d, axis=(0, 2))
    thermal_speed_pw_ms = np.where(part_sum > 0, np.sum(c_2d * num_part_2d, axis=(0, 2)) / part_sum, np.nan)
    temp_trans_pw_K = np.where(part_sum > 0, np.sum(temp_trans_2d * num_part_2d, axis=(0, 2)) / part_sum, np.nan)

    mean_free_path_valid_m = np.array([
        mfp_2d[:, i, :][valid_mask[:, i, :]].mean() if valid_mask[:, i, :].any() else np.nan
        for i in range(altitude_km.size)
    ], dtype=float)
    thermal_speed_valid_ms = np.array([
        c_2d[:, i, :][valid_mask[:, i, :]].mean() if valid_mask[:, i, :].any() else np.nan
        for i in range(altitude_km.size)
    ], dtype=float)
    scale_height_valid_m = np.array([
        scale_h_2d[:, i, :][valid_mask[:, i, :]].mean() if valid_mask[:, i, :].any() else np.nan
        for i in range(altitude_km.size)
    ], dtype=float)

    # Exact HARRAH VHS formula applied to domain-averaged density_m3 and temp_trans_pw_K (immune to empty-cell sentinels)
    m_kg = ATOMIC_O_MASS_AMU * AMU_KG
    c_mb_ms = np.sqrt(8.0 * BOLTZMANN_K_JK * temp_trans_pw_K / (np.pi * m_kg))
    v_rel_ms = np.sqrt(4.0 * BOLTZMANN_K_JK * temp_trans_pw_K / (np.pi * m_kg))
    coll_freq_s = 2.0 * density_m3 * VHS_SIGMA_REF_M2 * v_rel_ms * (VHS_T_REF_K / temp_trans_pw_K) ** (vhs_omega - 0.5)
    mean_free_path_from_bg_m = c_mb_ms / coll_freq_s

    knudsen_number = mean_free_path_m / scale_height_m
    knudsen_from_bg = mean_free_path_from_bg_m / scale_height_valid_m

    # Exobase altitude where lambda(z) == H(z) (Kn == 1) via log-linear interpolation
    log_kn = np.log(knudsen_number)
    cross_idx = int(np.where(np.diff(np.sign(log_kn)) > 0)[0][0])
    frac = -log_kn[cross_idx] / (log_kn[cross_idx + 1] - log_kn[cross_idx])
    exobase_altitude_km = float(altitude_km[cross_idx] + frac * (altitude_km[cross_idx + 1] - altitude_km[cross_idx]))

    # Period P and angular frequency omega configured in simulation parameters
    period_s = float(params.wave_period)
    omega_rad_s = float(params.omega_freq)

    # 1. Standard Chapman-Enskog prefactor: nu = (5*pi/32) * c_bar * lambda
    ce_prefactor = CHAPMAN_ENSKOG_NU_PREFACTOR
    diff_factor = 4.0 / 3.0 + (gamma - 1.0) / prandtl  # = 7/3 for monatomic O (gamma=5/3, Pr=2/3)
    nu_m2s = ce_prefactor * thermal_speed_ms * mean_free_path_m
    D_m2s = diff_factor * nu_m2s
    delta_m = np.sqrt(D_m2s * period_s / np.pi)
    delta_km = delta_m / 1.0e3

    # 2. VHS viscosity temperature-index corrected variant (omega = 0.67, Bird 1994 Eq. 3.65 / 4.64)
    vhs_corr = vhs_viscosity_factor(vhs_omega)
    vhs_prefactor = ce_prefactor * vhs_corr
    nu_vhs_m2s = vhs_prefactor * thermal_speed_ms * mean_free_path_m
    D_vhs_m2s = diff_factor * nu_vhs_m2s
    delta_vhs_km = np.sqrt(D_vhs_m2s * period_s / np.pi) / 1.0e3

    # 3. Background-reconstructed transport profiles (eliminates empty-cell effects above ~570 km)
    nu_from_bg_m2s = ce_prefactor * thermal_speed_pw_ms * mean_free_path_from_bg_m
    D_from_bg_m2s = diff_factor * nu_from_bg_m2s
    delta_from_bg_km = np.sqrt(D_from_bg_m2s * period_s / np.pi) / 1.0e3

    # Optional continuous-altitude lock-in L_att(z) profile from S6 reduction cache
    L_att_km = np.full_like(altitude_km, np.nan)
    sigma_L_total_km = np.full_like(altitude_km, np.nan)
    sigma_L_boot_km = np.full_like(altitude_km, np.nan)
    sigma_L_cycle_km = np.full_like(altitude_km, np.nan)
    L_att_n_points = np.zeros(altitude_km.size, dtype=int)
    L_att_r_squared = np.full_like(altitude_km, np.nan)
    period_fft_s = np.nan

    if reduction is not None:
        period_fft_s = float(reduction["period_s"])
        all_rows = attenuation_table(
            dict(reduction),
            altitudes_km=altitude_km,
            snr_cut=snr_cut,
            max_distance_km=max_distance_km,
            n_boot=n_boot,
            seed=seed,
        )
        for i, r in enumerate(all_rows):
            L_att_km[i] = r["L_att_km"]
            sigma_L_total_km[i] = r["sigma_L_total_km"]
            sigma_L_boot_km[i] = r["sigma_L_boot_km"]
            sigma_L_cycle_km[i] = r["sigma_L_cycle_km"]
            L_att_n_points[i] = int(r["n_points"])
            L_att_r_squared[i] = r["r_squared"]
        # Ensure exact RNG parity with Table S3 at the three benchmark altitudes (150, 230, 320 km)
        s3_rows = attenuation_table(
            dict(reduction),
            altitudes_km=(150.0, 230.0, 320.0),
            snr_cut=snr_cut,
            max_distance_km=max_distance_km,
            n_boot=n_boot,
            seed=seed,
        )
        for r in s3_rows:
            zi = int(np.argmin(np.abs(altitude_km - float(r["altitude_km"]))))
            L_att_km[zi] = r["L_att_km"]
            sigma_L_total_km[zi] = r["sigma_L_total_km"]
            sigma_L_boot_km[zi] = r["sigma_L_boot_km"]
            sigma_L_cycle_km[zi] = r["sigma_L_cycle_km"]

    return {
        "alias": str(run.alias),
        "n_azi": n_azi,
        "altitude_km": altitude_km,
        "mean_free_path_m": mean_free_path_m,
        "thermal_speed_ms": thermal_speed_ms,
        "nu_m2s": nu_m2s,
        "D_m2s": D_m2s,
        "delta_km": delta_km,
        "nu_vhs_m2s": nu_vhs_m2s,
        "D_vhs_m2s": D_vhs_m2s,
        "delta_vhs_km": delta_vhs_km,
        "scale_height_m": scale_height_m,
        "scale_height_km": scale_height_m / 1.0e3,
        "knudsen_number": knudsen_number,
        "density_m3": density_m3,
        "temp_trans_K": temp_trans_cell_K,
        "temp_kin_K": temp_kin_cell_K,
        "mean_particles_per_cell": mean_particles_per_cell,
        "valid_cell_fraction": valid_cell_fraction,
        "mean_free_path_valid_m": mean_free_path_valid_m,
        "thermal_speed_valid_ms": thermal_speed_valid_ms,
        "thermal_speed_pw_ms": thermal_speed_pw_ms,
        "temp_trans_pw_K": temp_trans_pw_K,
        "mean_free_path_from_bg_m": mean_free_path_from_bg_m,
        "knudsen_from_bg": knudsen_from_bg,
        "nu_from_bg_m2s": nu_from_bg_m2s,
        "D_from_bg_m2s": D_from_bg_m2s,
        "delta_from_bg_km": delta_from_bg_km,
        "L_att_km": L_att_km,
        "sigma_L_total_km": sigma_L_total_km,
        "sigma_L_boot_km": sigma_L_boot_km,
        "sigma_L_cycle_km": sigma_L_cycle_km,
        "L_att_n_points": L_att_n_points,
        "L_att_r_squared": L_att_r_squared,
        "ratio_L_att_to_delta": L_att_km / delta_km,
        "ratio_L_att_to_delta_vhs": L_att_km / delta_vhs_km,
        "exobase_altitude_km": exobase_altitude_km,
        "wave_start_time_s": float(params.wave_start_time),
        "wave_start_step": int(round(params.wave_start_time / params.timestep)),
        "forcing_start_index": int(end_idx),
        "preforcing_start_index": int(start_idx),
        "preforcing_end_index": int(end_idx - 1),
        "preforcing_n_snapshots": int(n_snapshots),
        "preforcing_time_start_s": float(pre_times[0]),
        "preforcing_time_end_s": float(pre_times[-1]),
        "preforcing_times_s": pre_times,
        "wave_period_s": period_s,
        "wave_period_fft_s": period_fft_s,
        "omega_freq_rad_s": omega_rad_s,
        "ce_prefactor": ce_prefactor,
        "vhs_omega": float(vhs_omega),
        "vhs_correction_factor": float(vhs_corr),
        "vhs_prefactor": float(vhs_prefactor),
        "gamma": float(gamma),
        "prandtl": float(prandtl),
        "stokes_diffusivity_factor": float(diff_factor),
    }


def plot_preforcing_transport_profiles(
    profiles: Mapping[str, object],
    table_s3_rows: Sequence[Mapping[str, object]] | None = None,
    *,
    z_max_km: float = 550.0,
):
    """Diagnostic 4-panel figure of pre-forcing lambda(z), c_bar(z), nu(z)/D(z), and delta(z) vs. L_att(z)."""
    import seaborn as sns
    sns.set_theme(context="paper", font_scale=1.2, style="ticks")

    z = np.asarray(profiles["altitude_km"], dtype=float)
    mfp_m = np.asarray(profiles["mean_free_path_m"], dtype=float)
    mfp_bg_m = np.asarray(profiles["mean_free_path_from_bg_m"], dtype=float)
    scale_h_m = np.asarray(profiles["scale_height_m"], dtype=float)
    c_ms = np.asarray(profiles["thermal_speed_ms"], dtype=float)
    c_pw_ms = np.asarray(profiles["thermal_speed_pw_ms"], dtype=float)
    nu_m2s = np.asarray(profiles["nu_m2s"], dtype=float)
    nu_vhs_m2s = np.asarray(profiles["nu_vhs_m2s"], dtype=float)
    D_m2s = np.asarray(profiles["D_m2s"], dtype=float)
    delta_km = np.asarray(profiles["delta_km"], dtype=float)
    delta_vhs_km = np.asarray(profiles["delta_vhs_km"], dtype=float)
    delta_bg_km = np.asarray(profiles["delta_from_bg_km"], dtype=float)
    L_att_km = np.asarray(profiles["L_att_km"], dtype=float)
    sigma_L_km = np.asarray(profiles["sigma_L_total_km"], dtype=float)
    z_exo = float(profiles["exobase_altitude_km"])
    period_s = float(profiles["wave_period_s"])
    t_lo = float(profiles["preforcing_time_start_s"])
    t_hi = float(profiles["preforcing_time_end_s"])
    n_snaps = int(profiles["preforcing_n_snapshots"])
    n_azi = int(profiles["n_azi"])

    mask = z <= z_max_km

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 6.2), sharey=True)
    c_main = "#1f4e78"
    c_vhs = "#2e8b57"
    c_diff = "#d95f02"
    c_latt = "#b2182b"
    c_exo = "#555555"

    for ax in axes:
        ax.axhline(z_exo, color=c_exo, ls="--", lw=1.3, alpha=0.85, zorder=2)
        ax.axhline(230.0, color="0.65", ls=":", lw=1.0, alpha=0.8, zorder=1)
        ax.axhspan(150.0, 320.0, color="#f4efe6", alpha=0.55, zorder=0)
        ax.grid(True, which="both", alpha=0.25)

    # Panel (a): Mean free path lambda(z) and scale height H(z)
    ax0 = axes[0]
    ax0.plot(mfp_m[mask] / 1e3, z[mask], "-", lw=2.3, color=c_main, label=r"$\lambda(z)$ (sampled)")
    if z_max_km > 570.0:
        ax0.plot(mfp_bg_m[mask] / 1e3, z[mask], ":", lw=1.8, color=c_main, label=r"$\lambda(z)$ (from $\bar{n}, \bar{T}$)")
    ax0.plot(scale_h_m[mask] / 1e3, z[mask], "-.", lw=1.8, color="#7570b3", label=r"Scale height $H(z)$")
    ax0.set_xscale("log")
    ax0.set_xlabel(r"Length scale, $\lambda(z),\,H(z)$ (km)")
    ax0.set_ylabel("Altitude, $z = r - R_M$ (km)")
    ax0.set_title(r"(a) Mean Free Path $\lambda(z)$", fontweight="bold")
    ax0.text(
        0.38, z_exo + 8.0,
        rf"Exobase $\lambda = H$ ({z_exo:.1f} km)",
        transform=ax0.get_yaxis_transform(), fontsize=9.2, color=c_exo, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax0.legend(loc="lower right", fontsize=9.0, frameon=True, framealpha=0.9)

    box_style = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.78",
        alpha=0.92,
    )

    # Panel (b): Mean thermal speed c_bar(z)
    ax1 = axes[1]
    ax1.plot(c_ms[mask], z[mask], "-", lw=2.3, color=c_main,
             label=r"$\bar{c}(z) = \langle |\mathbf{v} - \mathbf{c}_0| \rangle_{t,\phi}$ (cell mean)")
    if z_max_km > 570.0:
        ax1.plot(c_pw_ms[mask], z[mask], "--", lw=1.8, color=c_diff, label=r"$\bar{c}(z)$ (particle-weighted)")
    c_mb_270 = np.sqrt(8.0 * BOLTZMANN_K_JK * 270.0 / (np.pi * ATOMIC_O_MASS_AMU * AMU_KG))
    ax1.axvline(c_mb_270, color="0.5", ls=":", lw=1.3,
                label=rf"$\bar{{c}}_{{\rm MB}}(T_0) = {c_mb_270:.1f}$ m/s ($270$ K)")
    ax1.set_xlabel(r"Mean thermal speed, $\bar{c}(z)$ (m s$^{-1}$)")
    ax1.set_title(r"(b) Thermal Speed $\bar{c}(z)$", fontweight="bold")
    if z_max_km <= 560.0:
        ax1.set_xlim(591.5, 602.5)
    ax1.text(
        0.04, 0.78,
        r"$\mathbf{Formulations:}$" + "\n"
        + r"$\bar{c}(z) = \langle |\mathbf{v} - \mathbf{c}_0| \rangle_{t,\phi}$" + "\n"
        + r"$\bar{c}_{\rm MB}(T_0) = \sqrt{\frac{8 k_B T_0}{\pi m_{\rm O}}}$" + "\n"
        + r"($T_0 = 270\ \mathrm{K},\ m_{\rm O} = 16\ \mathrm{u}$)",
        transform=ax1.transAxes, fontsize=8.5, va="top", ha="left", bbox=box_style, zorder=5,
    )
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)

    # Panel (c): Kinematic viscosity nu(z) and acoustic diffusivity D(z)
    ax2 = axes[2]
    ax2.plot(nu_m2s[mask], z[mask], "-", lw=2.3, color=c_main,
             label=r"$\nu_{\rm CE} = \frac{5\pi}{32}\bar{c}\lambda$")
    ax2.plot(nu_vhs_m2s[mask], z[mask], "--", lw=1.8, color=c_vhs,
             label=r"$\nu_{\rm VHS}$ ($\omega=0.67$)")
    ax2.plot(D_m2s[mask], z[mask], "-.", lw=2.0, color=c_diff,
             label=r"$D = \frac{7}{3}\nu_{\rm CE}$ ($\gamma=\frac{5}{3}, \mathrm{Pr}=\frac{2}{3}$)")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Diffusivity, $\nu(z),\,D(z)$ (m$^2$ s$^{-1}$)")
    ax2.set_title(r"(c) Diffusivities $\nu(z)$ & $D(z)$", fontweight="bold")
    ax2.text(
        0.04, 0.96,
        r"$\mathbf{Formulations:}$" + "\n"
        + r"$\nu_{\rm CE}(z) = \frac{5\pi}{32}\,\bar{c}(z)\,\lambda(z) \approx 0.491\,\bar{c}\lambda$" + "\n"
        + r"$\nu_{\rm VHS}(z) = \frac{24}{(5-2\omega)(7-2\omega)}\,\nu_{\rm CE} \approx 1.159\,\nu_{\rm CE}$" + "\n"
        + r"$D(z) = \nu_{\rm CE}\left(\frac{4}{3} + \frac{\gamma - 1}{\mathrm{Pr}}\right) = \frac{7}{3}\,\nu_{\rm CE}$" + "\n"
        + r"($\omega = 0.67,\ \gamma = \frac{5}{3},\ \mathrm{Pr} = \frac{2}{3}$)",
        transform=ax2.transAxes, fontsize=8.2, va="top", ha="left", bbox=box_style, zorder=5,
    )
    ax2.legend(loc="lower right", fontsize=8.6, frameon=True, framealpha=0.92)

    # Panel (d): Stokes penetration length delta(z) vs measured S6 L_att(z)
    ax3 = axes[3]
    ax3.plot(delta_km[mask], z[mask], "-", lw=2.4, color=c_main,
             label=r"$\delta(z) = \sqrt{DP/\pi}$ ($5\pi/32$)")
    ax3.plot(delta_vhs_km[mask], z[mask], "--", lw=1.9, color=c_vhs,
             label=r"$\delta_{\rm VHS}(z)$ ($\omega=0.67$, $+7.6\%$)")
    if z_max_km > 570.0:
        ax3.plot(delta_bg_km[mask], z[mask], ":", lw=1.7, color=c_main,
                 label=r"$\delta(z)$ (from $\bar{n}, \bar{T}$)")

    valid_l = mask & np.isfinite(L_att_km) & (z <= 360.0)
    if valid_l.any():
        ax3.plot(L_att_km[valid_l], z[valid_l], "-", lw=1.6, color=c_latt, alpha=0.55,
                 label=r"S6 lock-in $L_{\rm att}(z)$ profile")
        ax3.fill_betweenx(
            z[valid_l],
            np.maximum(L_att_km[valid_l] - sigma_L_km[valid_l], 1.0),
            L_att_km[valid_l] + sigma_L_km[valid_l],
            color=c_latt, alpha=0.18,
        )

    if table_s3_rows is not None:
        s6_rows = [r for r in table_s3_rows if str(r.get("run", "")) == "S6" and np.isfinite(float(r["L_att_km"]))]
        if s6_rows:
            z_pts = [float(r["altitude_km"]) for r in s6_rows]
            l_pts = [float(r["L_att_km"]) for r in s6_rows]
            e_pts = [float(r["sigma_L_total_km"]) for r in s6_rows]
            ax3.errorbar(
                l_pts, z_pts, xerr=e_pts, fmt="o", ms=7.0, color=c_latt, mec="k", mew=0.8,
                elinewidth=1.8, capsize=3.5, zorder=6,
                label=r"S6 Table S3 $L_{\rm att}$ ($150, 230, 320$ km)",
            )
            offsets = [(1.55, -18.0), (1.55, -24.0), (0.09, -5.0)]
            for idx_pt, (zp, lp, ep) in enumerate(zip(z_pts, l_pts, e_pts)):
                zi = int(np.argmin(np.abs(z - zp)))
                dp = delta_km[zi]
                mult, dz_off = offsets[idx_pt] if idx_pt < len(offsets) else (1.4, -14.0)
                ax3.annotate(
                    rf"$L_{{\rm att}}={lp:.1f}\pm{ep:.1f}$ km" + "\n" + rf"($\delta={dp:.1f}$ km)",
                    xy=(lp, zp), xytext=(lp * mult, zp + dz_off),
                    fontsize=8.1, color=c_latt,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color=c_latt, lw=0.9),
                )

    ax3.set_xscale("log")
    ax3.set_xlabel(r"Lateral scale, $\delta(z),\,L_{\rm att}(z)$ (km)")
    ax3.set_title(r"(d) Stokes $\delta(z)$ vs. Measured $L_{\rm att}(z)$", fontweight="bold")
    ax3.text(
        0.04, 0.77,
        r"$\mathbf{Formulations:}$" + "\n"
        + r"$\delta(z) = \sqrt{\frac{D(z)\,P}{\pi}} = \sqrt{\frac{35}{96}\,\bar{c}\lambda P}$" + "\n"
        + r"$\delta_{\rm VHS}(z) = \sqrt{\frac{D_{\rm VHS}(z)\,P}{\pi}} \approx 1.076\,\delta$" + "\n"
        + rf"($P = 2\pi/\omega = {period_s:.1f}\ \mathrm{{s}}$)",
        transform=ax3.transAxes, fontsize=8.1, va="top", ha="left", bbox=box_style, zorder=5,
    )
    ax3.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.92)

    axes[0].set_ylim(float(z.min()), float(min(z_max_km, z.max())))
    fig.suptitle(
        rf"S6 Pre-Forcing Transport & Stokes Attenuation Profiles ($N_{{\phi}}={n_azi}$, "
        f"t = {t_lo:.1f}\u2013{t_hi:.1f} s, n = {n_snaps} snapshots, P = {period_s:.1f} s)",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig

