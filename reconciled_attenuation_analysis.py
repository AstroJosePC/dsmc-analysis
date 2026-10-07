#!/usr/bin/env python3
"""Reconciled horizontal-attenuation analysis.

Two existing pipelines measure the same physical quantity -- the horizontal decay
of the relative density perturbation outward from the forcing wedge -- and disagree
by more than an order of magnitude:

* ``compute_horizontal_wavelengths.py`` -> ``export/horizontal_results.pkl`` ->
  ``azimuthal_resolution_convergence.ipynb`` (cell 46): amplitudes up to ~2.4,
  L_att = 121.4 km at 231 km.
* ``outside_wedge_density_analyzer.py``: amplitudes within +/-0.16,
  L_att = 84.2 km at 231 km.

Root causes (all confirmed against the data):

1. Background mis-indexing in ``compute_horizontal_wavelengths.py`` (lines 463-470):
   the wedge-column pre-wave background is Pchip-resampled onto a *uniform* radial
   grid and then indexed with the *non-uniform* native grid index.  Native index 27
   ("231.4 km") reads the background at ~354 km.  The error is a pure multiplicative
   factor n0(z_true)/n0(z_used) = 17.94 / 16.29 / 9.84 at 148 / 231 / 315 km, which
   is the whole y-scale gap and which silently disables the noise debias.
2. Two different amplitude estimators (2-cycle RMS vs. single-phase tracked trough).
3. Two different L_att definitions (exponential fit vs. interpolated 1/e crossing
   vs. a bare index lookup).
4. Two different distance axes (per-altitude arc length vs. one mean cell width
   evaluated at R_M + 100 km).
5. Two different side conventions (left-only vs. left/right averaged).

This script redoes the analysis under ONE self-consistent technique and exports the
two figures that the two pipelines used to produce separately:

* Figure 1: altitude vs. distance-from-forcing-edge snapshot with the mean-free-path
  overlay (style of ``outside_wedge_density_analyzer.py --snapshot-averaged --overlay-mfp``).
* Figure 2: five stacked attenuation panels across simulations (style of
  ``azimuthal_resolution_convergence.ipynb`` cell 46).

Single shared reduction, used by both figures::

    delta[t, z, phi] = (n - n0) / n0    n0 = processor.get_background_profiles()["density"]
                                             on the NATIVE non-uniform radial grid
    A_avg[t, z, k]   = 0.5 * (delta[t, z, jL - k] + delta[t, z, jR + 1 + k])
    d[z, k]          = k * (R_M + z) * dphi
    sigma_y[z, k]    = std(A_avg[cycle 4-5, z, k])
    sigma_n[z]       = std(A_avg[quiescent, z, :])   # same pair basis as sigma_y
    A[z, k]          = sqrt(2 * (sigma_y^2 - sigma_n^2)), clipped at 0

Nothing else in the repository is modified; this script is purely additive.

Usage::

    python3 reconciled_attenuation_analysis.py --only-azi 1024 --no-plot
    python3 reconciled_attenuation_analysis.py --use-cache
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from helper_funcs import detect_wave_period
from optimized_loader import OptimizedDSMCDataProcessor

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exports"
TRANSPORT_FILE = EXPORT_DIR / "pre_wave_transport_profiles.npz"

# --- Simulation inventory -------------------------------------------------------
# Same six runs, in the same order, as azimuthal_resolution_convergence.ipynb.
AZI_CELLS = [128, 256, 384, 512, 640, 1024]
RESULTS_PATHS = [
    "I:\\ArchiveSims\\Results_20240719_003802.440",  # Azi Cells = 128
    "I:\\ArchiveSims\\Results_20240719_025013.785",  # Azi Cells = 256
    "I:\\ArchiveSims\\Results_20240719_025200.987",  # Azi Cells = 384
    # S4 is the 2024 run: same batch, build and +-dphi latitudinal band as S5/S6.  The 2025
    # re-run (Results_20250904_113551.097, fixed +-5 deg band) agrees with it within 1 sigma;
    # see compare_s4_band_thickness_diagnostics.py.  Its outputs are in exports/s4_2025_archive/.
    "I:\\ArchiveSims\\Results_20240722_034301.982",  # Azi Cells = 512
    "I:\\ArchiveSims\\Results_20240722_040346.456",  # Azi Cells = 640
    "I:\\ArchiveSims\\Results_20240722_041906.318",  # Azi Cells = 1024
]

# Cross-platform path handling: convert Windows (I:\) to macOS (/Volumes/) if needed.
# The same fallback as compute_horizontal_wavelengths.py:50-57, plus the "/Volumes/MyBook"
# mount point actually used on this machine (the bare "/Volumes/ArchiveSims" form does
# not resolve here, which is why that script only prints warnings on macOS).
_POSIX_ROOTS = ["/Volumes/", "/Volumes/MyBook/"]
for _i, _path in enumerate(RESULTS_PATHS):
    if os.path.exists(_path):
        continue
    _candidates = [_path.replace("I:\\", root).replace("\\", "/") for root in _POSIX_ROOTS]
    _resolved = next((c for c in _candidates if os.path.exists(c)), None)
    if _resolved is not None:
        RESULTS_PATHS[_i] = _resolved
    else:
        print(f"Warning: Path does not exist: {_path} or any of {_candidates}")

# The panel figure drops the 128-cell run: notebook takes indices [-5, -4, -3, -2, -1].
PANEL_INDICES = [-5, -4, -3, -2, -1]
LABELS_REMAP = {128: "S1", 256: "S2", 384: "S3", 512: "S4", 640: "S5", 1024: "S6"}
# Colors indexed over the FULL six-entry list so S2..S6 keep their original hues.
VIRIDIS_COLORS = [plt.get_cmap("viridis")(i / (len(AZI_CELLS) - 1)) for i in range(len(AZI_CELLS))]

# --- Analysis parameters (mirrors compute_horizontal_wavelengths.py) ------------
PERIOD_DETECTION_ALTITUDE_KM = 230.0
N_CYCLES_START = 3  # start at the 4th cycle (index 3)
N_CYCLES_END = 5  # end at the 5th cycle
QUIESCENT_WINDOW_S = 1000.0
CACHE_MAX_DISTANCE_KM = 800.0  # how far outward the cached reduction reaches
SNAPSHOT_TRIM_STEPS = 60  # the `end_idx -= 60` trim used by outside_wedge_density_analyzer.py
# Altitude band over which the background-index assertion is enforced.  Above ~400 km the
# pre-wave cells are sparsely populated, so the cell-mean density is a noisy reference.
BACKGROUND_CHECK_RANGE_KM = (120.0, 400.0)
BACKGROUND_CHECK_TOLERANCE = 0.25

DEFAULT_ALTITUDES_KM = [150.0, 200.0, 230.0, 280.0, 320.0]
DEFAULT_MAX_DISTANCE_KM = 320.0
DEFAULT_SNAPSHOT_X_MAX_KM = 400.0
DEFAULT_SNAPSHOT_ALT_MIN_KM = 150.0
DEFAULT_SNAPSHOT_ALT_MAX_KM = 320.0
DEFAULT_REF_AZI = 1024

# --- Snapshot styling (copied from outside_wedge_density_analyzer.py) -----------
EXOBASE_ALTITUDE_KM = 230.0
# This snapshot is panel (b) of the manuscript's Figure 4.
SNAPSHOT_PANEL_LABEL = "(b)"
SNAPSHOT_VMIN = -0.15
SNAPSHOT_VMAX = 0.15

# --- Provenance strings ---------------------------------------------------------
ESTIMATOR_PROVENANCE = (
    "2-cycle RMS with noise debias: sigma_y = std of the left/right-symmetric "
    "normalized density perturbation over the 4th-5th forcing cycle; "
    "sigma_n = std of the SAME left/right-averaged pairs over a 1000 s quiescent "
    "window ending at wave onset, pooled over all cell offsets; "
    "A = sqrt(2*(sigma_y^2 - sigma_n^2)), "
    "clipped at 0 when noise-dominated."
)
BACKGROUND_PROVENANCE = (
    "processor.get_background_profiles()['density'][alt_idx] on the NATIVE "
    "non-uniform radial grid (no Pchip resampling onto a uniform grid)."
)
DISTANCE_PROVENANCE = "d[z, k] = k * (mars_radius + radial_cells[z]) * dphi (per-altitude arc length)."
SIDE_PROVENANCE = "A_avg[t, z, k] = 0.5*(delta[t, z, jL - k] + delta[t, z, jR + 1 + k])."
LATT_PROVENANCE = (
    "Canonical L_att: weighted least-squares fit of A(d) = a*exp(-d/L) over "
    "0 <= d <= max_dist, sigma = sigma_A, absolute_sigma=True. Diagnostic L_att: "
    "linear interpolation of A(d) down to A(0)/e."
)


# ================================================================================
# Small helpers copied from outside_wedge_density_analyzer.py
# (that module parses argv at import time, so it cannot be imported here).
# ================================================================================
def configure_perturbation_colorbar(colorbar, vmin, vmax):
    """Use stable, readable ticks for relative density perturbations."""

    ticks = np.linspace(vmin, vmax, 7)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
    colorbar.set_label(r"$(n - n_0)/n_0$")


def add_mfp_overlay(ax, mfp_km, mfp_altitudes, altitudes_plot):
    """Add mean free path overlay to axis (white line under a black dashed line)."""

    mfp_interp = interp1d(mfp_altitudes, mfp_km, bounds_error=False, fill_value="extrapolate")
    mfp_at_altitudes = mfp_interp(altitudes_plot)

    ax.plot(mfp_at_altitudes, altitudes_plot, "w-", linewidth=4.0, label="MFP", zorder=10)
    ax.plot(mfp_at_altitudes, altitudes_plot, "k--", linewidth=2.0, zorder=11)


def load_transport_profile(transport_file: Path = TRANSPORT_FILE):
    """Load the pre-wave mean-free-path profile (km) and its altitude grid (km)."""

    transport = np.load(transport_file)
    return transport["mean_free_path"] / 1e3, transport["radial_cells"]


# ================================================================================
# Reduction -- one simulation at a time
# ================================================================================
def reduce_simulation(data_dir: str, n_azi: int) -> dict:
    """Load one simulation and return the small reduced arrays shared by both figures.

    The processor (and the multi-GB density cube) is dropped by the caller as soon as
    this function returns.
    """

    print(f"\n{'=' * 78}")
    print(f"Reducing {n_azi}-azimuthal-cell simulation: {data_dir}")
    print(f"{'=' * 78}")

    processor = OptimizedDSMCDataProcessor(data_dir)
    processor.load_parameters()
    processor.load_data()

    params = processor.params
    radial_cells = params.radial_cells
    time_array = params.time
    density = processor.density_data
    n_time, n_alt, n_phi = density.shape
    print(f"Density cube: {density.shape} (time, altitude, azimuth)")

    # --- Background on the NATIVE grid -----------------------------------------
    background = processor.get_background_profiles()
    background_density = background["density"]

    if background_density.shape != radial_cells.shape:
        raise ValueError(
            f"Background profile has {background_density.shape} entries but the native radial grid has "
            f"{radial_cells.shape}; the background is not on the native grid."
        )
    if n_alt != radial_cells.size:
        raise ValueError(f"Density cube has {n_alt} altitude rows but radial_cells has {radial_cells.size}.")

    # Explicit assertion that background[alt_idx] really is the background AT
    # radial_cells[alt_idx].  This is the check that fails for the Pchip-resampled
    # background used by compute_horizontal_wavelengths.py (off by a factor 10-18).
    bg_window = processor.bg_window
    native_prewave = np.mean(density[bg_window["start"] : bg_window["end"], :, :], axis=(0, 2))
    check_band = np.where(
        (radial_cells >= BACKGROUND_CHECK_RANGE_KM[0])
        & (radial_cells <= BACKGROUND_CHECK_RANGE_KM[1])
        & (native_prewave > 0)
    )[0]
    bg_ratio = background_density[check_band] / native_prewave[check_band]
    worst = int(check_band[np.argmax(np.abs(np.log(bg_ratio)))])
    print(
        f"Background-index check over {radial_cells[check_band[0]]:.1f}-{radial_cells[check_band[-1]]:.1f} km: "
        f"background[alt_idx] / mean(pre-wave density[alt_idx]) in "
        f"[{bg_ratio.min():.4f}, {bg_ratio.max():.4f}] (worst at {radial_cells[worst]:.1f} km)"
    )
    assert np.all(np.abs(bg_ratio - 1.0) < BACKGROUND_CHECK_TOLERANCE), (
        "Background profile is not aligned with the native radial grid: "
        f"background[{worst}] = {background_density[worst]:.4e} but the pre-wave density at "
        f"radial_cells[{worst}] = {radial_cells[worst]:.2f} km is {native_prewave[worst]:.4e}. "
        "This is exactly the failure mode of the Pchip-resampled background in "
        "compute_horizontal_wavelengths.py:463-470."
    )

    # --- Wedge edges ------------------------------------------------------------
    osc_cols = processor.compute_osc_cols()
    j_left, j_right = int(osc_cols[0]), int(osc_cols[-1])
    center_col = (j_left + j_right) // 2
    print(f"Forcing wedge: azimuthal columns {j_left} to {j_right} ({j_right - j_left + 1} cells)")

    # --- Azimuthal geometry -----------------------------------------------------
    azi_edges_rad = np.deg2rad(params.azi_edges)
    dphi_all = np.diff(azi_edges_rad)
    dphi = float(np.mean(dphi_all))
    if np.std(dphi_all) / dphi > 1e-6:
        raise ValueError("Azimuthal grid is not uniform; the k*(R+z)*dphi distance axis assumes uniform cells.")
    mars_radius_km = float(params.mars_radius)

    # --- 4th-5th cycle window ---------------------------------------------------
    wave_start_time = params.start_step_wave * params.timestep
    wave_end_time = params.final_step_wave * params.timestep
    wave_indices = np.where((time_array >= wave_start_time) & (time_array <= wave_end_time))[0]
    wave_times = time_array[wave_indices]

    period_alt_idx = int(np.argmin(np.abs(radial_cells - PERIOD_DETECTION_ALTITUDE_KM)))
    center_series = density[wave_indices, period_alt_idx, center_col]
    center_perturbation = (center_series - background_density[period_alt_idx]) / background_density[period_alt_idx] * 100
    period, frequency, _, _ = detect_wave_period(wave_times, center_perturbation)
    print(
        f"Wave period detected at {radial_cells[period_alt_idx]:.1f} km (index {period_alt_idx}): "
        f"{period:.1f} s ({frequency:.6f} Hz)"
    )

    cycle_start_time = wave_start_time + N_CYCLES_START * period
    cycle_end_time = min(wave_start_time + N_CYCLES_END * period, wave_end_time)
    cycle_indices = np.where((time_array >= cycle_start_time) & (time_array <= cycle_end_time))[0]
    print(
        f"Analysis window (cycles {N_CYCLES_START + 1}-{N_CYCLES_END}): "
        f"{cycle_start_time:.1f} - {cycle_end_time:.1f} s ({len(cycle_indices)} samples)"
    )

    # --- Noise floors from the quiescent pre-wave window ------------------------
    quiescent_start_time = max(float(time_array[0]), wave_start_time - QUIESCENT_WINDOW_S)
    quiescent_indices = np.where((time_array >= quiescent_start_time) & (time_array < wave_start_time))[0]
    print(
        f"Quiescent window: {quiescent_start_time:.1f} - {wave_start_time:.1f} s "
        f"({len(quiescent_indices)} samples)"
    )

    # sigma_n is measured below, on the SAME left/right-averaged pairs that sigma_y
    # is measured on.  Measuring it on single columns instead would over-subtract:
    # averaging two columns reduces the noise variance, so a single-column floor is
    # ~2x too large in variance and biases A (and hence L_att) low near the
    # detection limit.  Measuring the pairs directly also avoids assuming the two
    # columns are independent.

    # --- Snapshot time index ----------------------------------------------------
    start_idx, end_idx = processor.get_wave_time_indices()
    snapshot_idx = int(end_idx) - SNAPSHOT_TRIM_STEPS
    print(f"Snapshot time index: {snapshot_idx} (t = {time_array[snapshot_idx]:.1f} s)")

    # --- Cell offsets -----------------------------------------------------------
    # d = k * (R_M + z) * dphi; the largest k for a given distance is needed at the
    # lowest altitude in the grid, so size the cache from radial_cells.min().
    cell_width_min_km = (mars_radius_km + float(radial_cells.min())) * dphi
    k_max_distance = int(np.ceil(CACHE_MAX_DISTANCE_KM / cell_width_min_km)) + 1
    k_max_domain = min(j_left, n_phi - 1 - (j_right + 1))
    k_offsets = np.arange(0, min(k_max_distance, k_max_domain) + 1)
    print(
        f"Cell width at {radial_cells.min():.1f} km: {cell_width_min_km:.2f} km; "
        f"using {len(k_offsets)} cell offsets (k = 0..{k_offsets[-1]})"
    )

    # --- Shared reduction -------------------------------------------------------
    n_k = len(k_offsets)
    sigma_y = np.full((n_alt, n_k), np.nan)
    snapshot_avg = np.full((n_alt, n_k), np.nan)
    quiescent_pairs = np.full((n_alt, len(quiescent_indices) * n_k), np.nan)

    inv_bg = 1.0 / background_density  # (n_alt,)
    for k_idx, k in enumerate(k_offsets):
        col_left = j_left - k
        col_right = j_right + 1 + k

        left_cycle = (density[cycle_indices, :, col_left] - background_density[None, :]) * inv_bg[None, :]
        right_cycle = (density[cycle_indices, :, col_right] - background_density[None, :]) * inv_bg[None, :]
        avg_cycle = 0.5 * (left_cycle + right_cycle)
        sigma_y[:, k_idx] = np.std(avg_cycle, axis=0)

        left_snap = (density[snapshot_idx, :, col_left] - background_density) * inv_bg
        right_snap = (density[snapshot_idx, :, col_right] - background_density) * inv_bg
        snapshot_avg[:, k_idx] = 0.5 * (left_snap + right_snap)

        # Same pair, quiescent window: this is what the noise floor is measured on.
        left_quiet = (density[quiescent_indices, :, col_left] - background_density[None, :]) * inv_bg[None, :]
        right_quiet = (density[quiescent_indices, :, col_right] - background_density[None, :]) * inv_bg[None, :]
        avg_quiet = 0.5 * (left_quiet + right_quiet)
        n_q = len(quiescent_indices)
        quiescent_pairs[:, k_idx * n_q : (k_idx + 1) * n_q] = avg_quiet.T

    # Noise floor on the averaged pairs, pooled over the quiescent window and all
    # cell offsets (the pooling across offsets mirrors the original recipe's pooling
    # across azimuthal columns in compute_horizontal_wavelengths.py:359-365).
    sigma_n = np.std(quiescent_pairs, axis=1)
    print(
        f"Noise floor on averaged pairs: {len(quiescent_indices)} samples x {n_k} pairs; "
        f"sigma_n min={np.nanmin(sigma_n):.5f} max={np.nanmax(sigma_n):.5f}"
    )
    del quiescent_pairs

    distances = k_offsets[None, :] * (mars_radius_km + radial_cells[:, None]) * dphi

    reduction = {
        "n_azi": int(n_azi),
        "data_dir": str(data_dir),
        "radial_cells": radial_cells,
        "k_offsets": k_offsets,
        "distances": distances,
        "sigma_y": sigma_y,
        "sigma_n": sigma_n,
        "snapshot_avg": snapshot_avg,
        "background_density": background_density,
        "n_cycle_samples": int(len(cycle_indices)),
        "n_noise_samples": int(len(quiescent_indices) * n_k),
        "j_left": j_left,
        "j_right": j_right,
        "dphi": dphi,
        "mars_radius_km": mars_radius_km,
        "period_s": float(period),
        "frequency_hz": float(frequency),
        "cycle_start_s": float(cycle_start_time),
        "cycle_end_s": float(cycle_end_time),
        "quiescent_start_s": float(quiescent_start_time),
        "quiescent_end_s": float(wave_start_time),
        "wave_start_s": float(wave_start_time),
        "wave_end_s": float(wave_end_time),
        "snapshot_idx": int(snapshot_idx),
        "snapshot_time_s": float(time_array[snapshot_idx]),
        "wave_start_idx": int(start_idx),
        "wave_end_idx": int(end_idx),
        "n_phi": int(n_phi),
    }
    return reduction


def cache_path(n_azi: int) -> Path:
    return EXPORT_DIR / f"reconciled_attenuation_{n_azi}azi.npz"


def save_reduction(reduction: dict) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(reduction["n_azi"])
    np.savez_compressed(path, **reduction)
    print(f"Cached reduction: {path}")
    return path


def load_reduction(n_azi: int) -> dict:
    path = cache_path(n_azi)
    with np.load(path, allow_pickle=False) as data:
        reduction = {key: data[key] for key in data.files}
    for key in ("n_azi", "n_cycle_samples", "n_noise_samples", "j_left", "j_right", "snapshot_idx", "n_phi",
                "wave_start_idx", "wave_end_idx"):
        reduction[key] = int(reduction[key])
    for key in ("dphi", "mars_radius_km", "period_s", "frequency_hz", "cycle_start_s", "cycle_end_s",
                "quiescent_start_s", "quiescent_end_s", "wave_start_s", "wave_end_s", "snapshot_time_s"):
        reduction[key] = float(reduction[key])
    reduction["data_dir"] = str(reduction["data_dir"])
    print(f"Loaded cached reduction: {path}")
    return reduction


# ================================================================================
# Estimator, fits and diagnostics
# ================================================================================
def amplitude_profile(reduction: dict, alt_idx: int, max_distance_km: float):
    """Return (distances, A, sigma_A) at one altitude for d <= max_distance_km.

    A = sqrt(2 * (sigma_y^2 - sigma_n^2)), clipped at 0 when noise-dominated.

    sigma_A is propagated in the variance domain.  With N (resp. M) independent
    samples, var(sigma^2) = 2 sigma^4 / (N - 1); for V = 2 (sigma_y^2 - sigma_n^2),
    sigma_V = 2 sqrt(2 sigma_y^4/(N-1) + 2 sigma_n^4/(M-1)) and sigma_A = sigma_V /
    (2 max(A, sqrt(sigma_V))).  The regularisation keeps the error bar finite where
    the amplitude has been clipped to zero, in which case it reports half the
    detection limit.
    """

    distances = reduction["distances"][alt_idx]
    sigma_y = reduction["sigma_y"][alt_idx]
    sigma_n = float(reduction["sigma_n"][alt_idx])

    mask = distances <= max_distance_km
    distances = distances[mask]
    sigma_y = sigma_y[mask]

    variance = 2.0 * (sigma_y**2 - sigma_n**2)
    amplitude = np.sqrt(np.clip(variance, 0.0, None))

    n_cycle = reduction["n_cycle_samples"]
    n_noise = reduction["n_noise_samples"]
    sigma_variance = 2.0 * np.sqrt(2.0 * sigma_y**4 / max(n_cycle - 1, 1) + 2.0 * sigma_n**4 / max(n_noise - 1, 1))
    amplitude_reg = np.maximum(amplitude, np.sqrt(sigma_variance))
    sigma_amplitude = sigma_variance / (2.0 * amplitude_reg)

    return distances, amplitude, sigma_amplitude, sigma_n


def _exp_decay(d, a, length):
    return a * np.exp(-d / length)


def fit_attenuation_length(distances, amplitude, sigma_amplitude):
    """Canonical L_att: weighted fit of A(d) = a*exp(-d/L). Returns (L, sigma_L, r2, a)."""

    finite = np.isfinite(distances) & np.isfinite(amplitude) & np.isfinite(sigma_amplitude) & (sigma_amplitude > 0)
    if np.count_nonzero(finite) < 3 or np.max(amplitude[finite]) <= 0:
        return np.nan, np.nan, np.nan, np.nan

    d_fit = distances[finite]
    a_fit = amplitude[finite]
    s_fit = sigma_amplitude[finite]

    span = float(d_fit.max() - d_fit.min())
    p0 = (float(max(a_fit.max(), 1e-6)), max(span / 3.0, 1.0))

    popt, pcov = curve_fit(
        _exp_decay,
        d_fit,
        a_fit,
        p0=p0,
        sigma=s_fit,
        absolute_sigma=True,
        bounds=([0.0, 1.0], [np.inf, 1e5]),
        maxfev=20000,
    )
    a_hat, length = popt
    length_std = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) and pcov[1, 1] > 0 else np.nan

    fitted = _exp_decay(d_fit, a_hat, length)
    ss_res = float(np.sum((a_fit - fitted) ** 2))
    ss_tot = float(np.sum((a_fit - np.mean(a_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(length), length_std, r_squared, float(a_hat)


def crossing_attenuation_length(distances, amplitude):
    """Diagnostic L_att: linear interpolation of A(d) down to A(0)/e."""

    order = np.argsort(distances)
    d_sorted = distances[order]
    a_sorted = amplitude[order]

    if a_sorted.size < 2 or a_sorted[0] <= 0:
        return np.nan

    target = a_sorted[0] / np.e
    below = np.where(a_sorted <= target)[0]
    if below.size == 0:
        return np.nan

    i = int(below[0])
    if i == 0:
        return float(d_sorted[0])

    d1, a1 = float(d_sorted[i - 1]), float(a_sorted[i - 1])
    d2, a2 = float(d_sorted[i]), float(a_sorted[i])
    if a2 == a1:
        return d2
    t = (target - a1) / (a2 - a1)
    return float(d1 + t * (d2 - d1))


def analyze_run(reduction: dict, altitudes_km, max_distance_km: float) -> dict:
    """Compute the amplitude profile, both L_att estimates and diagnostics per altitude."""

    radial_cells = reduction["radial_cells"]
    entries = []
    for target_alt in altitudes_km:
        alt_idx = int(np.argmin(np.abs(radial_cells - target_alt)))
        distances, amplitude, sigma_amplitude, sigma_n = amplitude_profile(reduction, alt_idx, max_distance_km)
        length, length_std, r_squared, a_hat = fit_attenuation_length(distances, amplitude, sigma_amplitude)
        crossing = crossing_attenuation_length(distances, amplitude)

        entries.append(
            {
                "target_altitude_km": float(target_alt),
                "alt_idx": alt_idx,
                "altitude_km": float(radial_cells[alt_idx]),
                "distances_km": distances,
                "amplitude": amplitude,
                "sigma_amplitude": sigma_amplitude,
                "sigma_n": float(sigma_n),
                "L_att_fit_km": length,
                "L_att_fit_unc_km": length_std,
                "L_att_fit_r_squared": r_squared,
                "L_att_fit_a": a_hat,
                "L_att_crossing_km": crossing,
            }
        )
    return {"n_azi": reduction["n_azi"], "entries": entries}


# ================================================================================
# Figure 1 -- altitude vs. x-distance snapshot
# ================================================================================
def plot_snapshot(reduction: dict, args, output_path: Path):
    """Altitude vs. distance-from-forcing-edge snapshot with the mean-free-path overlay.

    Styling matches ``outside_wedge_density_analyzer.py --snapshot-averaged --overlay-mfp``.
    The only differences are that the data is the shared A_avg cube (no per-side
    interpolation onto an r100 grid) and that x is a 2-D coordinate array so the
    distance axis matches the attenuation panels row by row.
    """

    sns.set_theme(context="paper", font_scale=2.0, style="ticks")

    radial_cells = reduction["radial_cells"]
    alt_mask = (radial_cells >= args.snapshot_alt_min) & (radial_cells <= args.snapshot_alt_max)
    altitudes = radial_cells[alt_mask]
    data = reduction["snapshot_avg"][alt_mask, :]
    x_grid = reduction["distances"][alt_mask, :]

    # Trim the cell-offset axis to what is visible, keeping one column past x_max.
    visible = np.any(x_grid <= args.snapshot_x_max, axis=0)
    last_visible = int(np.max(np.where(visible)[0]))
    keep = slice(0, min(last_visible + 2, x_grid.shape[1]))
    data = data[:, keep]
    x_grid = x_grid[:, keep]
    y_grid = np.repeat(altitudes[:, None], x_grid.shape[1], axis=1)

    mfp_km, mfp_altitudes = load_transport_profile()

    fig, ax = plt.subplots(figsize=(7, 10))
    pcm = ax.pcolormesh(
        x_grid,
        y_grid,
        data,
        cmap="RdBu_r",
        shading="nearest",
        vmin=SNAPSHOT_VMIN,
        vmax=SNAPSHOT_VMAX,
    )

    add_mfp_overlay(ax, mfp_km, mfp_altitudes, altitudes)

    ax.axhline(
        EXOBASE_ALTITUDE_KM,
        color="#555555",
        linestyle=":",
        linewidth=1.8,
        alpha=0.9,
        label="Nominal exobase",
    )

    ax.text(
        0.03,
        0.985,
        SNAPSHOT_PANEL_LABEL,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        # The corner sits on the deep-blue end of the colormap.
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )

    cell_spacing = (reduction["mars_radius_km"] + float(altitudes.min())) * reduction["dphi"]
    ax.set_xlabel("Distance from forcing edge (km)", fontsize=22)
    ax.set_ylabel("Altitude (km)", fontsize=22)
    ax.set_xlim(-cell_spacing, args.snapshot_x_max)
    ax.tick_params(axis="both", labelsize=20)

    cbar = fig.colorbar(pcm, ax=ax)
    configure_perturbation_colorbar(cbar, SNAPSHOT_VMIN, SNAPSHOT_VMAX)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved snapshot figure: {output_path}")

    if args.no_plot:
        plt.close(fig)
    else:
        plt.show()

    return output_path


# ================================================================================
# Figure 2 -- five stacked attenuation panels
# ================================================================================
def _round_up_nice(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / 10**exponent
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if fraction <= candidate:
            return float(candidate * 10**exponent)
    return float(10 ** (exponent + 1))


def plot_panels(analyses: dict, reference_azi: int, args, output_path: Path):
    """Five stacked attenuation panels, styled after notebook cell 46."""

    sns.set_theme(context="paper", font_scale=1.5, style="ticks")
    plt.rcParams["figure.dpi"] = 300

    panel_azi = [AZI_CELLS[i] for i in PANEL_INDICES]
    panel_labels = [LABELS_REMAP.get(AZI_CELLS[i], f"{AZI_CELLS[i]} cells") for i in PANEL_INDICES]
    panel_colors = [VIRIDIS_COLORS[i] for i in PANEL_INDICES]

    n_panels = len(args.altitudes)
    fig, axes = plt.subplots(n_panels, 1, figsize=(4.5, 10), sharey=True, sharex=True)

    plotted_max = 0.0
    for sim_idx, (n_azi, label, color) in enumerate(zip(panel_azi, panel_labels, panel_colors)):
        analysis = analyses.get(n_azi)
        if analysis is None:
            continue
        # Reverse order so the highest altitude is on top.
        for panel_idx, entry in enumerate(analysis["entries"][::-1]):
            ax = axes[panel_idx]
            distances = entry["distances_km"]
            amplitude = entry["amplitude"]
            sigma_amplitude = entry["sigma_amplitude"]

            ax.errorbar(
                distances,
                amplitude,
                yerr=2 * sigma_amplitude,
                marker="o",
                markersize=6,
                alpha=0.6,
                color=color,
                label=label,
                lw=4,
            )
            ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.text(
                0.98,
                0.92,
                f"$r-R_M \\approx$ {round(entry['altitude_km'], -1):.0f} km",
                transform=ax.transAxes,
                fontsize=16,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.15),
            )
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=20)

            if amplitude.size:
                plotted_max = max(plotted_max, float(np.nanmax(amplitude + 2 * sigma_amplitude)))

    axes[n_panels // 2].set_ylabel("Normalized Density Amplitude", fontsize=22)
    axes[-1].set_xlabel("Distance from forcing edge (km)", fontsize=22)

    if args.ylim is not None:
        y_low, y_high = args.ylim
    else:
        y_low, y_high = 0.0, _round_up_nice(plotted_max)
    axes[-1].set_ylim(y_low, y_high)
    axes[-1].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

    for ax in axes:
        ax.tick_params(which="both", direction="out", top=False, right=False)

    # Gray dashed vline per panel at the reference run's canonical L_att.
    reference = analyses.get(reference_azi)
    if reference is not None:
        for panel_idx, entry in enumerate(reference["entries"][::-1]):
            length = entry["L_att_fit_km"]
            if np.isfinite(length):
                axes[panel_idx].axvline(length, color="gray", linestyle="--", linewidth=2, alpha=0.7)

    fig.subplots_adjust(right=1.045)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved attenuation panel figure: {output_path}")

    if args.no_plot:
        plt.close(fig)
    else:
        plt.show()

    return output_path


# ================================================================================
# Export and reporting
# ================================================================================
def export_results(analyses: dict, reductions: dict, args) -> tuple[Path, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    # A single-run verification pass must not clobber the full multi-run export.
    stem = "reconciled_attenuation_results" if args.only_azi is None else f"reconciled_attenuation_results_{args.only_azi}azi"
    npz_path = EXPORT_DIR / f"{stem}.npz"
    meta_path = EXPORT_DIR / f"{stem}_meta.json"

    arrays = {}
    meta = {
        "provenance": {
            "estimator": ESTIMATOR_PROVENANCE,
            "background": BACKGROUND_PROVENANCE,
            "distance_axis": DISTANCE_PROVENANCE,
            "side_convention": SIDE_PROVENANCE,
            "attenuation_length": LATT_PROVENANCE,
        },
        "settings": {
            "altitudes_km": list(map(float, args.altitudes)),
            "max_distance_km": float(args.max_dist),
            "snapshot_x_max_km": float(args.snapshot_x_max),
            "snapshot_alt_min_km": float(args.snapshot_alt_min),
            "snapshot_alt_max_km": float(args.snapshot_alt_max),
            "reference_azi_cells": int(args.ref_azi),
            "n_cycles_start": N_CYCLES_START,
            "n_cycles_end": N_CYCLES_END,
            "quiescent_window_s": QUIESCENT_WINDOW_S,
            "snapshot_trim_steps": SNAPSHOT_TRIM_STEPS,
        },
        "runs": {},
    }

    for n_azi, analysis in sorted(analyses.items()):
        reduction = reductions[n_azi]
        run_meta = {
            "data_dir": reduction["data_dir"],
            "label": LABELS_REMAP.get(n_azi, str(n_azi)),
            "n_azimuthal_cells": int(n_azi),
            "j_left": reduction["j_left"],
            "j_right": reduction["j_right"],
            "dphi_rad": reduction["dphi"],
            "mars_radius_km": reduction["mars_radius_km"],
            "wave_period_s": reduction["period_s"],
            "wave_frequency_hz": reduction["frequency_hz"],
            "cycle_window_s": [reduction["cycle_start_s"], reduction["cycle_end_s"]],
            "cycle_window_label": f"cycles {N_CYCLES_START + 1}-{N_CYCLES_END}",
            "n_cycle_samples": reduction["n_cycle_samples"],
            "quiescent_window_s": [reduction["quiescent_start_s"], reduction["quiescent_end_s"]],
            "n_noise_samples": reduction["n_noise_samples"],
            "snapshot_time_s": reduction["snapshot_time_s"],
            "snapshot_idx": reduction["snapshot_idx"],
            "altitudes": {},
        }

        for entry in analysis["entries"]:
            key = f"a{n_azi}_z{entry['alt_idx']}"
            arrays[f"{key}_distances_km"] = entry["distances_km"]
            arrays[f"{key}_amplitude"] = entry["amplitude"]
            arrays[f"{key}_sigma_amplitude"] = entry["sigma_amplitude"]
            run_meta["altitudes"][f"{entry['altitude_km']:.1f}"] = {
                "target_altitude_km": entry["target_altitude_km"],
                "altitude_km": entry["altitude_km"],
                "alt_idx": entry["alt_idx"],
                "sigma_n": entry["sigma_n"],
                "L_att_fit_km": _jsonable(entry["L_att_fit_km"]),
                "L_att_fit_unc_km": _jsonable(entry["L_att_fit_unc_km"]),
                "L_att_fit_r_squared": _jsonable(entry["L_att_fit_r_squared"]),
                "L_att_fit_amplitude_at_zero": _jsonable(entry["L_att_fit_a"]),
                "L_att_crossing_km": _jsonable(entry["L_att_crossing_km"]),
                "npz_keys": {
                    "distances_km": f"{key}_distances_km",
                    "amplitude": f"{key}_amplitude",
                    "sigma_amplitude": f"{key}_sigma_amplitude",
                },
            }

        meta["runs"][str(n_azi)] = run_meta

    np.savez_compressed(npz_path, **arrays)
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\nExported results: {npz_path}")
    print(f"Exported metadata: {meta_path}")
    return npz_path, meta_path


def _jsonable(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def print_reconciliation_table(analyses: dict):
    print(f"\n{'=' * 96}")
    print("RECONCILED HORIZONTAL ATTENUATION  --  one estimator, one background, one distance axis")
    print(f"{'=' * 96}")
    print(f"  Estimator : {ESTIMATOR_PROVENANCE}")
    print(f"  Background: {BACKGROUND_PROVENANCE}")
    print(f"  Distance  : {DISTANCE_PROVENANCE}")
    print(f"  Sides     : {SIDE_PROVENANCE}")
    print(f"  L_att     : {LATT_PROVENANCE}")
    print(f"{'-' * 96}")
    header = (
        f"{'Run':>5s} {'n_azi':>6s} {'alt (km)':>9s} {'A(0)':>8s} {'sigma_n':>8s} "
        f"{'L_fit (km)':>18s} {'R^2':>6s} {'L_cross (km)':>13s}"
    )
    print(header)
    print(f"{'-' * 96}")

    for n_azi, analysis in sorted(analyses.items()):
        label = LABELS_REMAP.get(n_azi, str(n_azi))
        for entry in analysis["entries"]:
            amplitude = entry["amplitude"]
            a_zero = float(amplitude[0]) if amplitude.size else np.nan
            length = entry["L_att_fit_km"]
            length_unc = entry["L_att_fit_unc_km"]
            fit_str = f"{length:8.1f} +/- {length_unc:5.1f}" if np.isfinite(length) else f"{'n/a':>18s}"
            r2_str = f"{entry['L_att_fit_r_squared']:6.3f}" if np.isfinite(entry["L_att_fit_r_squared"]) else "   n/a"
            cross = entry["L_att_crossing_km"]
            cross_str = f"{cross:13.1f}" if np.isfinite(cross) else f"{'n/a':>13s}"
            print(
                f"{label:>5s} {n_azi:6d} {entry['altitude_km']:9.1f} {a_zero:8.4f} {entry['sigma_n']:8.4f} "
                f"{fit_str} {r2_str} {cross_str}"
            )
        print(f"{'-' * 96}")


# ================================================================================
# CLI
# ================================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconciled horizontal-attenuation analysis (single self-consistent technique).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--altitudes",
        type=float,
        nargs="+",
        default=DEFAULT_ALTITUDES_KM,
        help="Target altitudes in km for the attenuation panels",
    )
    parser.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DISTANCE_KM, help="Max distance for panels/fit (km)")
    parser.add_argument(
        "--snapshot-x-max", type=float, default=DEFAULT_SNAPSHOT_X_MAX_KM, help="Snapshot x-axis extent (km)"
    )
    parser.add_argument("--snapshot-alt-min", type=float, default=DEFAULT_SNAPSHOT_ALT_MIN_KM, help="Snapshot min altitude (km)")
    parser.add_argument("--snapshot-alt-max", type=float, default=DEFAULT_SNAPSHOT_ALT_MAX_KM, help="Snapshot max altitude (km)")
    parser.add_argument("--ref-azi", type=int, default=DEFAULT_REF_AZI, help="Reference run (azimuthal cell count)")
    parser.add_argument("--only-azi", type=int, default=None, help="Process a single simulation, for verification")
    parser.add_argument(
        "--no-plot", action="store_true", help="Save the figures but do not display them interactively"
    )
    parser.add_argument("--outdir", type=Path, default=BASE_DIR / "figures", help="Directory for the output figures")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached per-run reductions when present")
    parser.add_argument("--force", action="store_true", help="Recompute reductions even if a cache exists")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Panel y-limits")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_plot:
        plt.switch_backend("Agg")

    if args.only_azi is not None:
        wanted = [args.only_azi]
    else:
        wanted = [AZI_CELLS[i] for i in PANEL_INDICES]
        if args.ref_azi not in wanted:
            wanted.append(args.ref_azi)

    reductions = {}
    for n_azi in wanted:
        if n_azi not in AZI_CELLS:
            raise ValueError(f"Unknown azimuthal cell count {n_azi}; known runs are {AZI_CELLS}")
        data_dir = RESULTS_PATHS[AZI_CELLS.index(n_azi)]

        if args.use_cache and not args.force and cache_path(n_azi).exists():
            reductions[n_azi] = load_reduction(n_azi)
            continue

        reduction = reduce_simulation(data_dir, n_azi)
        save_reduction(reduction)
        reductions[n_azi] = reduction

    analyses = {n_azi: analyze_run(red, args.altitudes, args.max_dist) for n_azi, red in reductions.items()}

    print_reconciliation_table(analyses)
    export_results(analyses, reductions, args)

    snapshot_source = args.only_azi if args.only_azi is not None else args.ref_azi
    if snapshot_source in reductions:
        plot_snapshot(
            reductions[snapshot_source],
            args,
            args.outdir / f"reconciled_altitude_distance_snapshot_{snapshot_source}azi.png",
        )
    if args.only_azi is None:
        plot_panels(analyses, args.ref_azi, args, args.outdir / "reconciled_attenuation_panels.png")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
