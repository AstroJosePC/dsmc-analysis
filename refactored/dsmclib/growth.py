"""Density-amplitude growth with altitude: per-cycle peak perturbations normalised to ~150 km.

Method (root OptimizedDSMCDataProcessor.compute_density_growth):
  1. relative density perturbation against the pre-forcing background: "mad_column0" (root,
     default), "azimean_snapshot" (2024 pipeline) or "azimean_preforcing" (refactored S1-S6
     recipe); the azimuthal means are much less noisy for 2D runs, see background.py,
  2. restrict to the forcing window + PADDING seconds,
  3. per forced azimuth column, find num_periods peaks per altitude from the cell for
     alt_range[0] to the cell for alt_range[1] (altitude_cells: "first_above" = root default,
     "nearest" = 2024 pipeline / refactored S1-S6), normalised by the base-cell
     peaks (peaks.process_growth_num_amp),
  4. growth curve = median over the last n_last_peaks cycles.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import PADDING
from .macro import load_pickle
from .paths import mirror_legacy_cache, output_dir
from .peaks import process_growth_num_amp
from .run_data import DSMCRun

# Peak-detection settings used for every density-growth result in the manuscripts
GROWTH_PEAK_KWARGS = {"prominence": 1e-1, "distance": 12}


def forced_columns(run: DSMCRun) -> np.ndarray:
    """Azimuth columns under the forcing footprint: nonzero time std of the surface injection rate.

    Uses surface data, else escape data. Runs with neither (S6 keeps only .npy fields) fall
    back to the 'azi_osci' list stored in the run folder's growth_peak_data_*azi_*.pkl files.
    """
    if run.surface_data is None and run.escape_data is None:
        growth_peak_files = sorted(run.data_dir.glob("growth_peak_data_*azi_macro_grids_data.pkl"))
        if growth_peak_files:
            growth_data = load_pickle(growth_peak_files[0])
            if "azi_osci" in growth_data:
                return np.array(growth_data["azi_osci"])
        raise ValueError(f"{run.alias}: no surface/escape data and no growth_peak_data pickle to find the forced columns")

    if run.surface_data is not None:
        rates = run.surface_data["species_data"]["inject_rate"]
        if rates.ndim == 3:
            baseline = rates[0, 0, :]
            std_over_time = np.std(rates[0] - baseline[None, :], axis=0)
        else:
            baseline = rates[0, :]
            std_over_time = np.std(rates - baseline[None, :], axis=0)
    else:
        rates = run.escape_data["species_data"]["escape_rate"]
        if rates.ndim == 3:
            baseline = rates[0, :, :]
            std_over_time = np.std(rates - baseline[:, None, :], axis=0)
        else:
            baseline = rates[0, :]
            std_over_time = np.std(rates - baseline[None, :], axis=0)

    osc_cols = np.where(~np.isclose(std_over_time, 0.0, atol=1e-12))[0]
    if len(osc_cols) == 0:
        raise ValueError(f"{run.alias}: no oscillating azimuth columns found")
    return osc_cols


def compute_density_growth(
    run: DSMCRun,
    alt_range: Tuple[float, float] = (150, 320),
    azimuthal_indices: Optional[Sequence[int]] = None,
    window_sizes: Optional[np.ndarray] = None,
    use_dynamic_peak_detection: bool = True,
    interactive: bool = False,
    background: str = "mad_column0",
    altitude_cells: str = "first_above",
) -> Dict[str, Any]:
    """Growth peak tables for each azimuth column (default: the forced columns).

    Returns {"growth_results": {azi: {...} or None}, "azimuthal_indices", "alt_range", "alias", "params",
    "background", "altitude_cells"}.
    Each column entry holds growth_peaks (masked, n_alt x num_periods), baseline_peaks,
    radial_distances, cell_min_idx and cell_max_idx. A column whose peak processing raises
    is reported and stored as None (as in the root loader).
    """
    if azimuthal_indices is None:
        try:
            azimuthal_indices = forced_columns(run)
        except ValueError:
            if run.params.n_azi_cell != 1:
                raise
            azimuthal_indices = [0]
    azimuthal_indices = np.asarray(azimuthal_indices)

    density_perturbation = run.density_perturbation(window_sizes, background=background)
    time_mask = run.forcing_time_mask(PADDING)
    cell_min_idx = run.altitude_index(alt_range[0], altitude_cells)
    cell_max_idx = run.altitude_index(alt_range[1], altitude_cells)

    growth_results = {}
    for azi_idx in azimuthal_indices:
        if azi_idx >= run.density_data.shape[-1]:
            continue
        try:
            growth_peaks, baseline_peaks = process_growth_num_amp(
                density_perturbation[..., azi_idx],
                time_mask,
                run.params.radial_cells,
                cell_min_idx,
                cell_max_idx,
                run.params.num_periods,
                use_dynamic=use_dynamic_peak_detection,
                interactive=interactive,
                **GROWTH_PEAK_KWARGS,
            )
            growth_results[int(azi_idx)] = {
                "growth_peaks": growth_peaks,
                "baseline_peaks": baseline_peaks,
                "radial_distances": run.params.radial_cells,
                "cell_min_idx": cell_min_idx,
                "cell_max_idx": cell_max_idx,
            }
        except Exception as e:
            print(f"Warning: {run.alias}: peak processing failed for azimuth column {azi_idx}: {e}")
            growth_results[int(azi_idx)] = None

    return {
        "growth_results": growth_results,
        "azimuthal_indices": azimuthal_indices,
        "alt_range": alt_range,
        "alias": run.alias,
        "params": run.params,
        "background": background,
        "altitude_cells": altitude_cells,
    }


def growth_curve(growth_data: Dict[str, Any], azimuthal_idx: int, n_last_peaks: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """(altitudes [km], median growth over the last n_last_peaks cycles) for one column."""
    results = growth_data["growth_results"].get(int(azimuthal_idx))
    if results is None:
        raise ValueError(f"No growth data for azimuthal index {azimuthal_idx}")

    growth_peaks = results["growth_peaks"]
    altitudes = results["radial_distances"][results["cell_min_idx"]: results["cell_max_idx"] + 1]
    last_peaks = growth_peaks if growth_peaks.shape[1] < n_last_peaks else growth_peaks[:, -n_last_peaks:]
    return altitudes, np.median(last_peaks, axis=1)


def growth_matrix(growth_data: Dict[str, Any], n_last_peaks: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(altitudes, azimuth columns, growth[n_alt, n_azi]) with NaN for failed columns."""
    columns = np.asarray(growth_data["azimuthal_indices"])
    altitudes = None
    curves = []
    for azi in columns:
        try:
            alt, curve = growth_curve(growth_data, azi, n_last_peaks)
            altitudes = alt
            curves.append(curve)
        except ValueError:
            curves.append(None)
    if altitudes is None:
        raise ValueError("No valid growth data for any azimuthal column")
    matrix = np.full((len(altitudes), len(columns)), np.nan)
    for j, curve in enumerate(curves):
        if curve is not None:
            matrix[:, j] = curve
    return altitudes, columns, matrix


def centre_and_edge_growth(growth_data: Dict[str, Any], n_last_peaks: int = 2) -> Dict[str, Any]:
    """Column-pair-averaged growth profiles for the two central and two edge forced columns (Fig 3a / Table 2)."""
    from .heating import centre_and_edge_columns

    pairs = centre_and_edge_columns(growth_data["azimuthal_indices"])
    out: Dict[str, Any] = {"columns": pairs, "n_last_peaks": n_last_peaks}
    for name, cols in pairs.items():
        curves = []
        for col in cols:
            altitudes, curve = growth_curve(growth_data, col, n_last_peaks=n_last_peaks)
            curves.append(np.ma.filled(np.ma.asarray(curve, dtype=float), np.nan))
        out["altitudes"] = np.asarray(altitudes, dtype=float)
        out[name] = np.mean(curves, axis=0)
    return out


def exponential_growth_slope(altitudes, growth, target_alt_km: float = 230.0) -> Dict[str, float]:
    """Log-space exponential slope (1/km), R^2, and spline-interpolated growth factor c = ln G(target_alt_km)."""
    from scipy.interpolate import UnivariateSpline

    x = np.asarray(altitudes, dtype=float)
    y = np.ma.filled(np.ma.asarray(growth, dtype=float), np.nan)
    valid = (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid], y[valid]
    if len(x_clean) < 2:
        return {"slope_km": np.nan, "r2": np.nan, "g_at_target": np.nan, "c_at_target": np.nan}

    log_y = np.log(y_clean)
    slope, intercept = np.polyfit(x_clean, log_y, 1)
    ss_res = np.sum((log_y - (slope * x_clean + intercept)) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    order = np.argsort(x_clean)
    xs, ys = x_clean[order], y_clean[order]
    _, uniq_idx = np.unique(xs, return_index=True)
    spline = UnivariateSpline(xs[uniq_idx], ys[uniq_idx], s=0.0)
    g_target = float(spline(target_alt_km))
    return {
        "slope_km": float(slope),
        "r2": r2,
        "g_at_target": g_target,
        "c_at_target": float(np.log(g_target)) if g_target > 0 else np.nan,
    }


def save_growth(growth_data: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(growth_data, f)
    if path.parent.name == "cache":
        mirror_legacy_cache(path)
    return path


def load_growth(path: Path) -> Dict[str, Any]:
    return load_pickle(Path(path))


def cache_path(campaign: str, alias: str, background: str = "mad_column0", altitude_cells: str = "first_above") -> Path:
    """outputs/<campaign>/cache/density_growth_<alias>[_<background>][_<altitude_cells>].pkl (defaults omitted)."""
    suffix = ("" if background == "mad_column0" else f"_{background}") + \
             ("" if altitude_cells == "first_above" else f"_{altitude_cells}")
    return output_dir(campaign, "cache") / f"density_growth_{alias}{suffix}.pkl"


def cached_growth(alias: str, recompute: bool = False, **kwargs) -> Dict[str, Any]:
    """Growth for a registry alias from outputs/<campaign>/cache/, computing and saving it if absent.

    kwargs go to compute_density_growth. A cached result is returned as-is, whatever kwargs
    are passed; use recompute=True after changing settings.
    """
    from . import registry

    entry = registry.run(alias)
    path = cache_path(entry.campaign, alias, kwargs.get("background", "mad_column0"),
                      kwargs.get("altitude_cells", "first_above"))
    if path.exists() and not recompute:
        return load_growth(path)
    run = DSMCRun(entry.path, alias=alias).load()
    run.release_macro()
    growth_data = compute_density_growth(run, **kwargs)
    save_growth(growth_data, path)
    print(f"Saved {path}")
    return growth_data
