"""Cycle-averaged temperature perturbation and its vertical gradient (the manuscript heating metric).

Method (root OptimizedDSMCDataProcessor.compute_temperature_heating_rate +
helper_funcs.process_temperature_deviation_peaks / compute_heating_rate_slope, as used by
zenodo_release/Figure_2_left_right_panel.py for Fig 3b / Table 2):
  1. temperature deviation against a constant T0 = 270 K (no measured background, so the
     column-0 background issue of the density growth does not apply),
  2. restrict to the forcing window + PADDING seconds,
  3. per altitude, find the troughs of the deviation (peaks of -dT) and average dT between
     the last two of them: the last-cycle mean <dT>(z),
  4. cells from the first cell above 150 km to the first above 320 km (153.5-327.4 km),
  5. least-squares slope of <dT> vs altitude, in K/km and K/H0.
The root code fits the column-averaged profile (two centre or two edge columns) with
np.polyfit and quotes no uncertainty; heating_slope adds the OLS standard error and a
Student-t interval (dof = n - 2).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .constants import PADDING
from .growth import GROWTH_PEAK_KWARGS, forced_columns
from .peaks import process_amplitude
from .run_data import DSMCRun

T0_K = 270.0
# Isothermal scale height of the T23/P26 atomic-oxygen background (Figure_2_left_right_panel.py)
MOLAR_MASS_G = 16.0
GRAVITY_M_S2 = 3.51
H0_KM = 8.314 / MOLAR_MASS_G * 1e3 * T0_K / GRAVITY_M_S2 / 1e3


def process_temperature_deviation_peaks(temperature_deviation_azi, time_mask, mid_rad_km, cell_min_idx, cell_max_idx,
                                        num_periods, last_cycle_offset: int = 0, **kwargs):
    """Per-cycle trough values and the last-cycle mean of dT for one column (copy of the root helper).

    Returns (peaks (n_alt x num_periods, masked), last_cycle (n_alt, masked)). An altitude whose
    trough detection fails is masked together with all higher altitudes.
    """
    n_alt = cell_max_idx - cell_min_idx + 1
    peaks_table = np.ma.zeros((n_alt, int(num_periods)))
    last_cycle = np.ma.zeros(n_alt)
    for i in range(n_alt):
        series = temperature_deviation_azi[time_mask, cell_min_idx + i]
        peaks, peak_indices = process_amplitude(-series, 1, num_periods=num_periods,
                                                altitude=mid_rad_km[cell_min_idx + i], **kwargs)
        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            peaks_table[i:] = np.ma.masked
            last_cycle[i:] = np.ma.masked
            print(f"Temperature peak detection failed at {mid_rad_km[cell_min_idx + i]:.1f} km; "
                  f"masking this and higher altitudes.")
            break
        peaks_table[i] = -peaks
        if len(peak_indices) >= 2:
            last_cycle[i] = series[peak_indices[-2 - last_cycle_offset]:peak_indices[-1 - last_cycle_offset]].mean()
        else:
            last_cycle[i] = np.ma.masked
    return peaks_table, last_cycle


def compute_temperature_last_cycle(
    run: DSMCRun,
    alt_range: Tuple[float, float] = (150, 320),
    azimuthal_indices: Optional[Sequence[int]] = None,
    altitude_cells: str = "first_above",
    t0: float = T0_K,
    use_dynamic_peak_detection: bool = True,
) -> Dict[str, Any]:
    """Last-cycle mean dT profiles per column (default: the forced columns).

    Returns {"temp_last_cycle": {azi: masked (n_alt,)}, "temp_peaks": {...}, "altitudes",
    "cell_min_idx", "cell_max_idx", "azimuthal_indices", "alias", "t0"}.
    """
    if azimuthal_indices is None:
        azimuthal_indices = [0] if run.params.n_azi_cell == 1 else forced_columns(run)
    azimuthal_indices = np.asarray(azimuthal_indices)
    deviation = run.temperature_data - t0
    time_mask = run.forcing_time_mask(PADDING)
    c0 = run.altitude_index(alt_range[0], altitude_cells)
    c1 = run.altitude_index(alt_range[1], altitude_cells)

    last_cycle, peaks = {}, {}
    for azi in azimuthal_indices:
        if azi >= deviation.shape[-1]:
            continue
        peaks[int(azi)], last_cycle[int(azi)] = process_temperature_deviation_peaks(
            deviation[..., azi], time_mask, run.params.radial_cells, c0, c1, run.params.num_periods,
            use_dynamic=use_dynamic_peak_detection, **GROWTH_PEAK_KWARGS)
    return {
        "temp_last_cycle": last_cycle,
        "temp_peaks": peaks,
        "altitudes": np.asarray(run.params.radial_cells[c0:c1 + 1]),
        "cell_min_idx": c0,
        "cell_max_idx": c1,
        "azimuthal_indices": azimuthal_indices,
        "alias": run.alias,
        "t0": t0,
    }


def centre_and_edge_columns(columns: Sequence[int]) -> Dict[str, list]:
    """The two central forced columns and the innermost forced column at each edge (Table 2 / Fig 3)."""
    columns = list(columns)
    mid = len(columns) // 2
    return {"center": [int(columns[mid - 1]), int(columns[mid])], "edge": [int(columns[0]), int(columns[-1])]}


def heating_slope(altitudes, temperature, h_km: float = H0_KM, level: float = 0.95) -> Dict[str, float]:
    """OLS slope of <dT> vs altitude with its standard error and Student-t interval (dof = n - 2).

    Masked/NaN points are dropped. Slopes in K/km and in K/H0 (= K/km * h_km).
    """
    t = np.ma.filled(np.ma.asarray(temperature, dtype=float), np.nan)
    z = np.asarray(altitudes, dtype=float)
    ok = np.isfinite(t)
    fit = stats.linregress(z[ok], t[ok])
    n = int(ok.sum())
    q = stats.t.ppf(0.5 + level / 2, n - 2)
    return {
        "n": n, "z_min": float(z[ok].min()), "z_max": float(z[ok].max()),
        "slope_K_km": fit.slope, "intercept_K": fit.intercept, "se_K_km": fit.stderr,
        "slope_K_H": fit.slope * h_km, "se_K_H": fit.stderr * h_km,
        "ci_half_K_H": q * fit.stderr * h_km, "t_quantile": q, "r2": fit.rvalue ** 2, "level": level,
    }


def group_heating_slopes(temp: Dict[str, Any], h_km: float = H0_KM, level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """heating_slope of the column-averaged profile for the centre pair and the edge pair (root recipe)."""
    out = {}
    for name, cols in centre_and_edge_columns(temp["azimuthal_indices"]).items():
        profile = np.ma.mean(np.ma.vstack([temp["temp_last_cycle"][c] for c in cols]), axis=0)
        out[name] = {"columns": cols, **heating_slope(temp["altitudes"], profile, h_km, level)}
    return out


def cache_path(campaign: str, alias: str) -> "Path":
    """outputs/<campaign>/cache/temperature_last_cycle_<alias>.pkl"""
    from .paths import output_dir

    return output_dir(campaign, "cache") / f"temperature_last_cycle_{alias}.pkl"


def cached_temperature_last_cycle(alias: str, recompute: bool = False, **kwargs) -> Dict[str, Any]:
    """compute_temperature_last_cycle for a registry alias, cached in outputs/<campaign>/cache/.

    A cached result is returned as-is, whatever kwargs are passed; use recompute=True after
    changing settings. Computing loads the run's fields (GB-scale for S4-S6).
    """
    import pickle

    from . import registry

    entry = registry.run(alias)
    path = cache_path(entry.campaign, alias)
    if path.exists() and not recompute:
        with open(path, "rb") as f:
            return pickle.load(f)
    run = DSMCRun(entry.path, alias=alias).load()
    run.release_macro()
    temp = compute_temperature_last_cycle(run, **kwargs)
    with open(path, "wb") as f:
        pickle.dump(temp, f)
    from .paths import mirror_legacy_cache

    mirror_legacy_cache(path)
    print(f"Saved {path}")
    return temp
