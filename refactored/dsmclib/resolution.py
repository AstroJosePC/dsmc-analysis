"""Azimuthal-resolution convergence of the density-amplitude growth (manuscript Fig 6, Fig 7, Table 4).

G_max per run = max over forced columns and altitudes of the LAST-cycle growth peak
(growth_peaks[:, -1]); with monotonic profiles it sits at the top cell (~320 km).
Eq. 10: G_max(N) = A - B / N, fitted with curve_fit; band = delta method on (A, B).

Two sources share one long table (one row per run, column, altitude) with the column
names of the 2024 parquet, so every function below works on either:
  * last_cycle_profiles(): the staging caches (outputs/<campaign>/cache/), by default with
                           STAGING_RECIPE (the root optimized_loader recipe, "mad_column0" /
                           "first_above", leaves particle noise of several % per altitude in S1-S3)
  * legacy_profiles():     zenodo_release/2d_growth_peak_data.parquet, frozen 2024 numbers,
                           which the current code does not reproduce (different grid/window)
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from .constants import LAMBDA_H_KM, MARS_RADIUS
from .growth import cached_growth
from .paths import LEGACY_ZENODO

CAMPAIGN = "s1_s6_azimuthal"
P26_ALIAS = "1D_5p"  # 1D radial reference run (P26), Results_20250326_133319.916
# Refactored S1-S6 growth recipe (see dsmclib.background / growth): all-column background over the
# 17 common pre-forcing outputs, base/top at the cells nearest 150 / 320 km (148.3 / 315.4 km)
STAGING_RECIPE = {"background": "azimean_preforcing", "altitude_cells": "nearest"}

# Azimuthal cells -> simulation label, as in the manuscript
LEGACY_LABELS = {128: "S1", 256: "S2", 384: "S3", 512: "S4", 640: "S5", 1024: "S6"}

GROWTH = "Density Amplitude Growth"
ALTITUDE = "Radial Distance"


def last_cycle_profiles(aliases: Optional[Sequence[str]] = None, **recipe) -> pd.DataFrame:
    """Last-cycle growth per forced column and altitude, from the staging growth caches.

    recipe: background / altitude_cells for cached_growth; default STAGING_RECIPE.
    """
    recipe = recipe or STAGING_RECIPE
    from . import registry

    if aliases is None:
        aliases = [r.alias for r in registry.campaign(CAMPAIGN)]
    frames = []
    for alias in aliases:
        n_azi = registry.run(alias).n_azi_cell
        growth_data = cached_growth(alias, **recipe)
        for azi, res in growth_data["growth_results"].items():
            if res is None:
                continue
            altitudes = res["radial_distances"][res["cell_min_idx"]: res["cell_max_idx"] + 1]
            last = np.ma.filled(np.ma.asarray(res["growth_peaks"][:, -1], dtype=float), np.nan)
            frames.append(pd.DataFrame({
                "Simulation": alias, "Azi Cells": n_azi, "Azi": f"{azi}azi", "Azi Num": int(azi),
                ALTITUDE: altitudes, GROWTH: last,
            }))
    return pd.concat(frames, ignore_index=True)


def legacy_profiles() -> pd.DataFrame:
    """The frozen 2024 parquet behind the published Fig 6/7, restricted to S1-S6 at Amp = 0.25."""
    df = pd.read_parquet(LEGACY_ZENODO / "2d_growth_peak_data.parquet")
    df = df[df["Azi Cells"].isin(LEGACY_LABELS) & (df["Amp"] == 0.25)].copy()
    df["Azi Num"] = df["Azi"].str.strip("azi").astype(int)
    df.insert(0, "Simulation", df["Azi Cells"].map(LEGACY_LABELS))
    return df


def add_distance_from_centre(profiles: pd.DataFrame, radius_km: float = MARS_RADIUS / 1e3) -> pd.DataFrame:
    """Add 'Distance from Center (km)' and 'Abs Distance (km)' (cell width at radius_km + 100 km)."""
    profiles = profiles.copy()
    dist = pd.Series(np.nan, index=profiles.index)
    for n_azi, sub in profiles.groupby("Azi Cells"):
        nums = sub["Azi Num"].unique()
        centre = (min(nums) + max(nums)) / 2
        dist[sub.index] = (sub["Azi Num"] - centre) * cell_width_km(n_azi, radius_km)
    profiles["Distance from Center (km)"] = dist
    profiles["Abs Distance (km)"] = dist.abs()
    return profiles


def cell_width_km(n_azi, radius_km: float = MARS_RADIUS / 1e3):
    """Azimuthal cell width at 100 km altitude [km] (same expression as Figure_6_and_7.py)."""
    return (radius_km + 100) * np.deg2rad(360 / n_azi)


def gmax_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """One row per run: N_phi, G_max, and where it sits (altitude, column), plus column count."""
    rows = []
    for sim, sub in profiles.groupby("Simulation", sort=False):
        i = sub[GROWTH].idxmax()
        rows.append({
            "Simulation": sim, "N_phi": int(sub["Azi Cells"].iloc[0]), "G_max": sub.loc[i, GROWTH],
            "alt_at_max_km": sub.loc[i, ALTITUDE], "azi_at_max": sub.loc[i, "Azi"],
            "n_columns": sub["Azi"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("N_phi", ignore_index=True)


def p26_profile() -> Tuple[np.ndarray, np.ndarray]:
    """(altitudes [km], last-cycle growth) of the 1D radial reference run (P26)."""
    growth_data = cached_growth(P26_ALIAS)
    (res,) = [r for r in growth_data["growth_results"].values() if r is not None]
    altitudes = res["radial_distances"][res["cell_min_idx"]: res["cell_max_idx"] + 1]
    return altitudes, np.ma.filled(np.ma.asarray(res["growth_peaks"][:, -1], dtype=float), np.nan)


def p26_reference() -> float:
    """P26 reference line: maximum last-cycle growth of the 1D radial run."""
    return float(np.nanmax(p26_profile()[1]))


def inverse_n(x, A, B):
    """Eq. 10: G_max(N) = A - B / N."""
    return A - B / x


def fit_inverse_n(n_phi, g_max) -> Tuple[np.ndarray, np.ndarray]:
    """(popt, pcov) of Eq. 10, unweighted, as in Figure_6_and_7.py."""
    return curve_fit(inverse_n, n_phi, g_max)


def band_quantile(level: float = 0.95, dist: str = "t", n_points: Optional[int] = None) -> float:
    """Two-sided quantile: Student-t with n_points - 2 dof, or normal (the published band)."""
    q = 1 - (1 - level) / 2
    if dist == "t":
        return stats.t.ppf(q, n_points - 2)
    if dist == "normal":
        return stats.norm.ppf(q)
    raise ValueError(f"dist must be 't' or 'normal', got {dist!r}")


def fit_band(x, popt, pcov, level: float = 0.95, dist: str = "t", n_points: Optional[int] = None):
    """(y_fit, y_lower, y_upper) of Eq. 10 with the delta-method band on (A, B)."""
    x = np.asarray(x, dtype=float)
    y_fit = inverse_n(x, *popt)
    perr = np.sqrt(np.diag(pcov))
    z = band_quantile(level, dist, n_points)
    # analytic propagation for y = A - B/x (root plot_fit_with_confidence)
    dy_dA = np.ones_like(x)
    dy_dB = -1 / x
    y_err = np.sqrt((dy_dA * perr[0])**2 + (dy_dB * perr[1])**2 + 2 * dy_dA * dy_dB * pcov[0, 1])
    return y_fit, y_fit - z * y_err, y_fit + z * y_err


def fit_summary(gmax: pd.DataFrame, popt, pcov, level: float = 0.95, dist: str = "t") -> Dict[str, float]:
    """A, B with standard errors and CIs, corr(A, B), residual RMS and R^2 of the fit."""
    n, g = gmax["N_phi"].to_numpy(float), gmax["G_max"].to_numpy(float)
    perr = np.sqrt(np.diag(pcov))
    z = band_quantile(level, dist, len(n))
    resid = g - inverse_n(n, *popt)
    return {
        "A": popt[0], "A_err": perr[0], "A_lo": popt[0] - z * perr[0], "A_hi": popt[0] + z * perr[0],
        "B": popt[1], "B_err": perr[1], "B_lo": popt[1] - z * perr[1], "B_hi": popt[1] + z * perr[1],
        "corr_AB": pcov[0, 1] / (perr[0] * perr[1]),
        "rms_resid": float(np.sqrt(np.mean(resid**2))),
        "r2": float(1 - np.sum(resid**2) / np.sum((g - g.mean())**2)),
        "level": level, "dist": dist, "quantile": float(z), "n_points": len(n),
    }


def resolution_table(gmax: pd.DataFrame, popt, radius_km: float = MARS_RADIUS / 1e3,
                     lambda_h_km: float = LAMBDA_H_KM) -> pd.DataFrame:
    """Table 4: G_max, cell width, cells per horizontal wavelength and offset from the asymptote A."""
    table = gmax.copy()
    table["cell_width_km"] = cell_width_km(table["N_phi"], radius_km)
    table["cells_per_lambda_h"] = lambda_h_km / table["cell_width_km"]
    table["offset_pct"] = 100 * (table["G_max"] - popt[0]) / popt[0]
    return table
