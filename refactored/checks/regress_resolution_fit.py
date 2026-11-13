"""Regression: dsmclib.resolution vs the frozen manuscript Fig 7 / Table 4 (zenodo_release/Figure_6_and_7.py).

    python checks/regress_resolution_fit.py

Figure_6_and_7.py is NOT imported or run (it saves Figure_7.png and the Table 4 CSV into
zenodo_release/ at import). Its arithmetic is copied verbatim below as the reference:
  * legacy G_max per N_phi == the root parquet groupby max, exactly
  * legacy Eq. 10 popt/pcov == root curve_fit call, exactly
  * legacy 95% band (normal z) == root plot_fit_with_confidence arithmetic, exactly
  * legacy Table 4 == zenodo_release/Table_4_resolution_audit.csv to its 4 decimals
  * P26 line == max of growth_peaks_std[:, -1] in 2d_spherical_growth_profiles.pkl, and the
    staging 1D_5p last-cycle profile == spherical_1d_data.csv
  * staging G_max table == an independent recomputation from the raw cache dicts
Report: outputs/checks/tables/regress_resolution_fit.json (+ a summary on stdout).
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
import scipy as scp
from scipy.optimize import curve_fit

from dsmclib import registry
from dsmclib import resolution as R
from dsmclib.constants import LEGACY_RADIUS_KM
from dsmclib.growth import cache_path, load_growth
from dsmclib.paths import LEGACY_ZENODO, output_dir

TABLE4_DECIMALS = 4


def root_reference():
    """Verbatim arithmetic of Figure_6_and_7.py (Figure 7 part), minus the plotting."""
    df = pd.read_parquet(LEGACY_ZENODO / "2d_growth_peak_data.parquet")
    azi_cells_to_sim = {128: 'S1', 256: 'S2', 384: 'S3', 512: 'S4', 640: 'S5', 1024: 'S6'}
    _df = df[df['Azi Cells'].isin(azi_cells_to_sim) & (df['Amp'] == 0.25)]
    df_filtered_grouped = _df.groupby('Azi Cells').agg({'Density Amplitude Growth': 'max'}).reset_index()

    def func(x, A, B):
        return A - B / x

    popt, pcov = curve_fit(func, df_filtered_grouped['Azi Cells'], df_filtered_grouped['Density Amplitude Growth'])
    x = np.linspace(df_filtered_grouped['Azi Cells'].min(), df_filtered_grouped['Azi Cells'].max(), 200)

    # plot_fit_with_confidence(ax, x_fit, popt, pcov, func, confidence=0.95)
    confidence = 0.95
    y_fit = func(x, *popt)
    perr = np.sqrt(np.diag(pcov))
    z = scp.stats.norm.ppf(1 - (1 - confidence) / 2)
    dy_dA = np.ones_like(x)
    dy_dB = -1 / x
    y_err = np.sqrt((dy_dA * perr[0])**2 + (dy_dB * perr[1])**2 + 2 * dy_dA * dy_dB * pcov[0, 1])
    return df_filtered_grouped, popt, pcov, x, (y_fit, y_fit - z * y_err, y_fit + z * y_err)


def check_legacy(report):
    ref_grouped, ref_popt, ref_pcov, x_fit, ref_band = root_reference()

    gmax = R.gmax_table(R.legacy_profiles())
    popt, pcov = R.fit_inverse_n(gmax["N_phi"], gmax["G_max"])
    band = R.fit_band(x_fit, popt, pcov, dist="normal", n_points=len(gmax))
    table = R.resolution_table(gmax, popt, radius_km=LEGACY_RADIUS_KM)

    published = pd.read_csv(LEGACY_ZENODO / "Table_4_resolution_audit.csv")
    cols = ["G_max", "cell_width_km", "cells_per_lambda_h", "offset_pct"]
    table4_diff = {c: float(np.max(np.abs(table[c].to_numpy() - published[c].to_numpy()))) for c in cols}

    report["legacy"] = {
        "n_phi_equal": bool(np.array_equal(gmax["N_phi"], ref_grouped["Azi Cells"])),
        "gmax_identical": bool(np.array_equal(gmax["G_max"].to_numpy(), ref_grouped["Density Amplitude Growth"].to_numpy())),
        "popt_identical": bool(np.array_equal(popt, ref_popt)),
        "pcov_identical": bool(np.array_equal(pcov, ref_pcov)),
        "band_identical": bool(all(np.array_equal(a, b) for a, b in zip(band, ref_band))),
        "table4_labels_equal": bool(list(table["Simulation"]) == list(published["Simulation"])),
        "table4_max_abs_diff": table4_diff,
        "table4_within_rounding": bool(all(d <= 0.5 * 10**-TABLE4_DECIMALS + 1e-12 for d in table4_diff.values())),
        "fit": R.fit_summary(gmax, popt, pcov, dist="normal"),
        "gmax": gmax.to_dict(orient="records"),
    }


def check_p26(report):
    with open(LEGACY_ZENODO / "2d_spherical_growth_profiles.pkl", "rb") as f:
        profiles = pickle.load(f)
    ref_p26 = profiles["growth_peaks_std"].data[:, -1].max()
    csv = np.loadtxt(LEGACY_ZENODO / "spherical_1d_data.csv", delimiter=",")
    altitudes, growth = R.p26_profile()
    report["p26"] = {
        "value": R.p26_reference(),
        "identical_to_pkl": bool(R.p26_reference() == ref_p26),
        "profile_identical_to_csv": bool(np.array_equal(altitudes, csv[:, 1]) and np.array_equal(growth, csv[:, 0])),
    }


def check_staging(report):
    gmax = R.gmax_table(R.last_cycle_profiles())
    independent = {}
    for r in registry.campaign(R.CAMPAIGN):
        g = load_growth(cache_path(R.CAMPAIGN, r.alias, **R.STAGING_RECIPE))
        lasts = [np.ma.filled(np.ma.asarray(res["growth_peaks"], dtype=float), np.nan)[:, -1]
                 for res in g["growth_results"].values() if res is not None]
        independent[r.alias] = float(np.nanmax(np.concatenate(lasts)))
    popt, pcov = R.fit_inverse_n(gmax["N_phi"], gmax["G_max"])
    report["staging"] = {
        "gmax_identical_to_raw_caches": bool(all(independent[s] == v for s, v in zip(gmax["Simulation"], gmax["G_max"]))),
        "fit": R.fit_summary(gmax, popt, pcov, dist="t"),
        "gmax": gmax.to_dict(orient="records"),
    }


def passed(report) -> dict:
    lg, p26, st = report["legacy"], report["p26"], report["staging"]
    return {
        "legacy_gmax": lg["n_phi_equal"] and lg["gmax_identical"],
        "legacy_fit": lg["popt_identical"] and lg["pcov_identical"],
        "legacy_band": lg["band_identical"],
        "legacy_table4": lg["table4_labels_equal"] and lg["table4_within_rounding"],
        "p26": p26["identical_to_pkl"] and p26["profile_identical_to_csv"],
        "staging_gmax": st["gmax_identical_to_raw_caches"],
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


if __name__ == "__main__":
    report = {}
    check_legacy(report)
    check_p26(report)
    check_staging(report)
    report["passed"] = passed(report)

    out = output_dir("checks", "tables") / "regress_resolution_fit.json"
    out.write_text(json.dumps(to_jsonable(report), indent=2))

    for name, ok in report["passed"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    for src in ("legacy", "staging"):
        f = report[src]["fit"]
        print(f"{src:8s} A = {f['A']:.3f} ± {f['A_err']:.3f}, B = {f['B']:.1f} ± {f['B_err']:.1f}, "
              f"R² = {f['r2']:.3f}, band = {f['dist']}")
    print(f"Report: {out}")
    sys.exit(0 if all(report["passed"].values()) else 1)
