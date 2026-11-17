"""Cache the inputs of backup slides B1-B6 into small tracked files in deck/data/.

    cd refactored && conda run -n dsmc python deck/cache_backups.py

Sources (read-only here):
  B1  this repo's main checkout: exports/table_s1_manuscript.csv (Table S1), and in
      refactored/outputs/s1_s6_azimuthal/tables/ table4_resolution_staging.csv (G_max per run, corrected
      azimuthal-mean pre-forcing background, campaigns/s1_s6_azimuthal/resolution_fit.ipynb) and
      table_s3_lockin_attenuation_azimean_preforcing.csv (L_att); lambda_h from deck/data/slide06_facts.json.
  B2  ~/MarsAtmos-TruncBase wheel-fix report (reports/wheel_fix/numbers.tex) and its analysis outputs
      (analysis/out/null0c_ref12.json, nullfix_ref12.json: 2-D sector against twelve 1-D replicates);
      the residual-offset tests from ~/MarsAtmos-Temp2D/.../reports/temp2d_offset_explore.md.
  B3  ~/MarsAtmos-TruncBase docs/truncation_baseline_prereg.md (twins, metric) and the measured Stage 1B
      floor, analysis/out/floor1b_levels.csv with reports/1B_noise_floor.md (the Stage 2 tolerances it
      proposes are awaiting approval; B3 says so).
  B4  ~/MAGNUSP-isoO-baseline output/analysis/s01/meanflow/meanflow_S01A8.npz (1 m/s forcing) and the
      Plate 1 sidecar ../advisor_update_plates/plate_01_magnusp_isoo.json.
  B5  Plate 3 panel B, recomputed by ../advisor_update_plates/build_plate_03.load_and_compute() from the
      h2_ubc_rtop analysis files, checked against plate_03_dsmc_ubc_sweep.json.
  B6  outputs/s1_s6_azimuthal/analysis/phase_lag_S6_mad_column0.npz (deck/compute_phase_lag_cache.py),
      the root horizontal_phase_lag_figures/s6_max_distance_sweep_results.npz (s6_max_distance_sweep.py),
      the error budget of manuscript_review/phase_lag_reviewer_QA.md, and the column-mean vertical
      wavelengths of exports/fig5c_lambda_z_summary.csv.
Every quoted number is asserted against its source, so a changed source stops this script.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cache_slides78 import DATA, MAIN, PLATES, R_MARS_KM, S01, TB, TB_REPO, close, repo_head, tex_numbers

HERE = Path(__file__).resolve().parent
REFACTORED = HERE.parent
MAIN_REF = MAIN / "refactored"
S1S6 = MAIN_REF / "outputs" / "s1_s6_azimuthal"
WHEEL = TB / "reports" / "wheel_fix"
OUT_TB = TB / "analysis" / "out"
TEMP2D = Path.home() / "MarsAtmos-Temp2D" / "campaigns" / "truncation_baseline" / "reports" / "temp2d_offset_explore.md"
PREREG = TB_REPO / "docs" / "truncation_baseline_prereg.md"
MAGNUSP = S01.parents[2]
QA = MAIN / "manuscript_review" / "phase_lag_reviewer_QA.md"
SWEEP = MAIN / "horizontal_phase_lag_figures" / "s6_max_distance_sweep_results.npz"
RUNS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def text_has(path: Path, *phrases):
    t = path.read_text()
    for p in phrases:
        if p not in t:
            raise SystemExit(f"{path.name}: expected phrase not found: {p!r}")


# ---------------------------------------------------------------- B1
def b1() -> dict:
    s1 = pd.read_csv(MAIN / "exports" / "table_s1_manuscript.csv").set_index("Simulation")
    t4 = pd.read_csv(S1S6 / "tables" / "table4_resolution_staging.csv").set_index("Simulation")
    la = pd.read_csv(S1S6 / "tables" / "table_s3_lockin_attenuation_azimean_preforcing.csv")
    lam_h = json.loads((DATA / "slide06_facts.json").read_text())["lambda_h_odr_km"]
    la = la[la["target_altitude_km"] == 230.0].set_index("run")
    z_exo = float(la.loc["S6", "altitude_km"])                       # 231.4 km, where lambda_h is measured
    rows = []
    for r in RUNS:
        dphi = np.radians(s1.loc[r, "dPhi (deg)"])
        cw = dphi * (R_MARS_KM + z_exo)
        L = la["L_att_km"].get(r, np.nan)
        rows.append(dict(run=r, n_azi=int(s1.loc[r, "Total azimuth cells"]),
                         cell_km=float(cw), cells_per_lambda=float(lam_h / cw),
                         particles=float(s1.loc[r, "Total particles"]),
                         ppc_230=int(s1.loc[r, "Particles per cell 230 km"]),
                         weight=float(s1.loc[r, "Particle weight"]),
                         g_max=float(t4.loc[r, "G_max"]),
                         L_att_km=None if not np.isfinite(L) else float(L),
                         sigma_L_km=None if not np.isfinite(L) else float(la.loc[r, "sigma_L_total_km"]),
                         ensemble=r in ("S1", "S2", "S3")))
    # checks against Table S1 / Table 4 / slide 6
    close(rows[0]["particles"], 2.71e7, 1e5, "S1 total particles")
    close(rows[5]["particles"], 1.08e9, 1e7, "S6 total particles")
    close(rows[5]["L_att_km"], 118.947, 0.01, "S6 L_att at the exobase")
    close(rows[4]["g_max"], 5.9025, 1e-4, "S5 G_max (staging)")
    if rows[2]["L_att_km"] is not None:
        raise SystemExit("S3 now has an accepted L_att fit: revisit the B1 footnote")
    # Eq. 10 fit G_max = A - B / N_phi (unweighted least squares, as resolution_fit.ipynb)
    n = np.array([r["n_azi"] for r in rows], float)
    g = np.array([r["g_max"] for r in rows])
    X = np.column_stack([np.ones_like(n), -1 / n])
    (A, B), res, *_ = np.linalg.lstsq(X, g, rcond=None)
    dof = len(n) - 2
    cov = np.linalg.inv(X.T @ X) * (res[0] / dof)
    close(A, 5.9084, 5e-4, "G_max asymptote A")
    close(B, 82.0488, 0.05, "G_max fit B")
    return {"rows": rows, "fit": {"A": float(A), "A_err": float(np.sqrt(cov[0, 0])), "B": float(B),
                                  "B_err": float(np.sqrt(cov[1, 1]))},
            "z_exobase_km": z_exo, "lambda_h_km": lam_h,
            "sources": [str(MAIN / "exports" / "table_s1_manuscript.csv"),
                        str(S1S6 / "tables" / "table4_resolution_staging.csv"),
                        str(S1S6 / "tables" / "table_s3_lockin_attenuation_azimean_preforcing.csv")]}


# ---------------------------------------------------------------- B2
def _ref12(tag):
    d = json.loads((OUT_TB / f"{tag}.json").read_text())
    g = lambda k, f: np.array(d[k][f], float)
    return (np.array(d["z_km"], float), 100 * g("P1_num_dens", "rel_diff_2d_minus_1d"), 100 * g("P1_num_dens", "se"),
            100 * g("P1_temp_corr", "rel_diff_2d_minus_1d"), 100 * g("P1_temp_corr", "se"))


def b2() -> dict:
    N = tex_numbers(WHEEL / "numbers.tex")
    z, n_pre, se_n_pre, T_pre, se_T_pre = _ref12("null0c_ref12")
    z1, n_fix, se_n_fix, T_fix, se_T_fix = _ref12("nullfix_ref12")
    assert np.array_equal(z, z1)
    beta = np.radians(7.2)
    model = 100 * (((R_MARS_KM + 140.0) / (R_MARS_KM + z)) ** (np.cos(beta) ** 2) - 1)
    close(np.cos(beta) ** 2, N["expCos"], 1e-4, "cos^2 beta")
    for lev, key in ((192, "modelNA"), (292, "modelNB"), (392, "modelNC")):
        close(np.interp(lev, z, model), N[key], 0.02, f"model deficit at {lev} km")
    for lev, key, sekey in ((282, "fixXIITA", "fixXIITSEA"), (362, "fixXIITB", "fixXIITSEB")):
        k = int(np.argmin(abs(z - lev)))
        close(T_fix[k], N[key], 0.006, f"fixed T excess at {lev} km")
        close(se_T_fix[k], N[sekey], 0.006, f"fixed T SE at {lev} km")
    np.savez_compressed(DATA / "figB2_wheel_profiles.npz", z_km=z, n_pre=n_pre, se_n_pre=se_n_pre, n_fix=n_fix,
                        se_n_fix=se_n_fix, T_pre=T_pre, se_T_pre=se_T_pre, T_fix=T_fix, se_T_fix=se_T_fix,
                        model=model, beta_deg=7.2, base_km=140.0)
    text_has(TEMP2D, "+0.24% at 7.2° against +0.23% at 14.4°", "Variant A is statistically indistinguishable",
             "the 1-D reference is now the suspect")
    text_has(PREREG, "It is common to the base and all twins, so it cancels in δ.")
    keys = ("preNA", "preNB", "preNC", "fixNA", "fixNB", "fixNC", "modelNA", "modelNB", "modelNC", "fixXIITA",
            "fixXIITSEA", "fixXIITB", "fixXIITSEB", "fixXIINChi", "preXIINChi", "fixXIITChi", "expCos",
            "preEscRel", "fixEscRel")
    return {"numbers": {k: N[k] for k in keys}, "beta_deg": 7.2, "levels_km": [192, 292, 392],
            "offset_tests": {"estimator": "same excess in temp_kin (no per-cell mean subtraction)",
                             "collision_frame": "local-frame 2-D collisions: indistinguishable (+0.04%)",
                             "band": "band halved 14.4 -> 7.2 deg: +0.24% vs +0.23%"},
            "sources": [str(WHEEL / "numbers.tex"), str(OUT_TB / "null0c_ref12.json"),
                        str(OUT_TB / "nullfix_ref12.json"), str(TEMP2D)],
            "commit": repo_head(TB_REPO)}


# ---------------------------------------------------------------- B3
def b3() -> dict:
    """Measured Stage 1B floor (5 wave + 3 no-wave replicates) and the Stage 2 rule proposed from it."""
    rep = TB / "reports" / "1B_noise_floor.md"
    lv = pd.read_csv(OUT_TB / "floor1b_levels.csv")
    lv = lv[lv["in_zN"] == 1]
    K, n_rep, cut = 3.0, 5, 0.10
    out, tops = {}, {}
    for f in "nTuw":
        d = lv[lv["field"] == f].sort_values("z_km")
        term = K * d["delta_floor_cal"].to_numpy() / np.sqrt(n_rep)
        z = d["z_km"].to_numpy()
        ok = term <= cut                                           # scored: contiguous from the base
        tops[f] = float(z[:np.argmin(ok)][-1]) if not ok.all() else float(z[-1])
        out[f"{f}_z_km"], out[f"{f}_term"] = z, term
    for f, want in (("n", 232), ("T", 207), ("u", 252), ("w", 267)):
        close(tops[f], want, 1.0, f"B3 scored top for {f}")
    base = {f: float(out[f"{f}_term"][0]) for f in "nTuw"}
    close(100 * lv[(lv["field"] == "n")].sort_values("z_km")["delta_floor_cal"].iloc[0], 1.5, 0.05, "B3 n floor at 142 km")
    np.savez_compressed(DATA / "figB3_floor.npz", **out)
    text_has(rep, "ANALYSED; Stage 2 tolerances proposed, awaiting approval",
             "c = 1.09 (n'/n), 1.01 (T'/T), 1.01 (u), 1.07 (w)",
             "n'/n ≤ 232 km, T'/T ≤ 207, u ≤ 252, w ≤ 267",
             "Measured/model is 1.01–1.09 across fields", "**5%** (amplitude, or 2.9° in phase)")
    text_has(PREREG, "awaiting approval", "PASS if at most 5% of (level, field) pairs fail")
    twins = [("Sector width", "two wavelengths: 800 km, 100 cells"),
             ("Band thickness", "0.26° and 2.0° instead of 14.4°"),
             ("Azimuthal cell width", "4 km cells instead of 8 km"),
             ("Domain top", "600 km instead of 500 km")]
    return {"twins": twins, "K": K, "delta_phys": 0.05, "n_rep": n_rep, "score_cut": cut, "scored_top_km": tops,
            "z_N_km": float(lv["z_km"].max()), "term_base_pct": {f: 100 * v for f, v in base.items()},
            "c": {"n": 1.09, "T": 1.01, "u": 1.01, "w": 1.07}, "meas_over_model": [1.01, 1.09],
            "n_wave": 5, "n_nowave": 3,
            "status": "measured 1B floor; Stage 2 tolerances proposed, awaiting approval (prereg not tagged)",
            "sources": [str(OUT_TB / "floor1b_levels.csv"), str(rep), str(PREREG)],
            "commit": repo_head(TB_REPO)}


# ---------------------------------------------------------------- B4
def b4() -> dict:
    p1 = json.loads((PLATES / "plate_01_magnusp_isoo.json").read_text())
    hv = np.load(S01 / "meanflow" / "meanflow_S01A8.npz")
    hb = np.load(S01 / "meanflow" / "meanflow_S01B8.npz")
    zsel = hv["z_m"] <= 200.5e3
    t_h, z_km, u = hv["t_s"] / 3600, hv["z_m"][zsel] / 1e3, hv["u_m"][:, zsel]
    s = z_km <= 180
    k = int(np.argmin(u[-1, s]))
    close(float(u[-1, s][k]), p1["u_peak"], 1e-9, "B4 peak westward wind")
    close(float(z_km[s][k]), p1["u_peak_z_km"], 1e-9, "B4 peak wind altitude")
    uB = float(np.abs(hb["u_m"][:, hb["z_m"] <= 200e3]).max())
    close(uB, p1["uB_max"], 1e-9, "B4 weak-forcing wind")
    dop = json.loads((S01 / "isoo_doppler.json").read_text())["cells"]["S01A8"]["levels"]["140.25"]
    close(dop["dphase_obs_deg"], p1["dphase_obs"], 1e-9, "B4 phase shift")
    np.savez_compressed(DATA / "figB4_meanflow.npz", t_h=t_h, z_km=z_km, u=u.astype(np.float32))
    return {"u_peak_m_s": p1["u_peak"], "u_peak_z_km": p1["u_peak_z_km"], "u_peak_t_h": p1["u_peak_t_h"],
            "dphase_obs_deg": p1["dphase_obs"], "dphase_lin_deg": p1["dphase_lin"], "uB_max_m_s": uB,
            "period_min": 60, "lambda_x_km": p1["lx_km"], "amp_weak_m_s": p1["amp_B"], "n_weak_streams": p1["n_B_streams"],
            "sources": [str(S01 / "meanflow" / "meanflow_S01A8.npz"), str(S01 / "isoo_doppler.json"),
                        str(PLATES / "plate_01_magnusp_isoo.json")], "commit": repo_head(MAGNUSP)}


# ---------------------------------------------------------------- B5
def b5() -> dict:
    sys.path.insert(0, str(PLATES))
    import build_plate_03 as p3                                    # noqa: E402  (adds its own paths)
    dd = p3.load_and_compute()
    N, ref = dd["N"], json.loads((PLATES / "plate_03_dsmc_ubc_sweep.json").read_text())
    for key in ("res_band_offset_pct_min", "res_band_offset_pct_max", "exo_all_min_km", "exo_all_max_km",
                "res_band_lo_km", "res_band_hi_km", "dr_fine_km", "dr_coarse_km"):
        close(N[key], ref[key], 1e-9, f"B5 {key}")
    arrs = {}
    for (atm, ubc), sr in dd["series_b2"].items():
        arrs[f"{atm}_u{ubc}_alt_km"] = sr["alt_km"]
        arrs[f"{atm}_u{ubc}_ratio"] = sr["ratio"]
    np.savez_compressed(DATA / "figB5_grid_ratio.npz", **arrs)
    return {k: N[k] for k in ("res_band_offset_pct_min", "res_band_offset_pct_max", "res_max_dn_pct_min",
                              "res_max_dn_pct_max", "exo_all_min_km", "exo_all_max_km", "res_band_lo_km",
                              "res_band_hi_km", "dr_fine_km", "dr_coarse_km", "conv_thresh_n_pct", "n_runs",
                              "z_top_min_km")} | {
        "z_top_ratio_km": 20000, "ubcs": {"1": "specular reflection", "2": "ballistic hold", "3": "free escape"},
        "atms": {"H2": "pure H2", "H2O": "coupled O + H2"},
        "sources": [str(PLATES / "build_plate_03.py"), str(PLATES / "plate_03_dsmc_ubc_sweep.json"),
                    str(MAIN_REF / "outputs" / "h2_ubc_rtop")]}


# ---------------------------------------------------------------- B6
def b6() -> dict:
    pl = np.load(REFACTORED / "outputs" / "s1_s6_azimuthal" / "analysis" / "phase_lag_S6_mad_column0.npz")
    sw = np.load(SWEEP)
    close(float(pl["lambda_h_km"]), 450.4035, 1e-3, "B6 lambda_h")
    k300 = int(np.where(sw["baselines"] == 300)[0][0])
    close(float(sw["coupled_lam"][k300]), float(pl["lambda_h_km"]), 1e-6, "B6 sweep at 300 km = headline")
    text_has(QA, "| **Combined L+R (headline)** | **450.4** | 7.6 | 28 |", "| Left edge only | 443.6 | 12.6 | 14 |",
             "| Right edge only | 458.7 | 6.0 | 14 |", "λ_h = **450 ± 11 (stat., calibrated) ± 9 (syst., window/construction) km**",
             "max lag 0.640 P")
    lz = pd.read_csv(MAIN / "exports" / "fig5c_lambda_z_summary.csv")
    r = lz[(lz["run"] == "1024 azi") & (lz["segment"] == "full")].iloc[0]
    close(r["mean"], 441.3, 0.05, "S6 column-mean lambda_z")
    keep = sw["baselines"] <= 800
    np.savez_compressed(DATA / "figB6_phase_lag.npz",
                        **{k: pl[k] for k in ("distance_km", "lag_periods", "lag_err_periods", "distance_err_km",
                                              "is_left", "used", "slope_per_km", "intercept_periods")},
                        baselines_km=sw["baselines"][keep], coupled_lam=sw["coupled_lam"][keep],
                        coupled_err=sw["coupled_err"][keep], coupled_maxlag=sw["coupled_maxlag"][keep],
                        coupled_chi2=sw["coupled_chi2"][keep])
    stat, syst = 11.0, 9.0
    return {"lambda_h_km": float(pl["lambda_h_km"]), "sigma_odr_km": float(pl["lambda_h_err_km"]), "n": int(pl["n"]),
            "reduced_chi2": float(pl["reduced_chi2"]), "altitude_km": float(pl["altitude_km"]),
            "max_distance_km": float(pl["distance_km"].max()), "max_lag_periods": float(pl["lag_periods"].max()),
            "sigma_stat_km": stat, "sigma_syst_km": syst, "sigma_total_km": float(np.hypot(stat, syst)),
            "left": [443.6, 12.6], "right": [458.7, 6.0], "lag_ceiling_periods": 0.7,
            "lambda_z_colmean_km": float(r["mean"]), "lambda_z_colstd_km": float(r["std"]),
            "lambda_z_ncol": int(r["n_columns"]),
            "sweep_200_800_km": [float(sw["coupled_lam"][(sw["baselines"] >= 200) & keep].min()),
                                 float(sw["coupled_lam"][(sw["baselines"] >= 200) & keep].max())],
            "sweep_200_400_km": [float(sw["coupled_lam"][(sw["baselines"] >= 200) & (sw["baselines"] <= 400)].min()),
                                 float(sw["coupled_lam"][(sw["baselines"] >= 200) & (sw["baselines"] <= 400)].max())],
            "sources": [str(REFACTORED / "outputs/s1_s6_azimuthal/analysis/phase_lag_S6_mad_column0.npz"),
                        str(SWEEP), str(QA), str(MAIN / "exports" / "fig5c_lambda_z_summary.csv")]}


def main():
    facts = {"created": dt.date.today().isoformat(), "dsmc_analyzer_commit": repo_head(MAIN),
             "B1": b1(), "B2": b2(), "B3": b3(), "B4": b4(), "B5": b5(), "B6": b6()}
    (DATA / "backup_facts.json").write_text(json.dumps(facts, indent=1, default=float) + "\n")
    print(DATA / "backup_facts.json")


if __name__ == "__main__":
    main()
