"""h2_ubc_rtop full-matrix analysis (60 runs): card inventory, convergence, profiles, escape, and
Carberry Mogan et al. (2021) comparisons (Figs. 1-3 and Table 1).

    python -B campaigns/h2_ubc_rtop/compute_analysis.py

Writes:
- outputs/h2_ubc_rtop/analysis/<alias>_analysis.nc (per completed run)
- outputs/h2_ubc_rtop/analysis/h2_ubc_rtop_analysis.nc (cross-run summary & paired U1/U2/U3 series)
- outputs/h2_ubc_rtop/tables/{card_inventory,convergence_summary,escape_summary,
  carberry_exobase_table1,ballistic_rates_summary,ztop_resolution_convergence}.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dsmclib import convergence as cv
from dsmclib import escape as es
from dsmclib import exosphere as exo
from dsmclib import profiles as pf
from dsmclib import registry, tiers
from dsmclib.constants import G_NEWTON, K_BOLTZMANN
from dsmclib.ncrun import RUN_SUFFIX, open_run_nc
from dsmclib.paths import output_dir

CAMPAIGN = "h2_ubc_rtop"
SPINUP = 10000.0          # s, initial spin-up excluded from filling fits
PROFILE_WINDOW = (100000.0, 300000.0)  # trailing 200 ks steady window for profiles & escape
T_FIELD = "temp_trans_corr"
N_BATCHES = 10            # batch-means blocks for the paired U1/U2/U3 time series (30 ks each)
CONV_ALT_KM = (105.0, 2000.0, 80)   # altitude grid (start, stop, n) for z_top / dr_max comparisons
CONV_THRESH = {"n": 0.05, "T": 0.02, "escape": 0.10}   # EXPERIMENT_DESIGN.md acceptance criteria
SWEEP_ROOT = Path.home() / "MarsAtmos-H2UBC-sweep"


def parse_alias(alias: str) -> dict:
    m = re.match(r"^(H2O|H2)_U([123])_z(\d+)_dr(\d+)$", alias)
    if not m:
        raise ValueError(f"Unrecognised h2_ubc_rtop alias: {alias}")
    atm, ubc_num, z_nom, dr_max = m.groups()
    ubc_map = {"1": "reflect", "2": "ballistic", "3": "free_escape"}
    return {
        "atmosphere": atm,
        "ubc_num": int(ubc_num),
        "ubc": ubc_map[ubc_num],
        "z_top_nominal_km": int(z_nom),
        "dr_max_km": int(dr_max),
    }


def exobase_kn0(prof: xr.Dataset, r0: float) -> xr.DataArray:
    """Altitude [km] where Kn_0(r) = mean_free_path(r) / r0 first reaches 1 (Carberry Mogan et al. 2021)."""
    alt = prof["altitude"].values / 1e3
    out = []
    for sp in prof["species"].values:
        kn0 = prof["mean_free_path"].sel(species=sp).values / r0
        ok = np.isfinite(kn0) & (kn0 > 0)
        a, k = alt[ok], np.log(kn0[ok])
        above = np.where(k >= 0)[0]
        if len(above) == 0:
            out.append(np.nan)
        elif above[0] == 0:
            out.append(a[0])
        else:
            i = above[0]
            out.append(np.interp(0.0, [k[i - 1], k[i]], [a[i - 1], a[i]]))
    return xr.DataArray(out, dims="species", coords={"species": prof["species"]},
                        attrs={"units": "km", "long_name": "Exobase altitude (Kn_0 = l/r_0 = 1)"})


def build_card_inventory() -> pd.DataFrame:
    """Read every <id>_card.json in ~/MarsAtmos-H2UBC-sweep/runs/*/Results/ into a DataFrame."""
    cards = sorted((SWEEP_ROOT / "runs").glob("*/Results/*_card.json"))
    rows = []
    for cp in cards:
        c = json.loads(cp.read_text())
        alias = c["run"]["alias"]
        if alias.endswith("_pilot"):
            continue
        meta = parse_alias(alias)
        dom = c["domain"]
        tm = c["time"]
        prov = c["provenance"]
        sp_map = {s["name"]: s for s in c["species"]}
        h2 = sp_map["H2"]
        o = sp_map.get("O", {})
        status_ok = all(s["level"] in ("ok", "note") for s in c["status"])
        rows.append({
            "run": alias,
            **meta,
            "ubc_solver": dom["ubc"],
            "lower_alt_km": dom["lower_alt_km"],
            "upper_alt_km": dom["upper_alt_km"],
            "r_max_over_r0": (dom["planet_radius_km"] + dom["upper_alt_km"]) / (dom["planet_radius_km"] + dom["lower_alt_km"]),
            "n_cells": dom["n_cells"],
            "dr_min_km": dom["dr_min_km"],
            "dt_s": tm["dt_s"],
            "n_steps": tm["n_steps"],
            "t_end_s": tm["t_end_s"],
            "avg_window_s": tm["avg_window_s"],
            "n_outputs": tm["n_outputs"],
            "weight_H2": h2["weight"],
            "n0_H2_m3": h2["n0_m3"][0],
            "T0_K": h2["T0_K"][0],
            "flow_rate0_H2_s": h2["flow_rate_s"][0],
            "jeans_lambda0_H2": h2["jeans_lambda0"][0],
            "jeans_rate0_H2_s": h2["jeans_rate0_s"][0],
            "npart_final_H2": h2["npart_final"],
            "nphys_final_H2": h2["nphys_final"],
            "npart_escaped_H2": h2["npart_escaped"],
            "weight_O": o.get("weight", np.nan),
            "n0_O_m3": o.get("n0_m3", [np.nan])[0],
            "jeans_lambda0_O": o.get("jeans_lambda0", [np.nan])[0],
            "npart_final_O": o.get("npart_final", np.nan),
            "npart_escaped_O": o.get("npart_escaped", np.nan),
            "wall_s": prov["wall_s"],
            "commit": prov["commit"],
            "builder_version": prov["builder_version"],
            "status_ok": status_ok,
        })
    df = pd.DataFrame(rows).sort_values(["atmosphere", "z_top_nominal_km", "dr_max_km", "ubc_num"]).reset_index(drop=True)
    return df


def analyse_run(alias: str, window: tuple[float, float]) -> tuple[xr.Dataset, list, list, list]:
    run = open_run_nc(alias)
    meta = parse_alias(alias)
    run.macro[T_FIELD] = pf.temp_trans_corrected(run)
    t_end = run.t_end
    inv = cv.inventory(run)
    rates = cv.boundary_rates(run)
    ds = xr.Dataset({"N_phys": inv.N_phys, "N_part": inv.N_part})
    for v in ("inject_rate", "return_rate", "escape_rate", "escape_npart", "dN_dt"):
        if v in rates:
            ds[v] = rates[v]

    # Ballistic held reservoir N_ball(t) and its rate of change dN_ball/dt
    if run.escape is not None and "num_part_ball" in run.escape:
        n_ball_part = run.escape["num_part_ball"].sum("boundary_cell")
    else:
        n_ball_part = xr.zeros_like(inv.N_part)
    n_ball_phys = (n_ball_part * run.weight).assign_attrs(
        units="1", long_name="Physical molecules held in Keplerian ballistic reservoir above r_top")
    ds["N_ball_part"] = n_ball_part.assign_attrs(
        units="1", long_name="Computational particles held in Keplerian ballistic reservoir above r_top")
    ds["N_ball_phys"] = n_ball_phys
    dn_ball_dt = xr.DataArray(
        np.gradient(n_ball_phys.values, run.time, axis=n_ball_phys.get_axis_num("time")),
        dims=n_ball_phys.dims, coords=n_ball_phys.coords,
        attrs={"units": "s-1", "long_name": "Net rate of molecules accumulating in ballistic reservoir (Phi_ball_in - Phi_ball_ret)"})
    ds["dN_ball_dt"] = dn_ball_dt

    conv_rows = []
    steady_vars = {k: [] for k in ("steady", "t_steady_start", "rel_drift_ub", "drift_per_ks",
                                   "fill_tau", "fill_tau_err", "fill_N_inf", "fill_N_inf_err", "fill_tau_over_span")}
    for sp in run.species:
        r = cv.steady_window(inv.N_phys.sel(species=sp), t_end=t_end)
        f = cv.filling_timescale(inv.N_phys.sel(species=sp), SPINUP, t_end)
        # Also evaluate drift specifically on the analysis window [100, 300] ks
        sel_w = (run.time > window[0]) & (run.time <= window[1])
        dw = cv.drift_test(run.time[sel_w], inv.N_phys.sel(species=sp).values[sel_w])
        steady_vars["steady"].append(int(r.steady))
        steady_vars["t_steady_start"].append(np.nan if r.t_start is None else r.t_start)
        steady_vars["rel_drift_ub"].append(r.rel_drift_ub)
        steady_vars["drift_per_ks"].append(r.drift_per_ks)
        steady_vars["fill_tau"].append(f["tau"])
        steady_vars["fill_tau_err"].append(f["tau_err"])
        steady_vars["fill_N_inf"].append(f["N_inf"])
        steady_vars["fill_N_inf_err"].append(f["N_inf_err"])
        steady_vars["fill_tau_over_span"].append(f["tau_over_span"])
        conv_rows.append({
            "run": alias, **meta, "species": sp,
            "steady": r.steady,
            "t_steady_start_s": r.t_start,
            "drift_pct_per_ks": 100 * r.drift_per_ks,
            "window_drift_pct_per_ks": 100 * 1e3 * dw["slope"] / dw["mean"],
            "window_rel_drift_ub": dw["rel_drift_ub"],
            "rel_drift_ub": r.rel_drift_ub,
            "N_phys_end": float(inv.N_phys.sel(species=sp).isel(time=-1)),
            "N_part_end": float(inv.N_part.sel(species=sp).isel(time=-1)),
            "N_ball_phys_end": float(n_ball_phys.sel(species=sp).isel(time=-1)),
            "N_ball_part_end": float(n_ball_part.sel(species=sp).isel(time=-1)),
            "fill_tau_s": f["tau"], "fill_tau_err_s": f["tau_err"],
            "fill_tau_over_span": f["tau_over_span"], "t_end_s": t_end,
        })
    for k, v in steady_vars.items():
        ds[k] = ("species", np.array(v, dtype=float))

    prof = pf.time_mean_profiles(run, *window)
    for v in prof.data_vars:
        ds[f"profile_{v}"] = prof[v]
    n_iso = pf.isothermal_reference(run, prof)
    ds["profile_num_dens_isothermal"] = n_iso
    ds["profile_num_dens_ratio_isothermal"] = (prof["num_dens"] / n_iso).assign_attrs(units="1")

    edges = run.radial_edges()
    r0, r_top = float(edges[0]), float(edges[-1])
    z_top_actual_km = (r_top - run.planet_radius) / 1e3

    # Exobase at Kn = l/H = 1 and at Kn_0 = l/r0 = 1 (Carberry Mogan et al. 2021 Table 1)
    jexo = es.jeans_at_exobase(run, prof, temperature_field=T_FIELD)
    for v in jexo.data_vars:
        ds[v] = jexo[v]
    z_exo_kn0 = exobase_kn0(prof, r0)
    ds["exobase_kn0_altitude"] = z_exo_kn0
    n_exo_kn0 = pf.interp_at_altitude(prof, "num_dens", z_exo_kn0, log=True)
    t_exo_kn0 = pf.interp_at_altitude(prof, T_FIELD, z_exo_kn0)
    r_exo_kn0 = run.planet_radius + z_exo_kn0 * 1e3
    j_kn0 = es.jeans_escape(n_exo_kn0, t_exo_kn0, run.mass, r_exo_kn0, run.planet_mass)

    jlb = es.jeans_at_lower_boundary(run)
    ds["jeans_rate_lower_boundary"] = jlb["jeans_rate"].assign_attrs(units="s-1")
    ds["jeans_lambda_lower_boundary"] = jlb["jeans_parameter"].assign_attrs(units="1")

    esc = es.dsmc_escape(run, *window)
    for v in esc.data_vars:
        ds[f"dsmc_{v}"] = esc[v]
    bal = es.lower_boundary_balance(run, *window)
    for v in bal.data_vars:
        ds[f"balance_{v}"] = bal[v]

    # Collisionless Liouville reference profile & escape rate for H2 (anchored at DSMC exobase and at r0)
    m_h2 = float(run.mass.sel(species="H2"))
    n0_h2 = float(run.info_surface["num_dens"].sel(species="H2").isel(surface_cell=0))
    T0 = float(run.info_surface["T0"].values.ravel()[0])
    z_exo_h2 = float(jexo.exobase_altitude.sel(species="H2"))
    r_exo_h2 = run.planet_radius + z_exo_h2 * 1e3
    n_exo_h2 = float(jexo.n_exobase.sel(species="H2"))
    T_exo_h2 = float(jexo.T_exobase.sel(species="H2"))

    liou_exo = exo.profile(prof["radial"].values, n_exo_h2, T_exo_h2, m_h2, r_exo_h2, r_top,
                           ubc=meta["ubc"], n_v=400, vmax_inject=8000.0)
    ds["liouville_exo_num_dens_H2"] = ("radial", liou_exo["num_dens"].values)
    ds["liouville_exo_temp_trans_H2"] = ("radial", liou_exo["temp_trans"].values)
    ds["liouville_exo_vel_rad_H2"] = ("radial", liou_exo["vel_rad"].values)

    esc_liou_r0 = exo.escape_rate(n0_h2, T0, m_h2, r0, r_top, ubc=meta["ubc"], n_v=800)
    esc_liou_exo = exo.escape_rate(n_exo_h2, T_exo_h2, m_h2, r_exo_h2, r_top, ubc=meta["ubc"], n_v=800)

    # For ballistic UBC, compute analytic Liouville-Kepler held_count(t) and ballistic_rates(t)
    if meta["ubc"] == "ballistic":
        held_an_t = exo.held_count(n_exo_h2, T_exo_h2, m_h2, r_exo_h2, r_top, n_v=800, t=run.time)
        held_an_inf = exo.held_count(n_exo_h2, T_exo_h2, m_h2, r_exo_h2, r_top, n_v=800, t=None)
        br_an = exo.ballistic_rates(n_exo_h2, T_exo_h2, m_h2, r_exo_h2, r_top, n_v=800, t=run.time)
        ds["kepler_held_phys_H2"] = ("time", held_an_t)
        ds["kepler_ballistic_in_rate_H2"] = ("time", br_an["ballistic_in_rate"])
        ds["kepler_ballistic_return_rate_H2"] = ("time", br_an["ballistic_return_rate"])
        ds["kepler_ballistic_ratio_H2"] = ("time", br_an["ratio_in_over_return"])
        ds.attrs["kepler_held_asymptotic_H2"] = held_an_inf
        ds.attrs["kepler_ballistic_asymptotic_rate_H2"] = br_an["asymptotic_rate"]
    else:
        held_an_inf = 0.0

    # Values at reference altitudes (z = 2000 km)
    z_km = prof["altitude"].values / 1e3
    n2000_h2 = float(np.exp(np.interp(2000.0, z_km, np.log(prof["num_dens"].sel(species="H2").values))))
    n2000_se_h2 = float(np.interp(2000.0, z_km, prof["num_dens_sem"].sel(species="H2").values))
    T2000_h2 = float(np.interp(2000.0, z_km, prof[T_FIELD].sel(species="H2").values))
    T2000_se_h2 = float(np.interp(2000.0, z_km, prof[f"{T_FIELD}_sem"].sel(species="H2").values))
    Traw2000_h2 = float(np.interp(2000.0, z_km, prof["temp_trans"].sel(species="H2").values))
    ok_part = prof["num_part"].sel(species="H2").values >= 10
    bias_h2 = (prof["temp_trans"].sel(species="H2") - prof[T_FIELD].sel(species="H2")).values
    max_bias_K = float(np.nanmax(bias_h2[ok_part])) if ok_part.any() else np.nan

    esc_rows = []
    cm_rows = []
    for sp in run.species:
        phi_dsmc = float(esc.escape_rate.sel(species=sp))
        k_esc = int(esc.escape_npart.sel(species=sp))
        phi_dsmc_se = phi_dsmc / np.sqrt(max(k_esc, 1))
        phi_j_exo = float(jexo.jeans_rate.sel(species=sp))
        phi_j_0 = float(jlb.jeans_rate.sel(species=sp))
        lam_0 = float(jlb.jeans_parameter.sel(species=sp))
        lam_exo = float(jexo.jeans_parameter.sel(species=sp))
        r_x = run.planet_radius + float(jexo.exobase_altitude.sel(species=sp)) * 1e3
        r_x_kn0 = run.planet_radius + float(z_exo_kn0.sel(species=sp)) * 1e3
        phi_j_kn0 = float(j_kn0["rate"].sel(species=sp))
        lam_kn0 = float(j_kn0["lambda"].sel(species=sp))

        # Mean dN_ball/dt over window
        sel_w = (run.time > window[0]) & (run.time <= window[1])
        if meta["ubc"] == "ballistic":
            t_w = run.time[sel_w]
            nb_w = n_ball_phys.sel(species=sp).values[sel_w]
            fit_b = cv.drift_test(t_w, nb_w)
            dn_ball_w = fit_b["slope"]
            dn_ball_w_se = fit_b["slope_se"]
        else:
            dn_ball_w, dn_ball_w_se = 0.0, 0.0

        esc_rows.append({
            "run": alias, **meta, "z_top_actual_km": z_top_actual_km,
            "r_max_over_r0": r_top / r0,
            "species": sp,
            "exobase_km": float(jexo.exobase_altitude.sel(species=sp)),
            "exobase_kn0_km": float(z_exo_kn0.sel(species=sp)),
            "n_exobase_m3": float(jexo.n_exobase.sel(species=sp)),
            "T_exobase_K": float(jexo.T_exobase.sel(species=sp)),
            "jeans_lambda_exobase": lam_exo,
            "jeans_rate_exobase_s": phi_j_exo,
            "jeans_rate_lower_boundary_s": phi_j_0,
            "liouville_escape_r0_s": esc_liou_r0 if sp == "H2" else np.nan,
            "liouville_escape_exo_s": esc_liou_exo if sp == "H2" else np.nan,
            "dsmc_escape_npart": k_esc,
            "dsmc_escape_rate_s": phi_dsmc,
            "dsmc_escape_rate_se_s": phi_dsmc_se,
            "dsmc_over_jeans_exo": phi_dsmc / phi_j_exo if phi_j_exo > 0 else np.nan,
            "dsmc_over_jeans_0": phi_dsmc / phi_j_0 if phi_j_0 > 0 else np.nan,
            "balance_outflow_s": float(bal.outflow.sel(species=sp)),
            "balance_outflow_se_s": float(bal.outflow_se.sel(species=sp)),
            "dN_ball_dt_window_s": dn_ball_w,
            "dN_ball_dt_window_se_s": dn_ball_w_se,
            "total_top_loss_balance_s": float(bal.outflow.sel(species=sp)) - dn_ball_w,
            "n_2000km_m3": n2000_h2 if sp == "H2" else np.nan,
            "n_2000km_se_m3": n2000_se_h2 if sp == "H2" else np.nan,
            "Tcorr_2000km_K": T2000_h2 if sp == "H2" else np.nan,
            "Tcorr_2000km_se_K": T2000_se_h2 if sp == "H2" else np.nan,
            "Traw_2000km_K": Traw2000_h2 if sp == "H2" else np.nan,
            "max_T_bias_K": max_bias_K if sp == "H2" else np.nan,
            "held_end_phys": float(n_ball_phys.sel(species=sp).isel(time=-1)),
            "held_asymptotic_phys": held_an_inf if sp == "H2" else np.nan,
        })

        if sp == "H2":
            cm_rows.append({
                "run": alias, **meta, "z_top_actual_km": z_top_actual_km,
                "rx_over_r0_Kn1": r_x / r0,
                "rx_over_r0_Kn0": r_x_kn0 / r0,
                "nx_over_n0_Kn1": float(jexo.n_exobase.sel(species="H2")) / n0_h2,
                "nx_over_n0_Kn0": float(n_exo_kn0.sel(species="H2")) / n0_h2,
                "Tx_over_T0_Kn1": float(jexo.T_exobase.sel(species="H2")) / T0,
                "Tx_over_T0_Kn0": float(t_exo_kn0.sel(species="H2")) / T0,
                "lambda_x_over_lambda0_Kn1": lam_exo / lam_0,
                "lambda_x_over_lambda0_Kn0": lam_kn0 / lam_0,
                "phiJ_x_over_phiJ0_Kn1": phi_j_exo / phi_j_0,
                "phiJ_x_over_phiJ0_Kn0": phi_j_kn0 / phi_j_0,
                "Phi_over_phiJ_x_Kn1": phi_dsmc / phi_j_exo,
                "Phi_over_phiJ_x_Kn0": phi_dsmc / phi_j_kn0,
            })

    ds = ds.assign_coords(species_mass=run.mass, particle_weight=run.weight)
    ds.attrs.update({
        **tiers.analysis_attrs(f"{alias}: h2_ubc_rtop 1-D DSMC convergence, profiles, escape and ballistic diagnostics",
                               [run.path], "campaigns/h2_ubc_rtop/compute_analysis.py"),
        "run_alias": alias,
        **meta,
        "z_top_actual_km": z_top_actual_km,
        "r0_m": r0,
        "r_top_m": r_top,
        "r_max_over_r0": r_top / r0,
        "profile_window_t0": window[0],
        "profile_window_t1": window[1],
        "spinup_excluded": SPINUP,
        "planet_radius": run.planet_radius,
        "planet_mass": run.planet_mass,
    })
    return ds, conv_rows, esc_rows, cm_rows


def compute_cross_run_and_ballistic_tables(per_run: dict, esc_df: pd.DataFrame) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """Compute paired U1/U2/U3 Carberry Mogan Fig. 3 series & summary tables, plus z_top/dr_max convergence."""
    ball_rows = []
    series_vars = {}
    sources = []

    for atm in ("H2", "H2O"):
        for z_nom in (2500, 5000, 10000, 20000, 40000):
            for dr in (1000, 250):
                a1 = f"{atm}_U1_z{z_nom}_dr{dr}"
                a2 = f"{atm}_U2_z{z_nom}_dr{dr}"
                a3 = f"{atm}_U3_z{z_nom}_dr{dr}"
                if a2 not in per_run:
                    continue
                d2 = per_run[a2]
                t = d2["time"].values
                sel_late = (t > 100000.0) & (t <= 300000.0)
                sel_early = (t > 10000.0) & (t <= 50000.0)

                # Kepler analytic rates for U2
                k_in = d2["kepler_ballistic_in_rate_H2"].values
                k_ret = d2["kepler_ballistic_return_rate_H2"].values
                k_rat = d2["kepler_ballistic_ratio_H2"].values
                k_held = d2["kepler_held_phys_H2"].values
                k_held_inf = float(d2.attrs["kepler_held_asymptotic_H2"])

                # DSMC held inventory and dN_ball/dt
                dsmc_held = d2["N_ball_phys"].sel(species="H2").values
                dsmc_dn_ball_dt = d2["dN_ball_dt"].sel(species="H2").values
                phi2_t = d2["escape_rate"].sel(species="H2").values
                phi2_mean = float(phi2_t[sel_late].mean())

                # If U1 is available: escape ratio Phi_U2 / Phi_U1
                if a1 in per_run:
                    phi1_t = per_run[a1]["escape_rate"].sel(species="H2").values
                    phi1_mean = float(phi1_t[sel_late].mean())
                    esc_ratio_mean = phi2_mean / phi1_mean
                    # 10-batch series (30 outputs per batch = 30 ks blocks)
                    b_t = cv.batch_means(t, N_BATCHES)
                    b_phi1 = cv.batch_means(phi1_t, N_BATCHES)
                    b_phi2 = cv.batch_means(phi2_t, N_BATCHES)
                    b_esc_ratio = b_phi2 / b_phi1
                else:
                    phi1_mean, esc_ratio_mean = np.nan, np.nan
                    b_t = cv.batch_means(t, N_BATCHES)
                    b_esc_ratio = np.full(N_BATCHES, np.nan)

                # If U3 is available: DSMC upward bound crossing rate Phi_ball_in = Phi_U3 - Phi_U2
                if a3 in per_run:
                    phi3_t = per_run[a3]["escape_rate"].sel(species="H2").values
                    phi_ball_in_t = phi3_t - phi2_t
                else:
                    phi_ball_in_t = k_in

                phi_ball_in_late = float(phi_ball_in_t[sel_late].mean())
                dn_ball_late = float(cv.drift_test(t[sel_late], dsmc_held[sel_late])["slope"])
                dn_ball_early = float(cv.drift_test(t[sel_early], dsmc_held[sel_early])["slope"])
                phi_ret_late = phi_ball_in_late - dn_ball_late
                ratio_dsmc_late = phi_ball_in_late / phi_ret_late if phi_ret_late > 0 else np.nan

                # Batch-means (10 blocks of 30 ks) for smooth DSMC Ballistic Rate / Return Rate time series
                b_in = cv.batch_means(phi_ball_in_t, N_BATCHES)
                b_dnb = cv.batch_means(dsmc_dn_ball_dt, N_BATCHES)
                b_ret = np.maximum(b_in - b_dnb, 1e10)
                b_ball_ratio = b_in / b_ret

                tag = f"{atm}_z{z_nom}_dr{dr}"
                series_vars[f"{tag}_batch_time"] = ("batch", b_t)
                series_vars[f"{tag}_batch_esc_ratio_U2_U1"] = ("batch", b_esc_ratio)
                series_vars[f"{tag}_batch_ball_in_over_ret"] = ("batch", b_ball_ratio)
                series_vars[f"{tag}_batch_ball_in_rate"] = ("batch", b_in)
                series_vars[f"{tag}_batch_ball_ret_rate"] = ("batch", b_ret)

                # Orbital period of a bound orbit with apoapsis at r_top and periapsis at r0: a = (r0 + r_top)/2
                r0 = float(d2.attrs["r0_m"])
                r_top = float(d2.attrs["r_top_m"])
                a_orb = 0.5 * (r0 + r_top)
                t_orb_top = 2 * np.pi * np.sqrt(a_orb ** 3 / (G_NEWTON * float(d2.attrs["planet_mass"])))

                ball_rows.append({
                    "group": tag,
                    "atmosphere": atm,
                    "z_top_nominal_km": z_nom,
                    "z_top_actual_km": float(d2.attrs["z_top_actual_km"]),
                    "dr_max_km": dr,
                    "t_orb_top_ks": t_orb_top / 1e3,
                    "n_orbits_in_300ks": 300000.0 / t_orb_top,
                    "Phi_U1_s": phi1_mean,
                    "Phi_U2_s": phi2_mean,
                    "Phi_U2_over_U1": esc_ratio_mean,
                    "Phi_ball_in_DSMC_late_s": phi_ball_in_late,
                    "dN_ball_dt_DSMC_early_s": dn_ball_early,
                    "dN_ball_dt_DSMC_late_s": dn_ball_late,
                    "Phi_ball_ret_DSMC_late_s": phi_ret_late,
                    "ball_in_over_ret_DSMC_late": ratio_dsmc_late,
                    "ball_in_over_ret_Kepler_30ks": float(np.interp(30000.0, t, k_rat)),
                    "ball_in_over_ret_Kepler_100ks": float(np.interp(100000.0, t, k_rat)),
                    "ball_in_over_ret_Kepler_300ks": float(k_rat[-1]),
                    "N_ball_DSMC_300ks": float(dsmc_held[-1]),
                    "N_ball_Kepler_300ks": float(k_held[-1]),
                    "N_ball_Kepler_asymptotic": k_held_inf,
                    "held_fraction_of_asymptotic_300ks": float(dsmc_held[-1]) / k_held_inf if k_held_inf > 0 else np.nan,
                })

    # Convergence across z_top and dr_max at z <= 2000 km
    alt_grid = np.linspace(*CONV_ALT_KM[:2], CONV_ALT_KM[2])
    conv_z_rows = []
    for atm in ("H2", "H2O"):
        for ubc_num in (1, 2, 3):
            for dr in (1000, 250):
                z_levels = sorted([z for z in (2500, 5000, 10000, 20000, 40000)
                                   if f"{atm}_U{ubc_num}_z{z}_dr{dr}" in per_run])
                for z_lo, z_hi in zip(z_levels[:-1], z_levels[1:]):
                    a_lo = f"{atm}_U{ubc_num}_z{z_lo}_dr{dr}"
                    a_hi = f"{atm}_U{ubc_num}_z{z_hi}_dr{dr}"
                    d_lo, d_hi = per_run[a_lo], per_run[a_hi]
                    p_lo = d_lo[["profile_num_dens", f"profile_{T_FIELD}"]].rename(
                        {"profile_num_dens": "num_dens", f"profile_{T_FIELD}": T_FIELD})
                    p_hi = d_hi[["profile_num_dens", f"profile_{T_FIELD}"]].rename(
                        {"profile_num_dens": "num_dens", f"profile_{T_FIELD}": T_FIELD})
                    g_lo = pf.regrid(p_lo, alt_grid, fields=("num_dens", T_FIELD))
                    g_hi = pf.regrid(p_hi, alt_grid, fields=("num_dens", T_FIELD))
                    dn_rel = np.abs(g_lo["num_dens"].sel(species="H2").values / g_hi["num_dens"].sel(species="H2").values - 1.0)
                    dT_rel = np.abs(g_lo[T_FIELD].sel(species="H2").values / g_hi[T_FIELD].sel(species="H2").values - 1.0)
                    phi_lo = float(d_lo["dsmc_escape_rate"].sel(species="H2"))
                    phi_hi = float(d_hi["dsmc_escape_rate"].sel(species="H2"))
                    dphi_rel = abs(phi_lo / phi_hi - 1.0) if phi_hi > 0 else np.nan
                    conv_z_rows.append({
                        "comparison_type": "z_top_step",
                        "atmosphere": atm,
                        "ubc_num": ubc_num,
                        "ubc": {1: "reflect", 2: "ballistic", 3: "free_escape"}[ubc_num],
                        "dr_max_km": dr,
                        "z_top_nominal_km": np.nan,
                        "level_a": f"z{z_lo}",
                        "level_b": f"z{z_hi}",
                        "max_rel_dn_le_2000km": float(np.nanmax(dn_rel)),
                        "max_rel_dT_le_2000km": float(np.nanmax(dT_rel)),
                        "rel_diff_escape_H2": dphi_rel,
                        "pass_n_5pct": bool(np.nanmax(dn_rel) < CONV_THRESH["n"]),
                        "pass_T_2pct": bool(np.nanmax(dT_rel) < CONV_THRESH["T"]),
                        "pass_esc_10pct": bool(dphi_rel < CONV_THRESH["escape"]),
                    })

    # Resolution comparison dr_max 1000 vs 250 km at fixed (atm, ubc, z_top)
    for atm in ("H2", "H2O"):
        for ubc_num in (1, 2, 3):
            for z in (2500, 5000, 10000, 20000, 40000):
                a_1000 = f"{atm}_U{ubc_num}_z{z}_dr1000"
                a_250 = f"{atm}_U{ubc_num}_z{z}_dr250"
                if a_1000 not in per_run or a_250 not in per_run:
                    continue
                d_1000, d_250 = per_run[a_1000], per_run[a_250]
                p_1000 = d_1000[["profile_num_dens", f"profile_{T_FIELD}"]].rename(
                    {"profile_num_dens": "num_dens", f"profile_{T_FIELD}": T_FIELD})
                p_250 = d_250[["profile_num_dens", f"profile_{T_FIELD}"]].rename(
                    {"profile_num_dens": "num_dens", f"profile_{T_FIELD}": T_FIELD})
                g_1000 = pf.regrid(p_1000, alt_grid, fields=("num_dens", T_FIELD))
                g_250 = pf.regrid(p_250, alt_grid, fields=("num_dens", T_FIELD))
                dn_rel = np.abs(g_1000["num_dens"].sel(species="H2").values / g_250["num_dens"].sel(species="H2").values - 1.0)
                dT_rel = np.abs(g_1000[T_FIELD].sel(species="H2").values / g_250[T_FIELD].sel(species="H2").values - 1.0)
                phi_1000 = float(d_1000["dsmc_escape_rate"].sel(species="H2"))
                phi_250 = float(d_250["dsmc_escape_rate"].sel(species="H2"))
                dphi_rel = abs(phi_1000 / phi_250 - 1.0) if phi_250 > 0 else np.nan
                conv_z_rows.append({
                    "comparison_type": "dr_max_1000_vs_250",
                    "atmosphere": atm,
                    "ubc_num": ubc_num,
                    "ubc": {1: "reflect", 2: "ballistic", 3: "free_escape"}[ubc_num],
                    "dr_max_km": np.nan,
                    "z_top_nominal_km": z,
                    "level_a": "dr1000",
                    "level_b": "dr250",
                    "max_rel_dn_le_2000km": float(np.nanmax(dn_rel)),
                    "max_rel_dT_le_2000km": float(np.nanmax(dT_rel)),
                    "rel_diff_escape_H2": dphi_rel,
                    "pass_n_5pct": bool(np.nanmax(dn_rel) < CONV_THRESH["n"]),
                    "pass_T_2pct": bool(np.nanmax(dT_rel) < CONV_THRESH["T"]),
                    "pass_esc_10pct": bool(dphi_rel < CONV_THRESH["escape"]),
                })

    for a in per_run:
        sources.append(tiers.analysis_path(CAMPAIGN, a))
    sum_ds = xr.Dataset(series_vars)
    sum_ds.attrs.update({
        **tiers.analysis_attrs("h2_ubc_rtop: cross-run Carberry Mogan Fig. 3 ballistic and escape ratio series",
                               sources, "campaigns/h2_ubc_rtop/compute_analysis.py"),
        "profile_window_t0": PROFILE_WINDOW[0],
        "profile_window_t1": PROFILE_WINDOW[1],
        "n_completed_runs": len(per_run),
        "n_batches": N_BATCHES,
        "conv_alt_grid_km": np.linspace(*CONV_ALT_KM[:2], CONV_ALT_KM[2]),
        "conv_threshold_n": CONV_THRESH["n"],
        "conv_threshold_T": CONV_THRESH["T"],
        "conv_threshold_escape": CONV_THRESH["escape"],
    })
    return sum_ds, pd.DataFrame(ball_rows), pd.DataFrame(conv_z_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", nargs=2, type=float, default=list(PROFILE_WINDOW))
    args = parser.parse_args()
    window = (float(args.window[0]), float(args.window[1]))
    warnings.simplefilter("ignore", RuntimeWarning)

    tables_dir = output_dir(CAMPAIGN, "tables")
    card_df = build_card_inventory()
    card_df.to_csv(tables_dir / "card_inventory.csv", index=False)
    print(f"card_inventory.csv: {len(card_df)} completed run cards")

    aliases = [r.alias for r in registry.campaign(CAMPAIGN)
               if (r.path / f"{r.alias}{RUN_SUFFIX}").is_file()
               and (r.path.parent / "COMPLETE.ascii").is_file()]
    print(f"Analysing {len(aliases)} completed runs with *{RUN_SUFFIX} on window {window}:")

    per_run = {}
    conv_rows, esc_rows, cm_rows = [], [], []
    for alias in aliases:
        ds, c, e, cm = analyse_run(alias, window)
        out = tiers.analysis_path(CAMPAIGN, alias)
        ds.to_netcdf(out, engine="netcdf4")
        per_run[alias] = ds
        conv_rows += c
        esc_rows += e
        cm_rows += cm
        print(f"  {alias:22s} -> {out.name}")

    esc_df = pd.DataFrame(esc_rows)
    sum_ds, ball_df, ztop_df = compute_cross_run_and_ballistic_tables(per_run, esc_df)
    sum_out = tiers.analysis_path(CAMPAIGN, CAMPAIGN)
    sum_ds.to_netcdf(sum_out, engine="netcdf4")

    pd.DataFrame(conv_rows).to_csv(tables_dir / "convergence_summary.csv", index=False)
    esc_df.to_csv(tables_dir / "escape_summary.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(tables_dir / "carberry_exobase_table1.csv", index=False)
    ball_df.to_csv(tables_dir / "ballistic_rates_summary.csv", index=False)
    ztop_df.to_csv(tables_dir / "ztop_resolution_convergence.csv", index=False)
    print(f"Wrote {sum_out} and 6 CSV tables in {tables_dir}")


if __name__ == "__main__":
    main()
