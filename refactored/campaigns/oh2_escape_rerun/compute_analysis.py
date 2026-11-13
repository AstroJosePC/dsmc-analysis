"""O+H2 reruns: convergence, profiles, escape and wave response -> analysis files + tables.

    python campaigns/oh2_escape_rerun/compute_analysis.py            # needs the run files (compile_netcdf.py)
    python campaigns/oh2_escape_rerun/compute_analysis.py --window 6849 9849

Writes outputs/oh2_escape_rerun/analysis/<alias>_analysis.nc, oh2_escape_rerun_analysis.nc (cross-run) and
outputs/oh2_escape_rerun/tables/{convergence,escape,wave}_summary.csv.

Analysis choices (all recorded as attributes):
- Pre-forcing record: up to the wave start (OH2_A) or the last output (B runs).
- Steady state: per species, on the domain inventory, with convergence.steady_window
  (95% upper bound of the relative drift over the segment < 5%).
- Profile window: the same absolute window for every run (default: the 3000 s before OH2_A's
  wave start), so A, B and the B replicates are compared at the same stage of H2 filling.
- Escape counting window: (spin-up end, end of pre-forcing record], spin-up end = 1000 s.
- Wave background: the profile window (pre-forcing, O steady).
- Temperature: temp_trans_corr = temp_trans - m vel_rad^2/(3k) (dsmclib.profiles.temp_trans_corrected, attached here) for the exobase
  Jeans estimate and the wave dT; the solver's 1-D temp_trans counts radial bulk flow as heat.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from dsmclib import convergence as cv
from dsmclib import escape as es
from dsmclib import profiles as pf
from dsmclib import registry, tiers, wave
from dsmclib.ncrun import open_run_nc
from dsmclib.paths import output_dir

CAMPAIGN = "oh2_escape_rerun"
SPINUP = 1000.0  # s, excluded from escape counting and filling fits
PROFILE_SPAN = 3000.0  # s
T_FIELD = "temp_trans_corr"  # bulk-flow corrected translational temperature


def analyse_run(alias: str, window: tuple[float, float]) -> tuple[xr.Dataset, list, list, list]:
    run = open_run_nc(alias)
    run.macro[T_FIELD] = pf.temp_trans_corrected(run)  # analysis tier: the run file carries only solver output
    t_pre_end = run.wave_start_time or run.t_end
    inv = cv.inventory(run)
    rates = cv.boundary_rates(run)
    ds = xr.Dataset({"N_phys": inv.N_phys, "N_part": inv.N_part})
    for v in ("inject_rate", "return_rate", "escape_rate", "escape_npart", "dN_dt"):
        if v in rates:
            ds[v] = rates[v]

    conv_rows = []
    steady_vars = {k: [] for k in ("steady", "t_steady_start", "rel_drift_ub", "drift_per_ks",
                                   "fill_tau", "fill_tau_err", "fill_N_inf", "fill_N_inf_err", "fill_tau_over_span")}
    for sp in run.species:
        r = cv.steady_window(inv.N_phys.sel(species=sp), t_end=t_pre_end)
        f = cv.filling_timescale(inv.N_phys.sel(species=sp), SPINUP, t_pre_end)
        steady_vars["steady"].append(int(r.steady))
        steady_vars["t_steady_start"].append(np.nan if r.t_start is None else r.t_start)
        steady_vars["rel_drift_ub"].append(r.rel_drift_ub)
        steady_vars["drift_per_ks"].append(r.drift_per_ks)
        steady_vars["fill_tau"].append(f["tau"])
        steady_vars["fill_tau_err"].append(f["tau_err"])
        steady_vars["fill_N_inf"].append(f["N_inf"])
        steady_vars["fill_N_inf_err"].append(f["N_inf_err"])
        steady_vars["fill_tau_over_span"].append(f["tau_over_span"])
        conv_rows.append({"run": alias, "species": sp, "steady": r.steady, "t_steady_start_s": r.t_start,
                          "drift_pct_per_ks": 100 * r.drift_per_ks, "rel_drift_ub": r.rel_drift_ub,
                          "N_phys_end": float(inv.N_phys.sel(species=sp).sel(time=t_pre_end, method="nearest")),
                          "fill_tau_s": f["tau"], "fill_tau_err_s": f["tau_err"],
                          "fill_tau_over_span": f["tau_over_span"], "t_pre_end_s": t_pre_end})
    for k, v in steady_vars.items():
        ds[k] = ("species", np.array(v, dtype=float))
    ds["steady"].attrs["comment"] = "1 if the 95% upper bound of the relative inventory drift on [t_steady_start, pre-forcing end] < 5%"
    ds["fill_tau"].attrs["comment"] = ("e-folding time of N(t) = N_inf - A exp(-t/tau) fitted after spin-up; "
                                      "unconstrained when fill_tau_over_span >> 1")

    prof = pf.time_mean_profiles(run, *window)
    for v in prof.data_vars:
        ds[f"profile_{v}"] = prof[v]
    n_iso = pf.isothermal_reference(run, prof)
    ds["profile_num_dens_isothermal"] = n_iso
    ds["profile_num_dens_ratio_isothermal"] = (prof["num_dens"] / n_iso).assign_attrs(units="1")

    jexo = es.jeans_at_exobase(run, prof, temperature_field=T_FIELD)
    for v in jexo.data_vars:
        ds[v] = jexo[v]
    jlb = es.jeans_at_lower_boundary(run)
    ds["jeans_rate_lower_boundary"] = jlb["jeans_rate"].assign_attrs(units="s-1")
    esc = es.dsmc_escape(run, SPINUP, t_pre_end)
    for v in esc.data_vars:
        ds[f"dsmc_{v}"] = esc[v]
    bal = es.lower_boundary_balance(run, *window)
    for v in bal.data_vars:
        ds[f"balance_{v}"] = bal[v]

    # Expected top-boundary count for the exobase Jeans rate, with the exobase-to-top travel time
    # (radial launches, constant rate from t = 0: an upper bound on the count)
    transit = {k: [] for k in ("transit_expected_npart", "transit_median_travel_time", "transit_min_travel_time",
                               "transit_p_zero")}
    for sp in run.species:
        rate = float(jexo.jeans_rate.sel(species=sp))
        if not np.isfinite(rate):
            for v in transit.values():
                v.append(np.nan)
            continue
        tr = es.transit_expected_count(rate, float(run.weight.sel(species=sp)), float(jexo.T_exobase.sel(species=sp)),
                                       float(run.mass.sel(species=sp)),
                                       run.planet_radius + 1e3 * float(jexo.exobase_altitude.sel(species=sp)),
                                       esc.attrs["top_radius"], run.planet_mass, 0.0, esc.attrs["t0"], esc.attrs["t1"])
        transit["transit_expected_npart"].append(tr["expected_npart"])
        transit["transit_median_travel_time"].append(tr["median_travel_time"])
        transit["transit_min_travel_time"].append(tr["min_travel_time"])
        transit["transit_p_zero"].append(tr["p_zero"])
    for k, v in transit.items():
        ds[k] = ("species", np.array(v, dtype=float))
    ds["transit_expected_npart"].attrs.update(
        long_name="Escaping computational particles expected at the top boundary in the escape window for the exobase Jeans rate",
        comment="Flux-weighted Maxwellian above escape speed, radial trajectories, rate constant from t = 0 (upper bound)")
    ds["transit_median_travel_time"].attrs["units"] = "s"
    ds["transit_min_travel_time"].attrs["units"] = "s"

    esc_rows = []
    for sp in run.species:
        esc_rows.append({
            "run": alias, "species": sp,
            "exobase_km": float(jexo.exobase_altitude.sel(species=sp)),
            "n_exobase_m3": float(jexo.n_exobase.sel(species=sp)),
            "T_exobase_K": float(jexo.T_exobase.sel(species=sp)),
            "jeans_lambda_exobase": float(jexo.jeans_parameter.sel(species=sp)),
            "jeans_rate_exobase_s": float(jexo.jeans_rate.sel(species=sp)),
            "jeans_rate_lower_boundary_s": float(jlb.jeans_rate.sel(species=sp)),
            "dsmc_escape_npart": int(esc.escape_npart.sel(species=sp)),
            "dsmc_escape_rate_s": float(esc.escape_rate.sel(species=sp)),
            "dsmc_escape_ul95_s": float(esc.escape_rate_ul95.sel(species=sp)),
            "dsmc_one_part_per_window_s": float(esc.escape_rate_resolution.sel(species=sp)),
            "expected_npart_at_jeans_exobase": float(jexo.jeans_rate.sel(species=sp) * esc.attrs["duration"]
                                                    / run.weight.sel(species=sp)),
            "transit_expected_npart": float(ds.transit_expected_npart.sel(species=sp)),
            "transit_median_travel_time_s": float(ds.transit_median_travel_time.sel(species=sp)),
            "transit_p_zero": float(ds.transit_p_zero.sel(species=sp)),
            "balance_outflow_s": float(bal.outflow.sel(species=sp)),
            "balance_outflow_se_s": float(bal.outflow_se.sel(species=sp)),
            "escape_window_s": f"{esc.attrs['t0']:.0f}-{esc.attrs['t1']:.0f}",
        })

    wave_rows = []
    fw = wave.forcing_window(run)
    if fw is not None and fw["n_periods"] >= 1:
        resp = wave.forcing_response(run, background=window, fields=("num_dens", T_FIELD, "vel_rad"))
        for v in resp.data_vars:
            ds[f"wave_{v}"] = resp[v]
        ds.attrs.update({f"wave_{k}": v for k, v in resp.attrs.items() if k != "run"})
        alt = resp["altitude"].values / 1e3
        for sp in run.species:
            for i in np.where(resp["resolved"].sel(species=sp).values)[0]:
                wave_rows.append({"run": alias, "species": sp, "altitude_km": alt[i],
                                  "dn_amp": float(resp.num_dens_amp.sel(species=sp)[i]),
                                  "dn_amp_se": float(resp.num_dens_amp_se.sel(species=sp)[i]),
                                  "dn_phase_rel_forcing_rad": float(resp.num_dens_phase_rel_forcing.sel(species=sp)[i]),
                                  "dT_amp": float(resp[f"{T_FIELD}_amp"].sel(species=sp)[i]),
                                  "dT_amp_se": float(resp[f"{T_FIELD}_amp_se"].sel(species=sp)[i]),
                                  "dv_amp_ms": float(resp.vel_rad_amp.sel(species=sp)[i]),
                                  "dv_amp_se_ms": float(resp.vel_rad_amp_se.sel(species=sp)[i]),
                                  "dv_phase_rel_forcing_rad": float(resp.vel_rad_phase_rel_forcing.sel(species=sp)[i]),
                                  "dn_null_amp": float(resp.num_dens_null_amp.sel(species=sp)[i]),
                                  "dT_null_amp": float(resp[f"{T_FIELD}_null_amp"].sel(species=sp)[i]),
                                  "dv_null_amp_ms": float(resp.vel_rad_null_amp.sel(species=sp)[i]),
                                  "background_npart": float(resp.background_num_part.sel(species=sp)[i])})

    ds = ds.assign_coords(species_mass=run.mass, particle_weight=run.weight)
    ds.attrs.update({
        **tiers.analysis_attrs(f"{alias}: O+H2 1-D DSMC convergence, profiles, escape and wave response",
                               [run.path], "compute_analysis.py"),
        "run_alias": alias,
        "source_commit": str(run.attrs.get("source_commit", "")),
        "source_branch": str(run.attrs.get("source_branch", "")),
        "builder_version": str(run.attrs.get("builder_version", "")),
        "profile_window_t0": window[0], "profile_window_t1": window[1],
        "pre_forcing_end": t_pre_end, "spinup_excluded": SPINUP,
        "escape_window_t0": esc.attrs["t0"], "escape_window_t1": esc.attrs["t1"],
        "top_radius": esc.attrs["top_radius"], "planet_radius": run.planet_radius, "planet_mass": run.planet_mass,
        "isothermal_temperature": n_iso.attrs["temperature"],
    })
    return ds, conv_rows, esc_rows, wave_rows


def summary(per_run: dict, b_aliases: list, a_alias: str) -> xr.Dataset:
    """B replicate statistics on the B grid, and A vs B-mean on B's cell centres."""
    fields = ("num_dens", "temp_trans", T_FIELD)

    def prof(alias):
        return per_run[alias][[f"profile_{f}" for f in fields]].rename({f"profile_{f}": f for f in fields})

    rep = pf.replicate_stats([prof(a) for a in b_aliases], fields=fields)
    rep = rep.rename({v: f"B_{v}" for v in rep.data_vars})
    alt_b = per_run[b_aliases[0]]["altitude"].values / 1e3
    a_on_b = pf.regrid(prof(a_alias), alt_b, fields=fields)
    out = rep.assign_coords(altitude_km=("radial", alt_b))
    for f in fields:
        out[f"A_{f}_on_B"] = (("species", "radial"), a_on_b[f].values)
    out["A_over_B_num_dens"] = out["A_num_dens_on_B"] / out["B_num_dens_mean"]
    out["A_minus_B_num_dens_nsigma"] = (out["A_num_dens_on_B"] - out["B_num_dens_mean"]) / out["B_num_dens_std"]
    out.attrs.update({**tiers.analysis_attrs("oh2_escape_rerun: B replicate statistics and A vs B",
                                             [tiers.analysis_path(CAMPAIGN, a) for a in b_aliases + [a_alias]],
                                             "compute_analysis.py"),
                      "B_runs": ",".join(b_aliases), "A_run": a_alias,
                      "comment": ("A is cell-averaged on its own 128-cell grid and log-interpolated to B cell "
                                  "centres; B cells above ~240 km are thousands of km wide, so A/B there "
                                  "compares a point value with a wide cell average.")})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=[r.alias for r in registry.campaign(CAMPAIGN)])
    parser.add_argument("--window", nargs=2, type=float, default=None,
                        help="profile/background window [s]; default: 3000 s before OH2_A's wave start")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    if args.window is None:
        t_wave = open_run_nc("OH2_A").wave_start_time
        window = (t_wave - PROFILE_SPAN, t_wave)
    else:
        window = tuple(args.window)
    print(f"profile window: {window[0]:.1f}-{window[1]:.1f} s", flush=True)

    tables = output_dir(CAMPAIGN, "tables")
    per_run, conv, escr, waver = {}, [], [], []
    for alias in args.runs:
        ds, c, e, w = analyse_run(alias, window)
        out = tiers.analysis_path(CAMPAIGN, alias)
        ds.to_netcdf(out, engine="netcdf4")
        per_run[alias] = ds
        conv += c
        escr += e
        waver += w
        print(f"{alias}: wrote {out}", flush=True)

    b_runs = [a for a in args.runs if a.startswith("OH2_B")]
    if "OH2_A" in per_run and len(b_runs) >= 2:
        s = summary(per_run, b_runs, "OH2_A")
        s.attrs.update(profile_window_t0=window[0], profile_window_t1=window[1])
        out = tiers.analysis_path(CAMPAIGN, CAMPAIGN)
        s.to_netcdf(out, engine="netcdf4")
        print(f"wrote {out}")

    pd.DataFrame(conv).to_csv(tables / "convergence_summary.csv", index=False)
    pd.DataFrame(escr).to_csv(tables / "escape_summary.csv", index=False)
    pd.DataFrame(waver).to_csv(tables / "wave_summary.csv", index=False)
    print(f"tables in {tables}")
