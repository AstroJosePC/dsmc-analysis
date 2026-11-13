"""h2_hold analysis: Kepler hold (UBC 2) vs explicit RK4 ballistic tracking (UBC 5) vs reflect (UBC 1), and the
collisionless E7-E9 verification runs (docs/reports/ubc_hold_carberry).

    python -B campaigns/h2_hold/compute_analysis.py

Writes (docs/CONVENTIONS.md, analysis tier):
- outputs/h2_hold/analysis/<alias>_analysis.nc   one per finished run (registry campaign h2_hold, incl. verification):
    profiles over the run's window, escape / held / inventory series, the BallisticHold diagnostic series
    (dim hold_time), escape over the window, wall time; verification runs E7a/E7b/E9 also carry the
    cell-averaged collisionless analytic profile of their UBC.
- outputs/h2_hold/analysis/h2_hold_analysis.nc   cross-run: collisionless references (escape vs r_top, held(t),
    tau90, longest return per z_top) and the E7-E9 verification metrics (attribute verification_json).
- outputs/h2_hold/tables/{run_inventory,verification_summary}.csv

Windows: verification runs use t > T0_VER; production runs the trailing steady window of the H2 inventory
(convergence.steady_window), else the second half of the run. The BallisticHold files (raw tier, one per output)
are traced by count, glob and a combined SHA-256 (attributes hold_source_*), the run file by source_files/sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dsmclib import convergence, escape, exosphere, profiles, registry, tiers
from dsmclib.constants import G_NEWTON, K_BOLTZMANN
from dsmclib.ncrun import RUN_SUFFIX, open_run_nc
from dsmclib.paths import output_dir

warnings.simplefilter("ignore", RuntimeWarning)

CAMPAIGN = "h2_hold"
SCRIPT = "campaigns/h2_hold/compute_analysis.py"
SP = "H2"
T0_VER = 3.0e4                  # verification steady-window start (MarsAtmos verification/evaluate_hold.py --t0)
RESTART_WIN = (1.5e4, 3.0e4)    # E7b vs E7d restart comparison window
NV_HOLD = 2500                  # velocity grid of the collisionless held count / escape at each z_top
PROFILE_FIELDS = ("num_dens", "num_part", "temp_trans", "temp_trans_corr", "vel_rad", "mean_free_path",
                  "scale_height", "knudsen")
SWEEP_MATRIX = Path.home() / "MarsAtmos-H2UBC-sweep" / "matrix.csv"   # z_top values of the sweep (references)
HOLD_MATRIX = Path.home() / "MarsAtmos-H2Hold-campaign" / "matrix.csv"
UBC_CODE = {"UBC_REFLECT": 1, "UBC_BALLISTIC": 2, "UBC_FREE_ESCAPE": 3, "UBC_BALLISTIC_EXPLICIT": 5}
VERIF = {  # alias -> (UBC code, role)
    "E7a_kepler": (2, "Kepler hold, fixed binary"),
    "E7b_explicit": (5, "explicit RK4, fixed binary"),
    "E7c_explicit_dbg": (5, "explicit RK4, -fcheck=all build"),
    "E7d_explicit_restart": (5, "explicit RK4, restarted at 15 ks"),
    "E8_main_kepler": (2, "Kepler hold, main (bug 2)"),
    "E9_reflect": (1, "reflect, fixed binary"),
}
ANALYTIC_UBC = {"E7a_kepler": "ballistic", "E7b_explicit": "ballistic", "E9_reflect": "reflect"}


# ------------------------------------------------------------------ BallisticHold diagnostics (raw tier reader)
# Same format and logic as MarsAtmos verification/hold_diagnostics.py (feature/ubc-hold-explicit, 1206efc6).
HOLD_COLUMNS = ["time", "avg_time_interval", "species", "part_weight", "n_hold_in", "n_hold_ret",
                "n_esc_direct", "n_esc_orbit", "held_end", "held_mean"]
HOLD_EXTRA = ["ret_dt_sum", "ret_dt_sumsq", "r_held_max"]
HOLD_COUNTS = ["n_hold_in", "n_hold_ret", "n_esc_direct", "n_esc_orbit"]
_HOLD_NAME = re.compile(r"Output(\d+)_Proc(\d+)_BallisticHold\.ascii$")


def hold_files(results: Path) -> list[Path]:
    d = Path(results) / "Diagnostics"
    fs = [f for f in d.glob("Output*_Proc*_BallisticHold.ascii") if _HOLD_NAME.search(f.name)] if d.is_dir() else []
    return sorted(fs, key=lambda f: tuple(int(g) for g in _HOLD_NAME.search(f.name).groups()))


def read_hold(files: list[Path]) -> xr.Dataset:
    """Dataset(time, species) of the per-interval hold counters; rows of several ranks summed per output."""
    by_out = {}
    for f in files:
        by_out.setdefault(int(_HOLD_NAME.search(f.name).group(1)), []).append(f)
    out_idx = sorted(by_out)
    tables = []
    for n in out_idx:
        parts = [np.atleast_2d(np.loadtxt(f, comments="#")) for f in sorted(by_out[n])]
        tab = parts[0].copy()
        for p in parts[1:]:
            tab[:, 4:12] += p[:, 4:12]
            if tab.shape[1] > 12:
                tab[:, 12] = np.maximum(tab[:, 12], p[:, 12])
        if tab.shape[1] < len(HOLD_COLUMNS) + len(HOLD_EXTRA):
            tab = np.hstack([tab, np.zeros((tab.shape[0], len(HOLD_COLUMNS) + len(HOLD_EXTRA) - tab.shape[1]))])
        tables.append(tab)
    arr = np.stack(tables)
    col = {c: arr[:, :, i] for i, c in enumerate(HOLD_COLUMNS + HOLD_EXTRA)}
    dims = ("time", "species")
    data = {"output": ("time", np.array(out_idx)),
            "avg_time_interval": ("time", col["avg_time_interval"][:, 0], {"units": "s"}),
            "part_weight": ("species", col["part_weight"][0]),
            "held_end": (dims, np.rint(col["held_end"]).astype(np.int64)),
            "held_mean": (dims, col["held_mean"])}
    for c in HOLD_COUNTS:
        data[c] = (dims, np.rint(col[c]).astype(np.int64))
    n_ret = col["n_hold_ret"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_ret > 0, col["ret_dt_sum"] / n_ret, np.nan)
        var = np.where(n_ret > 0, col["ret_dt_sumsq"] / n_ret - mean**2, np.nan)
    data["ret_dt_mean"] = (dims, mean, {"units": "s"})
    data["ret_dt_std"] = (dims, np.sqrt(np.maximum(var, 0.0)), {"units": "s"})
    data["r_held_max"] = (dims, col["r_held_max"], {"units": "m"})
    return xr.Dataset(data, coords={"time": col["time"][:, 0], "species": col["species"][0].astype(int)})


def hold_invariant(ds: xr.Dataset) -> float:
    """Max |Delta held_end - (n_hold_in - n_hold_ret)| over contiguous intervals (0 = exact hold balance)."""
    held = ds["held_end"].values.astype(np.int64)
    net = (ds["n_hold_in"] - ds["n_hold_ret"]).values.astype(np.int64)
    out = ds["output"].values
    viol = [np.zeros(1, dtype=np.int64)]
    if out[0] == 1:
        viol.append(np.abs(held[0] - net[0]))
    d = np.diff(held, axis=0) - net[1:]
    viol.append(np.abs(d[np.diff(out) == 1]))
    return float(max(np.max(v) for v in viol if np.size(v)))


def combined_sha256(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ helpers
def read_meta(results: Path) -> dict:
    f = Path(results).parent / "run_meta.txt"
    return dict(re.findall(r"^(\w+):\s+(.*)$", f.read_text(), re.M)) if f.exists() else {}


def wall_seconds(meta: dict) -> float:
    try:
        a, b = (datetime.fromisoformat(meta[k].strip()) for k in ("started", "finished"))
    except (KeyError, ValueError):
        return np.nan
    return (b - a).total_seconds()


def longest_return_time(r_top, r_hill, mu):
    """Time of flight above r_top of the slowest returning orbit: radial, apoapsis at r_hill (a = r_hill / 2)."""
    a = r_hill / 2
    E_top = np.arccos(1 - r_top / a)
    return float(2 * np.sqrt(a**3 / mu) * (np.pi - (E_top - np.sin(E_top))))


def _attr(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def write(ds: xr.Dataset, path: Path):
    ds.attrs = {k: _attr(v) for k, v in ds.attrs.items() if v is not None}
    for v in ds.variables.values():
        v.attrs = {k: _attr(x) for k, x in v.attrs.items() if x is not None}
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(path)


def run_path(r: registry.Run) -> Path:
    return r.path / f"{r.alias}{RUN_SUFFIX}"


def open_run(r: registry.Run):
    run = open_run_nc(run_path(r))
    run.macro["temp_trans_corr"] = profiles.temp_trans_corrected(run)
    return run


def source_params(run) -> dict:
    """Lower-boundary parameters (n0, T0, lambda0, phi_J0, r0, m, M) of an H2 run."""
    s = run.info_surface
    n0 = float(s["num_dens"].sel(species=SP).isel(surface_cell=0))
    T0 = float(s["T0"].values.ravel()[0])
    m = float(run.mass.sel(species=SP))
    r0 = float(run.radial_edges()[0])
    M = run.planet_mass
    return {"n0": n0, "T0": T0, "m": m, "r0": r0, "M": M, "R": run.planet_radius,
            "lambda0": G_NEWTON * M * m / (K_BOLTZMANN * T0 * r0),
            "phiJ0": exosphere.jeans_rate(n0, T0, m, r0, M),
            "phiJ0_info": float(s["jeans_escape_rate"].sel(species=SP).isel(surface_cell=0)),
            "lambda0_info": float(s["jeans_parameter"].sel(species=SP).isel(surface_cell=0)),
            "W": float(run.weight.sel(species=SP)), "dt": run.dt}


def cell_avg_analytic(run, sp: dict, ubc: str):
    """Collisionless profile of `ubc` averaged over each cell (volume-weighted n, density-weighted T)."""
    gx, gw = np.polynomial.legendre.leggauss(4)
    edges = run.radial_edges()
    m = float(run.mass.sel(species=SP))
    r0, rt = edges[0], edges[-1]
    n_c, T_c = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rr = 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)
        w = gw * rr**2
        p = exosphere.profile(rr, sp["n0"], sp["T0"], m, r0, rt, ubc=ubc, n_v=500, planet_mass=run.planet_mass)
        n = p.num_dens.values
        n_c.append((w * n).sum() / w.sum())
        T_c.append((w * n * p.temp_trans.values).sum() / (w * n).sum())
    return np.array(n_c), np.array(T_c)


# ------------------------------------------------------------------ per-run analysis
def analyse_run(r: registry.Run, sp_ref: dict | None) -> xr.Dataset:
    run = open_run(r)
    t = run.time
    verification = r.role == "verification"
    inv = convergence.inventory(run).N_phys.sel(species=SP)
    try:
        sw = convergence.steady_window(inv)
        steady, t_start, drift = bool(sw.steady), sw.t_start, sw.drift_per_ks
    except Exception:  # noqa: BLE001  (too few outputs for the drift test)
        steady, t_start, drift = False, None, np.nan
    t0 = t_start if (steady and t_start is not None) else float(t[-1]) / 2
    window_rule = "steady_window" if (steady and t_start is not None) else "second_half"
    if verification:
        t0, window_rule = T0_VER, "fixed_T0_VER"
    if (t > t0).sum() < 20:
        t0 = float(t[max(0, len(t) - 20) - 1]) if len(t) > 20 else float(t[0]) - 1
        window_rule += "+min20"
    prof = profiles.time_mean_profiles(run, t0, fields=PROFILE_FIELDS).sel(species=SP)
    e = escape.dsmc_escape(run, t0)
    esc = run.escape
    dtav = run.macro["avg_time_interval"].values
    esc_rate = esc["escape_rate"].sum("boundary_cell").sel(species=SP).values
    W = float(run.weight.sel(species=SP))
    held = (esc["num_part_ball"].sum("boundary_cell").sel(species=SP).values * W
            if "num_part_ball" in esc else np.full(len(t), np.nan))
    surf = run.surface
    data = {f"profile_{k}": ("radial", prof[k].values) for k in PROFILE_FIELDS}
    data |= {f"profile_{k}_sem": ("radial", prof[k + "_sem"].values) for k in PROFILE_FIELDS}
    data |= {"altitude": ("radial", prof["altitude"].values, {"units": "m"}),
             "escape_rate": ("time", esc_rate, {"units": "s-1"}),
             "escape_npart": ("time", esc_rate * dtav / W, {"units": "1"}),
             "N_ball_phys": ("time", held, {"units": "1", "long_name": "Physical molecules held above r_top"}),
             "N_phys": ("time", inv.values, {"units": "1"}),
             "avg_time_interval": ("time", dtav, {"units": "s"}),
             "inject_rate": ("time", surf["inject_rate"].sum("boundary_cell").sel(species=SP).values, {"units": "s-1"}),
             "return_rate": ("time", surf["return_rate"].sum("boundary_cell").sel(species=SP).values, {"units": "s-1"})}
    ds = xr.Dataset(data, coords={"time": t, "radial": prof["radial"].values})
    ubc_attr = str(run.attrs.get("upper_boundary_condition", ""))
    meta = read_meta(r.path)
    ds.attrs.update(tiers.analysis_attrs(f"{r.alias}: h2_hold 1-D DSMC profiles, escape and ballistic hold",
                                         [run_path(r)], SCRIPT))
    ds.attrs.update(run_alias=r.alias, role=r.role, species=SP, ubc_attr=ubc_attr,
                    ubc=UBC_CODE.get(ubc_attr.strip(), VERIF.get(r.alias, (0,))[0]),
                    window_t0=t0, window_t1=float(t[-1]), window_rule=window_rule, steady=int(steady),
                    steady_t_start=t_start if t_start is not None else np.nan, drift_per_ks=drift,
                    dsmc_escape_rate=float(e.escape_rate.sel(species=SP)),
                    dsmc_escape_npart=int(e.escape_npart.sel(species=SP)),
                    dsmc_escape_rate_ul95=float(e.escape_rate_ul95.sel(species=SP)),
                    r0_m=float(run.radial_edges()[0]), r_top_m=float(run.radial_edges()[-1]),
                    particle_weight=W, species_mass=float(run.mass.sel(species=SP)),
                    planet_mass=run.planet_mass, planet_radius=run.planet_radius,
                    held_mean_window=float(np.nanmean(held[t > t0])), n_cells=int(prof.sizes["radial"]),
                    run_started=meta.get("started", ""), run_finished=meta.get("finished", ""),
                    wall_s=wall_seconds(meta), dt=run.dt)
    if r.alias in ANALYTIC_UBC and sp_ref is not None:
        na, Ta = cell_avg_analytic(run, sp_ref, ANALYTIC_UBC[r.alias])
        ds["analytic_num_dens"] = ("radial", na, {"units": "m-3"})
        ds["analytic_temp_trans"] = ("radial", Ta, {"units": "K"})
        ds.attrs["analytic_ubc"] = ANALYTIC_UBC[r.alias]
    files = hold_files(r.path)
    if files:
        h = read_hold(files)
        viol = hold_invariant(h)
        h = h.isel(species=-1).drop_vars("species")
        ds.attrs["hold_part_weight"] = float(h.part_weight)
        h = h.drop_vars("part_weight").rename({"time": "hold_time"})
        ds = ds.merge(h.rename({v: f"hold_{v}" for v in h.data_vars}))
        ds.attrs.update(hold_invariant_violation=viol, hold_source_glob=str(files[0].parent / "Output*_Proc*_BallisticHold.ascii"),
                        hold_source_count=len(files), hold_source_sha256_combined=combined_sha256(files))
    return ds


# ------------------------------------------------------------------ cross-run
def analytic_references(sp: dict, tops_km: list[float]) -> xr.Dataset:
    n0, T0, m, r0, M = sp["n0"], sp["T0"], sp["m"], sp["r0"], sp["M"]
    x = np.geomspace(1.25, 14.0, 34)
    esc = {u: np.array([exosphere.escape_rate(n0, T0, m, r0, xi * r0, ubc=u, planet_mass=M, n_v=900) for xi in x])
           for u in ("reflect", "ballistic", "free_escape")}
    tgrid = np.geomspace(1e2, 1e9, 141)
    ztops = np.array(sorted(tops_km))
    held = np.zeros((len(ztops), len(tgrid)))
    steady = np.zeros(len(ztops))
    tau90 = np.zeros(len(ztops))
    tret = np.zeros(len(ztops))
    esc_ztop = {u: np.zeros(len(ztops)) for u in ("reflect", "ballistic", "free_escape")}
    for i, z in enumerate(ztops):
        rt = sp["R"] + z * 1e3
        held[i] = exosphere.held_count(n0, T0, m, r0, rt, planet_mass=M, t=tgrid, n_v=NV_HOLD)
        steady[i] = exosphere.held_count(n0, T0, m, r0, rt, planet_mass=M, n_v=NV_HOLD)
        frac = held[i] / steady[i]
        k = int(np.argmax(frac >= 0.9))
        tau90[i] = float(np.exp(np.interp(0.9, frac[k - 1:k + 1], np.log(tgrid[k - 1:k + 1]))))
        tret[i] = longest_return_time(rt, exosphere.MARS_HILL, G_NEWTON * M)
        for u in esc_ztop:
            esc_ztop[u][i] = exosphere.escape_rate(n0, T0, m, r0, rt, ubc=u, planet_mass=M, n_v=NV_HOLD)
    ds = xr.Dataset({f"esc_{u}": ("x", v, {"units": "s-1"}) for u, v in esc.items()}
                    | {"held_t": (("ztop", "t"), held), "held_steady": ("ztop", steady),
                       "tau90": ("ztop", tau90, {"units": "s"}), "t_return_max": ("ztop", tret, {"units": "s"})}
                    | {f"escz_{u}": ("ztop", v, {"units": "s-1"}) for u, v in esc_ztop.items()},
                    coords={"x": ("x", x, {"long_name": "r_top / r0"}), "t": ("t", tgrid, {"units": "s"}),
                            "ztop": ("ztop", ztops, {"units": "km"})})
    ds.attrs.update({f"source_{k}": v for k, v in sp.items()}, hill_m=exosphere.MARS_HILL, nv_hold=NV_HOLD)
    return ds


def verification_metrics(red: dict, sp: dict) -> dict:
    """E7-E9 checks from the per-run analysis datasets (keys = verification aliases)."""
    out = {"analytic": {}, "hold": {}}
    for name, ubc in ANALYTIC_UBC.items():
        d = red.get(name)
        if d is None:
            continue
        n, nse, T = d.profile_num_dens.values, d.profile_num_dens_sem.values, d.profile_temp_trans_corr.values
        na, Ta = d.analytic_num_dens.values, d.analytic_temp_trans.values
        good = d.profile_num_part.values > 5
        esc, k = d.attrs["dsmc_escape_rate"], d.attrs["dsmc_escape_npart"]
        rt = d.attrs["r_top_m"]
        esc_an = exosphere.escape_rate(sp["n0"], sp["T0"], sp["m"], d.attrs["r0_m"], rt, ubc=ubc,
                                       planet_mass=d.attrs["planet_mass"])
        tt = d.time.values
        held = float(d.N_ball_phys.values[tt > T0_VER].mean())
        held_an = float(np.mean(exosphere.held_count(sp["n0"], sp["T0"], sp["m"], d.attrs["r0_m"], rt,
                                                     planet_mass=d.attrs["planet_mass"], t=tt[tt > T0_VER]))) \
            if ubc == "ballistic" else np.nan
        out["analytic"][name] = {
            "dn_max": float(np.nanmax(np.abs((n - na) / na)[good])),
            "zn_max": float(np.nanmax(np.abs((n - na) / np.maximum(nse, 1e-300))[good])),
            "dT_max": float(np.nanmax(np.abs((T - Ta) / Ta)[good])),
            "esc": esc, "k": k, "esc_an": esc_an, "esc_z": (esc - esc_an) / (esc_an / np.sqrt(max(k, 1))),
            "held": held, "held_an": held_an, "r_top_km": (rt - d.attrs["planet_radius"]) / 1e3,
            "n_cells": int(d.sizes["radial"]), "t_end": float(tt[-1])}
    if "E7a_kepler" in red and "E7b_explicit" in red:
        pa, pb = red["E7a_kepler"], red["E7b_explicit"]
        z = (pa.profile_num_dens - pb.profile_num_dens) / np.hypot(pa.profile_num_dens_sem, pb.profile_num_dens_sem)
        out["kepler_vs_explicit"] = {"zmax": float(np.abs(z).max()),
                                     "held_ratio": out["analytic"]["E7b_explicit"]["held"]
                                     / out["analytic"]["E7a_kepler"]["held"]}
    return out


def restart_metrics(rb, rd) -> dict:
    """E7b vs E7d (restarted at 15 ks) over RESTART_WIN, from the run files (window profiles, 5 batches)."""
    pa = profiles.time_mean_profiles(rb, *RESTART_WIN, fields=("num_dens",), n_batches=5).sel(species=SP)
    pd_ = profiles.time_mean_profiles(rd, *RESTART_WIN, fields=("num_dens",), n_batches=5).sel(species=SP)
    z = float(np.abs((pa.num_dens - pd_.num_dens) / np.hypot(pa.num_dens_sem, pd_.num_dens_sem)).max())

    def held_win(run):
        s = run.escape["num_part_ball"].sum("boundary_cell").sel(species=SP).values * float(run.weight.sel(species=SP))
        t = run.time
        return float(s[(t > RESTART_WIN[0]) & (t <= RESTART_WIN[1])].mean())
    return {"zmax": z, "held_b": held_win(rb), "held_d": held_win(rd),
            "t_restart": float(rd.time[0] - rd.macro["avg_time_interval"].values[0])}


def hold_metrics(d: xr.Dataset, sp: dict, name: str) -> dict:
    nr = d.hold_n_hold_ret.values.astype(float)
    mu = np.nan_to_num(d.hold_ret_dt_mean.values)
    sd_i = np.nan_to_num(d.hold_ret_dt_std.values)
    th = d.hold_time.values
    late = th > T0_VER
    h = {"intervals": int(d.sizes["hold_time"]), "violation": float(d.attrs["hold_invariant_violation"]),
         "sum_in": int(d.hold_n_hold_in.sum()), "sum_ret": int(d.hold_n_hold_ret.sum()),
         "sum_esc_direct": int(d.hold_n_esc_direct.sum()), "sum_esc_orbit": int(d.hold_n_esc_orbit.sum()),
         "held_end": int(d.hold_held_end.values[-1]),
         "in_over_ret_late": float(d.hold_n_hold_in.values[late].sum() / max(nr[late].sum(), 1)),
         "in_eq_ret_all": int(bool((d.hold_n_hold_in == d.hold_n_hold_ret).all())),
         "first_output": int(d.hold_output.values[0]), "t_last": float(th[-1])}
    if nr.sum() > 0 and np.any(mu != 0):
        m_ = float((nr * mu).sum() / nr.sum())
        h["delay_mean"] = m_
        h["delay_std"] = float(np.sqrt(max((nr * (sd_i**2 + mu**2)).sum() / nr.sum() - m_**2, 0)))
        h["r_held_max_alt_km"] = float((d.hold_r_held_max.max() - sp["R"]) / 1e3)
    if name == "E8_main_kepler":
        h["growth_per_s"] = float(np.polyfit(th, d.hold_held_end.values.astype(float), 1)[0])
        h["entry_per_s"] = float(d.hold_n_hold_in.sum() / th[-1])
    return h


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    runs = registry.campaign(CAMPAIGN, include_alternates=True)
    done = [r for r in runs if run_path(r).is_file()]
    print(f"h2_hold: {len(done)}/{len(runs)} runs have a run file")
    ref = next(r for r in done if r.alias == "E7a_kepler")
    sp = source_params(open_run(ref))
    sp["from"] = ref.alias

    red, inv_rows = {}, []
    for r in done:
        ds = analyse_run(r, sp)
        write(ds, tiers.analysis_path(CAMPAIGN, r.alias))
        red[r.alias] = ds
        inv_rows.append({"alias": r.alias, "role": r.role, "ubc": ds.attrs["ubc"], "r_top_km": (ds.attrs["r_top_m"] - ds.attrs["planet_radius"]) / 1e3,
                         "t_end_s": float(ds.time[-1]), "window_t0_s": ds.attrs["window_t0"], "window_rule": ds.attrs["window_rule"],
                         "escape_rate_s-1": ds.attrs["dsmc_escape_rate"], "escape_npart": ds.attrs["dsmc_escape_npart"],
                         "held_end_phys": float(ds.N_ball_phys[-1]), "hold_intervals": ds.attrs.get("hold_source_count", 0),
                         "hold_invariant_violation": ds.attrs.get("hold_invariant_violation", np.nan),
                         "wall_s": ds.attrs["wall_s"]})
        print(f"  {r.alias}: window {ds.attrs['window_t0']:.0f}-{float(ds.time[-1]):.0f} s, "
              f"escape {ds.attrs['dsmc_escape_rate']:.3e} s-1 ({ds.attrs['dsmc_escape_npart']} part)")

    tops = set(pd.read_csv(SWEEP_MATRIX).query("atmosphere == 'H2' and dr_max_km == 250").z_top_actual_km)
    tops |= set(pd.read_csv(HOLD_MATRIX).z_top_actual_km)
    tops.add(round(red[ref.alias].attrs["r_top_m"] / 1e3 - sp["R"] / 1e3, 1))
    an = analytic_references(sp, sorted(tops))

    ver = verification_metrics(red, sp)
    if "E7b_explicit" in red and "E7d_explicit_restart" in red:
        rb, rd = open_run(next(r for r in done if r.alias == "E7b_explicit")), \
            open_run(next(r for r in done if r.alias == "E7d_explicit_restart"))
        if float(rd.time[-1]) >= RESTART_WIN[1] - 1:
            ver["restart"] = restart_metrics(rb, rd)
    for name in VERIF:
        if name in red and "hold_time" in red[name].dims:
            ver["hold"][name] = hold_metrics(red[name], sp, name)
    ver["dt"] = sp["dt"]

    srcs = [run_path(r) for r in done]
    an.attrs.update(tiers.analysis_attrs("h2_hold: collisionless references and E7-E9 verification metrics", srcs, SCRIPT))
    an.attrs.update(verification_json=json.dumps(ver, default=float), t0_verification=T0_VER,
                    restart_window=list(RESTART_WIN), n_runs_analysed=len(done), n_runs_registered=len(runs))
    write(an, tiers.analysis_path(CAMPAIGN, CAMPAIGN))

    tab = output_dir(CAMPAIGN, "tables")
    pd.DataFrame(inv_rows).to_csv(tab / "run_inventory.csv", index=False)
    rows = [{"run": k, "check": "analytic", **v} for k, v in ver["analytic"].items()]
    rows += [{"run": k, "check": "hold", **v} for k, v in ver["hold"].items()]
    pd.DataFrame(rows).to_csv(tab / "verification_summary.csv", index=False)
    print(f"wrote {tiers.analysis_path(CAMPAIGN, CAMPAIGN)} and {len(done)} per-run files; tables in {tab}")


if __name__ == "__main__":
    main()
