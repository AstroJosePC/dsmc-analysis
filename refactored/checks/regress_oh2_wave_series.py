"""Are the OH2_A wave series and profiles extracted correctly, and is their scatter just DSMC noise?

    MPLBACKEND=Agg python -B checks/regress_oh2_wave_series.py

1. Raw ASCII vs run file: every Macro output (all 2,613) is re-parsed independently of the builder, at
   the cells plotted in fig4a. num_part, num_dens, vel_rad and temp_trans must equal the run file bit for bit
   for both species. The column map comes from MOD_Output.f90: per-species fields are written
   field-major (num_part(1:ns), num_dens(1:ns), vel_rad(1:ns), ...) after 6 leading columns.
   Each file must hold one row per cell, with a complete cell index; time must be monotonic and
   uniform.
2. Internal consistency for all cells and outputs: n V / (W N_part) = 1. A species or column swap
   breaks this.
3. Shot noise: in resolved cells (background N >= 10) at 100-200 km, the pre-forcing scatter of
   dn/n is <= 1.3 / sqrt(N), the Poisson value (window averaging of correlated samples lowers it).
4. Noise-floor calibration: over the 4 periods before forcing, the fitted amplitude at omega
   divided by its AR(1) standard error should be Rayleigh (95% below 2.45, median 1.18). Pass
   requires >= 85% below 2.45 for every species and field. The median is reported because it
   shows by how much the errors are under- or over-estimated.
5. Detection: in the bottom cell the forced amplitude exceeds 5x the noise floor, for dn and dv_r
   and for both species.
Report: outputs/checks/tables/regress_oh2_wave_series.json; exits 1 on any failure.
"""

import json
import sys
import warnings

import numpy as np

from dsmclib import wave
from dsmclib.ncrun import open_run_nc
from dsmclib.paths import output_dir

warnings.simplefilter("ignore", RuntimeWarning)
ALIAS = "OH2_A"
PLOTTED_KM = [101, 150, 230, 460]  # fig4a altitudes (O: 101/150/230, H2: 101/150/460)
FIELD_ORDER = ["num_part", "num_dens", "vel_rad", "vel_zen", "vel_azi", "vel_therm", "temp_rad", "temp_zen",
               "temp_azi", "temp_kin", "temp_trans"]
CHECK_FIELDS = ["num_part", "num_dens", "vel_rad", "temp_trans"]
RAYLEIGH_95, RAYLEIGH_MEDIAN = np.sqrt(-2 * np.log(0.05)), np.sqrt(2 * np.log(2))


def main():
    run = open_run_nc(ALIAS)
    res = run.path.parent
    ns, nr = len(run.species), run.macro.sizes["radial"]
    t, steps = run.time, run.macro["output_step"].values
    alt = run.macro["altitude"].values / 1e3
    cells = sorted({int(np.argmin(abs(alt - a))) for a in PLOTTED_KM})
    report, fails = {"run": ALIAS, "cells_checked": cells, "cell_altitudes_km": alt[cells].tolist()}, []

    # 1. Raw ASCII vs run file
    raw = {f: np.full((len(t), len(cells), ns), np.nan) for f in CHECK_FIELDS}
    t_raw, bad_layout = np.empty(len(t)), []
    for k, step in enumerate(steps):
        a = np.loadtxt(res / "Macro" / f"Output{step}_Proc0_Macroscopic.ascii", ndmin=2)
        ci = a[:, 5].astype(int) - 1
        if a.shape[0] != nr or not np.array_equal(np.sort(ci), np.arange(nr)):
            bad_layout.append(int(step))
        t_raw[k] = a[0, 0]
        for j, c in enumerate(cells):
            row = a[ci == c][0]
            for f in CHECK_FIELDS:
                col = 6 + FIELD_ORDER.index(f) * ns
                raw[f][k, j] = row[col:col + ns]
    report["files_parsed"] = len(steps)
    report["files_with_bad_row_layout"] = bad_layout
    if bad_layout:
        fails.append(f"{len(bad_layout)} Macro files with missing/extra cell rows")
    dt = np.diff(t)
    report["time monotonic & uniform"] = bool(np.all(dt > 0) and np.ptp(dt) < 1e-9 * dt.mean())
    report["time equals raw c1"] = bool(np.array_equal(t_raw, t))
    if not (report["time monotonic & uniform"] and report["time equals raw c1"]):
        fails.append("time axis")
    for f in CHECK_FIELDS:
        rf = run.macro[f].transpose("time", "radial", "species").values[:, cells, :]
        same = bool(np.array_equal(rf, raw[f]))
        report[f"raw ASCII == run file: {f}"] = same
        if not same:
            fails.append(f"{f}: run file differs from raw ASCII")

    # 2. n V / (W N) = 1
    n = run.macro["num_dens"].transpose("time", "species", "radial").values
    N = run.macro["num_part"].transpose("time", "species", "radial").values
    V = np.asarray(run.cell_volume())
    W = run.weight.values[None, :, None]
    ok = N > 0
    dev = float(np.max(np.abs((n * V[None, None, :] / (W * N))[ok] - 1)))
    report["max |n V/(W N) - 1|"] = dev
    if dev > 1e-10:
        fails.append(f"n V/(W N) deviates by {dev:.2g}")

    # Wave response and noise floor (same call as compute_analysis.py)
    bg_t1 = run.wave_start_time
    background = (bg_t1 - 3000.0, bg_t1)
    resp = wave.forcing_response(run, background)
    nw = wave.null_window(run, background)
    report["null window [s]"] = list(nw[:2])
    report["null periods"] = nw[2]

    # 3. Shot noise, pre-forcing
    pre = (t > background[0]) & (t <= background[1])
    ratios = {}
    for s, sp in enumerate(run.species):
        Nbar = N[pre, s].mean(0)
        sel = (Nbar >= 10) & (alt >= 100) & (alt <= 200)
        y = n[pre, s][:, sel] / n[pre, s][:, sel].mean(0) - 1
        r = y.std(0) * np.sqrt(Nbar[sel])
        ratios[sp] = r
        report[f"shot noise {sp}: std(dn/n) * sqrt(N), 100-200 km (min, median, max)"] = [
            float(r.min()), float(np.median(r)), float(r.max())]
        if np.any(r > 1.3):
            fails.append(f"shot noise {sp}: std(dn/n) up to {r.max():.2f}/sqrt(N)")

    # 4. Noise-floor calibration
    for f in ("num_dens", "temp_trans", "vel_rad"):
        for sp in run.species:
            a0 = resp[f"{f}_null_amp"].sel(species=sp).values
            s0 = resp[f"{f}_null_amp_se"].sel(species=sp).values
            z = (a0 / s0)[np.isfinite(a0 / s0)]
            frac = float(np.mean(z < RAYLEIGH_95))
            report[f"null {f} {sp}: n={len(z)}, frac amp/se < 2.45 (Rayleigh 0.95)"] = frac
            report[f"null {f} {sp}: median amp/se (Rayleigh {RAYLEIGH_MEDIAN:.2f})"] = float(np.median(z))
            if frac < 0.85:
                fails.append(f"null {f} {sp}: only {frac:.0%} of amp/se < 2.45")

    # 5. Detection in the bottom cell
    for f in ("num_dens", "vel_rad"):
        for sp in run.species:
            amp = float(resp[f"{f}_amp"].sel(species=sp).isel(radial=0))
            a0 = float(resp[f"{f}_null_amp"].sel(species=sp).isel(radial=0))
            report[f"bottom cell {f} {sp}: forced amp, noise floor"] = [amp, a0]
            if not amp > 5 * a0:
                fails.append(f"bottom cell {f} {sp}: amp {amp:.3g} not > 5x noise floor {a0:.3g}")

    report["pass"] = not fails
    report["failures"] = fails
    out = output_dir("checks", "tables") / "regress_oh2_wave_series.json"
    out.write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"report: {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
