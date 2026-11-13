"""Run file (<alias>_run.nc, dsmc_netcdf_builder >= 1.2) vs the raw solver ASCII, plus physics sanity checks.

    MPLBACKEND=Agg python checks/regress_oh2_netcdf_vs_ascii.py          # all oh2_escape_rerun runs

For each run:
  1. ~20 random (output, cell) Macro rows and ~20 Surface/Escape rows: every run-file field equals the
     ASCII value bit-for-bit (the builder parses with the same float conversion).
  2. Geometry and metadata: rates radial = Grid.ascii centre of the boundary cell, boundary_radius
     = top/bottom face, upper_area = 4 pi r_top^2; per-species mass/weight = Info values.
  3. Sanity: N_phys = W * N_part; Jeans at the lower boundary reproduces Surface_Cells.ascii;
     inject_rate = num_part_inject * W; O in the collisional region (Kn < 0.1) follows the 270 K isothermal profile.
Report: outputs/checks/regress_oh2_netcdf_vs_ascii.json; exits 1 on any failure.
"""

import json
import sys
import warnings

import numpy as np

from dsmclib import convergence as cv
from dsmclib import escape as es
from dsmclib import profiles as pf
from dsmclib import registry
from dsmclib.constants import AMU
from dsmclib.ncrun import open_run_nc
from dsmclib.paths import output_dir

warnings.simplefilter("ignore", RuntimeWarning)
RNG = np.random.default_rng(20261109)
MACRO_ORDER = ["num_part", "num_dens", "vel_rad", "vel_zen", "vel_azi", "vel_therm", "temp_rad", "temp_zen",
               "temp_azi", "temp_kin", "temp_trans", "temp_int", "temp_ov"]
MACRO_TAIL = ["mean_free_path", "scale_height", "knudsen", "therm_conduct_up", "therm_conduct_down"]
SURF = ["num_part_inject", "num_part_return", "num_part_reemit", "num_part_surf", "inject_rate", "return_rate",
        "reemit_rate", "return_time", "inject_energy", "return_energy"]
ESC = ["num_part_escape", "num_part_ball", "escape_rate", "escape_time", "escape_speed", "escape_energy"]


def read_rows(path):
    return np.array([[float(t) for t in line.split()] for line in path.read_text().splitlines() if line.strip()])


def check_run(alias):
    run = open_run_nc(alias)
    res = run.path.parent
    ns = len(run.species)
    fails, n_cmp = [], 0
    steps = run.macro["output_step"].values

    # 1a. Macro rows
    for k in RNG.choice(len(steps), size=min(20, len(steps)), replace=False):
        rows = read_rows(res / "Macro" / f"Output{steps[k]}_Proc0_Macroscopic.ascii")
        if not np.isclose(rows[0, 0], run.time[k], rtol=0, atol=0):
            fails.append(f"time mismatch Output{steps[k]}")
        if not rows[0, 1] == run.macro["avg_time_interval"].values[k]:
            fails.append(f"avg_time_interval mismatch Output{steps[k]}")
        row = rows[RNG.integers(len(rows))]
        ir = int(row[5]) - 1
        col = 6
        for f in MACRO_ORDER:
            for s in range(ns):
                n_cmp += 1
                if row[col] != run.macro[f].isel(species=s, time=k, radial=ir).item():
                    fails.append(f"Macro {f}[{s}] Output{steps[k]} cell {ir + 1}")
                col += 1
        col += ns * (ns + 1)  # collision-frequency matrix (checked by the builder's own layout)
        for f in MACRO_TAIL:
            for s in range(ns):
                n_cmp += 1
                if row[col] != run.macro[f].isel(species=s, time=k, radial=ir).item():
                    fails.append(f"Macro {f}[{s}] Output{steps[k]} cell {ir + 1}")
                col += 1

    # 1b. Surface / Escape rows
    for kind, ds, fields in (("Surface", run.surface, SURF), ("Escape", run.escape, ESC)):
        for k in RNG.choice(len(steps), size=min(20, len(steps)), replace=False):
            row = read_rows(res / "Rates" / f"Output{steps[k]}_Proc0_{kind}.ascii")[0]
            col = 7
            for f in fields:
                for s in range(ns):
                    n_cmp += 1
                    if row[col] != ds[f].isel(species=s, time=k, boundary_cell=0).item():
                        fails.append(f"{kind} {f}[{s}] Output{steps[k]}")
                    col += 1
            area_name = "lower_area" if kind == "Surface" else "upper_area"
            if row[6] != ds[area_name].isel(boundary_cell=0).item():
                fails.append(f"{kind} area Output{steps[k]}")

    # 2. Geometry and metadata
    grid = read_rows(res / "Info" / "Grid.ascii")
    top, bot = grid[grid[:, 1] == grid[:, 1].max()][0], grid[grid[:, 1] == 1][0]
    geo = {
        "escape radial = top-cell centre": (run.escape["radial"].item(), 0.5 * (top[4] + top[5])),
        "surface radial = bottom-cell centre": (run.surface["radial"].item(), 0.5 * (bot[4] + bot[5])),
        "escape boundary_radius = top face": (run.escape["boundary_radius"].item(), top[5]),
        "surface boundary_radius = bottom face": (run.surface["boundary_radius"].item(), bot[4]),
        "upper_area = 4 pi r_top^2": (run.escape["upper_area"].item(), 4 * np.pi * top[5] ** 2),
    }
    info = (res / "Info" / "Simulation_Parameters.ascii").read_text().split("Collision Information")[0]
    masses = [float(l.split()[1]) for l in info.splitlines() if l.strip().startswith("Mass:") and "[amu]" in l]
    weights = [float(l.split()[2]) for l in info.splitlines() if "Particle Weight:" in l]
    for s, sp in enumerate(run.species):
        geo[f"mass[{sp}]"] = (run.mass.sel(species=sp).item(), masses[s] * AMU)
        geo[f"weight[{sp}]"] = (run.weight.sel(species=sp).item(), weights[s])
    for name, (got, want) in geo.items():
        n_cmp += 1
        if not np.isclose(got, want, rtol=1e-12):
            fails.append(f"{name}: {got} != {want}")

    # 3. Sanity
    inv = cv.inventory(run)
    sanity = {"max |N_phys/(W N_part) - 1|": float(abs(inv.N_phys / (inv.N_part * run.weight) - 1).max())}
    j = es.jeans_at_lower_boundary(run)
    sanity["max |Jeans_lb / Surface_Cells - 1|"] = float(abs(
        j.jeans_rate.values / run.info_surface["jeans_escape_rate"].isel(surface_cell=0).values - 1).max())
    inj = run.surface["inject_rate"] / (run.surface["num_part_inject"] * run.weight)
    sanity["max |inject_rate/(num_part_inject W) - 1|"] = float(abs(inj - 1).max())
    t_pre = run.wave_start_time or run.t_end
    run.macro["temp_trans_corr"] = pf.temp_trans_corrected(run)  # analysis tier, attached explicitly
    prof = pf.time_mean_profiles(run, t_pre - 3000.0, t_pre)
    n_iso = pf.isothermal_reference(run, prof).sel(species="O")
    # collisional region only (O Knudsen number < 0.1): above it the profile legitimately departs
    # from isothermal hydrostatics (temperature rises towards the exobase)
    below = prof.knudsen.sel(species="O").values < 0.1
    n_o, sem_o = prof.num_dens.sel(species="O").values, prof.num_dens_sem.sel(species="O").values
    sanity["O n/n_iso (Kn<0.1): max |r - 1|"] = float(abs(n_o[below] / n_iso.values[below] - 1).max())
    # error of the reference = anchor-cell SEM propagated along the profile (it shifts the whole curve)
    sem_ref = n_iso.values * sem_o[0] / n_o[0]
    z = (n_o - n_iso.values) / np.hypot(sem_o, sem_ref)
    z[0] = 0.0  # the anchor cell matches by construction
    sanity["O n vs n_iso (Kn<0.1): max |z|"] = float(abs(z)[below].max())
    limits = {"max |N_phys/(W N_part) - 1|": 1e-10, "max |Jeans_lb / Surface_Cells - 1|": 1e-12,
              "max |inject_rate/(num_part_inject W) - 1|": 1e-10, "O n vs n_iso (Kn<0.1): max |z|": 4.0}
    for k, lim in limits.items():
        if not sanity[k] <= lim:
            fails.append(f"sanity {k} = {sanity[k]:.3g} > {lim}")
    return {"run": alias, "values_compared": n_cmp, "failures": fails, "sanity": sanity, "pass": not fails}


if __name__ == "__main__":
    report = [check_run(r.alias) for r in registry.campaign("oh2_escape_rerun")]
    out = output_dir("checks", "tables") / "regress_oh2_netcdf_vs_ascii.json"
    out.write_text(json.dumps(report, indent=2))
    for r in report:
        print(f"{r['run']}: {'PASS' if r['pass'] else 'FAIL'} ({r['values_compared']} values) "
              + "; ".join(f"{k}={v:.2e}" for k, v in r["sanity"].items()))
        for f in r["failures"][:10]:
            print("   ", f)
    print(f"report: {out}")
    sys.exit(0 if all(r["pass"] for r in report) else 1)
