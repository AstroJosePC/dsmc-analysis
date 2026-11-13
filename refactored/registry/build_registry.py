"""Build refactored/registry/simulation_metadata.db: a copy of the root DB plus run aliases.

    python registry/build_registry.py            # first build: copy root DB, add alias tables
    python registry/build_registry.py --refresh  # re-copy the root DB, then rebuild aliases

The root simulation_metadata.db is only read (copied). Aliases and campaigns are defined
below in CAMPAIGNS / ALIASES; edit them here and re-run. Aliased runs that the root DB
does not know about (it was last written Aug 2025) get a minimal simulation_runs row from
their Info/Simulation_Parameters.ascii.
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

from dsmclib.params import read_simulation_parameters_ascii
from dsmclib.paths import REFACTORED_ROOT, REGISTRY_DB, archive_root

ROOT_DB = REFACTORED_ROOT.parent / "simulation_metadata.db"

CAMPAIGNS = {
    "s1_s6_azimuthal": "2D spherical wedge runs at 128-1024 azimuth cells (S1-S6), Lateral Energy Transport paper",
    "radial_1d": "1D radial DSMC runs (single column): density growth, heating, Tian et al. comparison",
    "synthetic_methods": "Method tests on synthetic signals; no DSMC runs",
    "oh2_escape_rerun": ("Mars O+H2 1D reruns (Nov 2026) of the Feb-2025 runs, 40,000 km top, UBC_FREE_ESCAPE; "
                         "MarsAtmos main c94282ef + O mass 16.0 amu + UBC info line (branch rerun/oh2-feb2025)"),
}

# Runs not yet copied into ArchiveSims: absolute Results folders (registry.Run.path uses them as-is).
OH2_RUNS = Path.home() / "MarsAtmos-OH2-rerun" / "runs"

# (alias, campaign, results_base, role, notes). Order within a campaign = sort order.
ALIASES = [
    ("S1", "s1_s6_azimuthal", "Results_20240719_003802.440", "current",
     "128 azi. Loader uses combined_10runs pkl = 10-member ensemble mean (noise only ~1.4-2x below one member); bg window [0,2]."),
    ("S2", "s1_s6_azimuthal", "Results_20240719_025013.785", "current",
     "256 azi. 10-member ensemble mean (combined_10runs pkl); bg window [0,2] inflates lock-in sigma_c."),
    ("S3", "s1_s6_azimuthal", "Results_20240719_025200.987", "current",
     "384 azi. 10-member ensemble mean (combined_10runs pkl); bg window [0,2] inflates lock-in sigma_c."),
    ("S4", "s1_s6_azimuthal", "Results_20240722_034301.982", "current",
     "512 azi, 2024 run (+-dphi band, W=1.43e24), same batch as S5/S6. Manuscript pipeline uses this one since 2026-09-14."),
    ("S4_2025", "s1_s6_azimuthal", "Results_20250904_113551.097", "alternate",
     "512 azi re-run, fixed 85-95 deg band, W=1.02e25. Agrees with S4 within 1 sigma (band-thickness control)."),
    ("S5", "s1_s6_azimuthal", "Results_20240722_040346.456", "current", "640 azi."),
    ("S6", "s1_s6_azimuthal", "Results_20240722_041906.318", "current",
     "1024 azi. Only .npy fields (no macro pkl / surface data): forced columns come from growth_peak_data pkls."),
    ("1D_5p", "radial_1d", "Results_20250326_133319.916", "current",
     "5 forcing periods. 'AW1' heating-rate data in generate_figure_3_updated.py; 'Spherical 1D' in compare_density_growth.py."),
    ("1D_10p", "radial_1d", "Results_20250326_135431.191", "current",
     "10 forcing periods. 'New Simulation' in compare_density_growth.py."),
    ("1D_10p_jun", "radial_1d", "Results_20250605_015133.742", "current",
     "10 forcing periods, June 2025. Labelled 'AW1' in density_growth_with_bootstrap.py (conflicts with 1D_5p)."),
    ("1D_20p_jun", "radial_1d", "Results_20250605_042405.584", "current",
     "20 forcing periods, June 2025. Labelled 'AW2' in density_growth_with_bootstrap.py."),
    ("OH2_A", "oh2_escape_rerun", str(OH2_RUNS / "OH2_A_128cell_20261109_180509" / "Results"), "current",
     "Rerun of Results_20250216_103247.560: 128 cells, 130,650 steps, avg every 50 steps. Flux wave from "
     "step 102,050 (omega 9.5e-3 rad/s, amp 0.25); run ends after ~4.2 of the 10 configured periods."),
    ("OH2_B_r1", "oh2_escape_rerun", str(OH2_RUNS / "OH2_B_35cell_r1_20261109_180509" / "Results"), "current",
     "Rerun of the Results_20250216_114013.979 / 144254.283 / 153021.481 settings: 35 cells, avg every 250 steps. "
     "Extended from 12,250 to 130,650 steps (user request); wave start step 828,750 never reached. Replicate 1/3."),
    ("OH2_B_r2", "oh2_escape_rerun", str(OH2_RUNS / "OH2_B_35cell_r2_20261109_180741" / "Results"), "current",
     "As OH2_B_r1; independent stochastic replicate 2/3."),
    ("OH2_B_r3", "oh2_escape_rerun", str(OH2_RUNS / "OH2_B_35cell_r3_20261109_180752" / "Results"), "current",
     "As OH2_B_r1; independent stochastic replicate 3/3."),
]

# h2_ubc_rtop coarse sweep (Nov 2026): one alias per run id in matrix.csv (e.g. H2O_U2_z10000_dr250).
# Only runs that have started (Info/ written) are registered, so the registry can be rebuilt mid-sweep.
CAMPAIGNS["h2_ubc_rtop"] = (
    "Mars pure-H2 and H2+O 1D sweep (Nov 2026): UBC {reflect, Kepler-hold ballistic, free escape} x z_top "
    "{2.5k-40k km} x dr_max {1000, 250 km}, 270 K, 3e5 s, no waves; MarsAtmos fix/ubc-reflect-ballistic. "
    "Design: ~/MarsAtmos-H2UBC-sweep/EXPERIMENT_DESIGN.md")
H2UBC_ROOT = Path.home() / "MarsAtmos-H2UBC-sweep"


def _h2ubc_aliases():
    import csv
    matrix = H2UBC_ROOT / "matrix.csv"
    if not matrix.exists():
        return []
    rows = []
    for r in csv.DictReader(open(matrix)):
        results = H2UBC_ROOT / "runs" / r["id"] / "Results"
        if (results / "Info" / "Simulation_Parameters.ascii").exists():
            rows.append((r["id"], "h2_ubc_rtop", str(results), "current",
                         f"{r['species']}, UBC {r['ubc_name']}, z_top {r['z_top_km']} km nominal "
                         f"({r.get('z_top_actual_km', '?')} km actual), dr_max {r['dr_max_km']} km, "
                         f"{r.get('n_cells', '?')} cells, k={r['k']}, {r['n_steps']} steps ({r['t_end_s']} s)"))
    return rows


ALIASES += _h2ubc_aliases()

# h2_hold campaign (Nov 2026): Kepler hold (UBC 2) vs explicit RK4 ballistic tracking (UBC 5) vs reflect (UBC 1),
# pure H2, 2e6 s, plus the collisionless E7-E9 verification runs of MarsAtmos feature/ubc-hold-explicit.
# Production aliases carry a "Hold_" prefix because the run ids (H2_U2_z40000_dr250, ...) are h2_ubc_rtop aliases.
CAMPAIGNS["h2_hold"] = (
    "Mars pure-H2 1D UBC comparison (Nov 2026) after Carberry Mogan et al. (2021): UBC {reflect, Kepler hold, "
    "explicit RK4 ballistic (UBC 5)} x z_top {2.5k, 10k, 40k km}, dr_max 250 km, 270 K, 2e6 s, BallisticHold "
    "diagnostics; plus E7-E9 collisionless verification. MarsAtmos feature/ubc-hold-explicit (PR #14). "
    "Report: docs/reports/ubc_hold_carberry")
H2HOLD_ROOT = Path.home() / "MarsAtmos-H2Hold-campaign"
H2HOLD_VERIFICATION = Path.home() / "MarsAtmos-UBC-explicit" / "verification"


def _h2hold_aliases():
    import csv
    rows = []
    matrix = H2HOLD_ROOT / "matrix.csv"
    if matrix.exists():
        for r in csv.DictReader(open(matrix)):
            results = H2HOLD_ROOT / "runs" / r["id"] / "Results"
            if (results / "Info" / "Simulation_Parameters.ascii").exists():
                rows.append((f"Hold_{r['id']}", "h2_hold", str(results), "current",
                             f"{r['species']}, UBC {r['ubc_name']}, z_top {r['z_top_km']} km nominal "
                             f"({r.get('z_top_actual_km', '?')} km actual), dr_max {r['dr_max_km']} km, "
                             f"{r.get('n_cells', '?')} cells, k={r['k']}, {r['n_steps']} steps ({r['t_end_s']} s)"))
    table = H2HOLD_VERIFICATION / "experiments.tsv"
    if table.exists():
        for line in open(table):
            f = line.rstrip("\n").split("\t")
            if not f or not f[0].startswith(("E7", "E8", "E9")):
                continue
            results = H2HOLD_VERIFICATION / "runs" / f[0] / "Results"
            if (results / "Info" / "Simulation_Parameters.ascii").exists():
                rows.append((f[0], "h2_hold", str(results), "verification",
                             f"Collisionless pure-H2 verification, binary {f[1]}, UBC {f[3]}, z_top {f[4]} km "
                             f"nominal, dr_max {f[5]} km, {f[6]} steps"))
    return rows


ALIASES += _h2hold_aliases()

# drift_mb campaign (Nov 2026): 1-D drifting-Maxwellian (VEL_WAVE_LINEAR) lower-boundary forcing in pure O and O+H2.
# Two spin-ups (SPIN_O, SPIN_OH2; role "spinup") and 20 forced runs restarted from them (role "current").
CAMPAIGNS["drift_mb"] = (
    "Mars 1-D drifting-Maxwellian lower-boundary forcing (Nov 2026): atmosphere {O, O+H2} x s_O {0, 0.05, 0.1, 0.2} "
    "x P {165, 330, 660 s}, 20 forcing + 5 ring-down periods restarted from a 1e5 s spin-up; 270 K, z_top 2500 km, "
    "UBC_FREE_ESCAPE. MarsAtmos feature/velwave-1d (9088aab3), builder 1.2.1. "
    "Design: ~/MarsAtmos-DriftMB-sweep/EXPERIMENT_DESIGN.md")
DRIFTMB_ROOT = Path.home() / "MarsAtmos-DriftMB-sweep"


def _driftmb_aliases():
    import csv
    matrix = DRIFTMB_ROOT / "matrix.csv"
    if not matrix.exists():
        return []
    rows = []
    for r in csv.DictReader(open(matrix)):
        results = DRIFTMB_ROOT / "runs" / r["id"] / "Results"
        if not (results / "Info" / "Simulation_Parameters.ascii").exists():
            continue
        common = (f"{r['species']}, {r['n_cells']} cells, z_top {r['z_top_actual_km']} km actual, "
                  f"dt {float(r['dt_expected']):.6g} s, {r['n_outputs']} outputs")
        if r["stage"] == "spin":
            rows.append((r["id"], "drift_mb", str(results), "spinup",
                         f"Spin-up from empty, no forcing, {r['n_steps']} steps ({r['t_run_s']} s); {common}"))
        else:
            rows.append((r["id"], "drift_mb", str(results), "current",
                         f"Restart of {r['spin_id']}: VEL_WAVE_LINEAR s_O {r['s_O']} (w' {r['w_mps']} m/s, "
                         f"s_H2 {r['s_H2']}), P {r['period_s']} s, {r['n_force']} forcing + 5 ring-down periods, "
                         f"avg {r['avg_steps']} steps (P/16), steps {r['step0_inject']}-{r['n_steps']}; {common}"))
    return rows


ALIASES += _driftmb_aliases()


def add_missing_run(conn: sqlite3.Connection, results_base: str) -> None:
    """Insert a minimal simulation_runs row for a run folder the root DB lacks."""
    run_dir = archive_root() / results_base
    ascii_path = run_dir / "Info" / "Simulation_Parameters.ascii"
    parsed = read_simulation_parameters_ascii(ascii_path)
    sim = parsed["Simulation Parameters"]
    radius = parsed["Planetary Body Information"]["Radius"]
    # Runs without waves (e.g. h2_ubc_rtop) have no wave lines in the Info ASCII -> 0 forcing periods.
    omega, dt = sim.get("Wave Injection Frequency", 0.0), sim["Time Step"]
    start = sim.get("Start Step of Wave Injection", 0)
    end = sim.get("End Step of Wave Injection", start)
    conn.execute(
        """INSERT INTO simulation_runs (results_base, num_periods, rmin, rmax, UBC, n_rad_cell, n_azi_cell,
                                        avg_time_interval, parameters_file, data_path)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            results_base,
            # forcing periods inside the run (the wave end step can lie beyond the last step)
            max(0, int(round((min(end, sim["Number of Total Simulations Steps"]) - start - 1)
                             * omega * dt / (2 * 3.141592653589793)))),
            round(sim["Lower Radial Boundary of Grid"] - radius, -1),
            round(sim["Upper Radial Boundary of Grid"] - radius, -1),
            str(sim.get("Upper Radial Boundary Condition", "")).strip(),
            sim["Number of Radial Cells"],
            sim["Number of Azimuth Cells"],
            sim["Averages Taken Every"] * dt,
            str(ascii_path),
            str(run_dir),
        ),
    )
    print(f"  added {results_base} to simulation_runs from its Info ASCII")


def build(refresh: bool) -> None:
    if refresh or not REGISTRY_DB.exists():
        shutil.copy2(ROOT_DB, REGISTRY_DB)
        print(f"Copied {ROOT_DB} -> {REGISTRY_DB}")

    with sqlite3.connect(REGISTRY_DB) as conn:
        conn.execute("DROP TABLE IF EXISTS run_aliases")
        conn.execute("DROP TABLE IF EXISTS campaigns")
        conn.execute("CREATE TABLE campaigns (name TEXT PRIMARY KEY, description TEXT)")
        conn.execute(
            """CREATE TABLE run_aliases (
                   alias TEXT PRIMARY KEY,
                   campaign TEXT REFERENCES campaigns(name),
                   results_base TEXT NOT NULL,
                   role TEXT,
                   notes TEXT,
                   sort_order INT)"""
        )
        conn.executemany("INSERT INTO campaigns VALUES (?,?)", CAMPAIGNS.items())
        conn.executemany(
            "INSERT INTO run_aliases VALUES (?,?,?,?,?,?)",
            [(*row, i) for i, row in enumerate(ALIASES)],
        )

        known = {base for (base,) in conn.execute("SELECT results_base FROM simulation_runs")}
        for _, _, results_base, _, _ in ALIASES:
            if results_base not in known:
                add_missing_run(conn, results_base)

    print(f"Registry: {len(ALIASES)} aliases in {len(CAMPAIGNS)} campaigns -> {REGISTRY_DB}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-copy the root simulation_metadata.db first")
    build(parser.parse_args().refresh)
