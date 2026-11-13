"""Are the Nov-2026 B reruns faithful to the Feb-2025 run? Archived Results_20250216_153021.481
(the only Feb-2025 O+H2 run with processed outputs: 49 outputs = 1,182 s, 35 cells) vs the
first 49 outputs of OH2_B_r1..r3.

    python checks/regress_oh2_rerun_vs_2025.py

Deterministic quantities must match exactly:
  output times, lower-boundary setup (density, flow rate, Jeans columns) and cell altitudes.
Stochastic quantities must agree within the replicate scatter:
  - computational particle inventory per species at each output, and
  - the time-mean density and kinetic temperature per cell over outputs 20-49 (t ~ 480-1182 s),
  as z = (archived - replicate mean) / replicate std. Pass if >= 90% of populated
  (>= 5 computational particles) cell/series values have |z| < 3. With 3 replicates the std is itself
  uncertain, hence the generous criterion; the z distribution is reported.
Report: outputs/checks/tables/regress_oh2_rerun_vs_2025.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from dsmclib.ncrun import open_run_nc
from dsmclib.paths import archive_root, output_dir

warnings.simplefilter("ignore", RuntimeWarning)
ARCHIVED = "Results_20250216_153021.481"
REPLICATES = ["OH2_B_r1", "OH2_B_r2", "OH2_B_r3"]
N_OUT, AVG_FROM = 49, 19  # outputs compared; time mean over outputs 20..49 (0-based 19..48)


def main():
    arc = archive_root() / ARCHIVED
    a_n = np.load(arc / "macro_Number Density.npy")[..., 0]  # (species, time, radial)
    a_np = np.load(arc / "macro_Number Particles.npy")[..., 0]
    a_tk = np.load(arc / "macro_Kinetic Temperature.npy")[..., 0]
    p = np.load(arc / "params_data.npz")
    runs = [open_run_nc(a) for a in REPLICATES]
    m = [r.macro.isel(time=slice(0, N_OUT)) for r in runs]
    report, fails = {}, []

    # Deterministic
    # The archived surface_data.npz is NOT used: its per-species columns were split species-major
    # by the 2025 processing script, while the solver writes them field-major, so for this
    # 2-species run its surface fields are scrambled. The raw Info/*.ascii files are compared instead.
    def info(run_dir, name):
        return np.array([[float(t) for t in l.split()] for l in (run_dir / "Info" / name).read_text().splitlines()
                         if l.strip()])

    rerun_dir = runs[0].path.parent
    det = {
        "time": (p["time"], runs[0].time[:N_OUT]),
        "radial_cells_km": (p["radial_cells"], runs[0].macro["altitude"].values / 1e3),
    }
    for name in ("Grid.ascii", "Surface_Cells.ascii", "Radial_Cells.ascii"):
        det[f"Info/{name}"] = (info(arc, name), info(rerun_dir, name))
    for k, (want, got) in det.items():
        want, got = np.asarray(want), np.asarray(got)
        nz = want != 0
        err = float(np.max(np.abs(got[nz] / want[nz] - 1))) if got.shape == want.shape else np.inf
        if got.shape == want.shape and np.any(got[~nz] != 0):
            err = np.inf
        report[f"deterministic {k} max rel diff"] = err
        if err > 1e-9:
            fails.append(f"{k}: max rel diff {err:.3g}")

    # Stochastic
    def zscores(arch, reps, mask):
        reps = np.stack(reps)
        mu, sd = reps.mean(0), reps.std(0, ddof=1)
        z = (arch - mu) / np.where(sd > 0, sd, np.nan)
        return z[mask & np.isfinite(z)]

    inv_arch = a_np.sum(-1)  # (species, time)
    inv_reps = [mm["num_part"].transpose("species", "time", "radial").values.sum(-1) for mm in m]
    zs = {"inventory": zscores(inv_arch[:, 5:], [x[:, 5:] for x in inv_reps], np.ones_like(inv_arch[:, 5:], bool))}
    sl = slice(AVG_FROM, N_OUT)
    pop = a_np[:, sl].mean(1) >= 5
    for name, arch, field in (("num_dens", a_n, "num_dens"), ("temp_kin", a_tk, "temp_kin")):
        reps = [mm[field].transpose("species", "time", "radial").values[:, sl].mean(1) for mm in m]
        zs[name] = zscores(arch[:, sl].mean(1), reps, pop)
    for k, z in zs.items():
        frac = float(np.mean(np.abs(z) < 3))
        report[f"stochastic {k}: n={len(z)}, frac |z|<3"] = frac
        report[f"stochastic {k}: median |z|"] = float(np.median(np.abs(z)))
        if frac < 0.90:
            fails.append(f"{k}: only {frac:.0%} of |z| < 3")
    report["final inventory archived (O, H2)"] = inv_arch[:, -1].tolist()
    report["final inventory reruns (O, H2)"] = [x[:, N_OUT - 1].tolist() for x in inv_reps]
    report["pass"] = not fails
    report["failures"] = fails
    out = output_dir("checks", "tables") / "regress_oh2_rerun_vs_2025.json"
    out.write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"report: {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
