import sys
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
from dsmclib.ncrun import open_run_nc
from dsmclib import escape, convergence
SW = Path.home() / "MarsAtmos-H2UBC-sweep/runs"
for p in sorted(SW.glob("H2_U[123]_z*_dr*")):
    f = p / "Results" / f"{p.name}_run.nc"
    if "finished:" not in (p / "run_meta.txt").read_text() or not f.exists():
        continue
    run = open_run_nc(f)
    s = run.escape["escape_rate"].sum("boundary_cell").isel(species=0)
    for tol in (0.05, 0.10):
        sw = convergence.steady_window(s, rel_tol=tol)
        t0 = sw.t_start
        if t0 is None:
            print(p.name, tol, "NOT steady", f"rel_drift_ub={sw.rel_drift_ub:.3f}")
            continue
        e = escape.dsmc_escape(run, t0)
        print(p.name, tol, f"t0={t0/1e3:.0f}ks  Phi/phiJ={float(e.escape_rate.isel(species=0))/2.85e26:.3f} k={int(e.escape_npart.isel(species=0))}")
    inv = convergence.inventory(run).N_phys.isel(species=0)
    sw = convergence.steady_window(inv)
    tt = s.time.values
    sel = tt > sw.t_start
    d = convergence.drift_test(tt[sel], s.values[sel], 10, 0.95)
    print("   inventory window", sw.t_start / 1e3, "ks; escape rel drift over it", round(d["rel_drift"], 3), "ub", round(d["rel_drift_ub"], 3))
