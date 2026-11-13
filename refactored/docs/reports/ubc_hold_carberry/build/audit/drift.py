import sys
from pathlib import Path
sys.path.insert(0, "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
from dsmclib.ncrun import open_run_nc
from dsmclib import escape
SW = Path.home() / "MarsAtmos-H2UBC-sweep/runs"
for name in ("H2_U3_z40000_dr250", "H2_U3_z40000_dr1000", "H2_U1_z40000_dr250", "H2_U2_z40000_dr250"):
    run = open_run_nc(SW / name / "Results" / f"{name}_run.nc")
    row = []
    for a, b in ((19e3, 32e3), (32e3, 100e3), (100e3, 200e3), (200e3, 300e3)):
        e = escape.dsmc_escape(run, a, b)
        row.append(f"{a/1e3:.0f}-{b/1e3:.0f}ks: {float(e.escape_rate.isel(species=0))/2.85e26:.3f} (k={int(e.escape_npart.isel(species=0))})")
    print(name, " | ".join(row))
