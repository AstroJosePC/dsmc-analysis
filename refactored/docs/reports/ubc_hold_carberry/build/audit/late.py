import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
from dsmclib.ncrun import open_run_nc
from dsmclib import escape
SW = Path.home() / "MarsAtmos-H2UBC-sweep/runs"
def esc(name, a, b):
    run = open_run_nc(SW / name / "Results" / f"{name}_run.nc")
    e = escape.dsmc_escape(run, a, b)
    return float(e.escape_rate.isel(species=0)), int(e.escape_npart.isel(species=0))
for z, dr in ((10000, 1000), (20000, 250), (20000, 1000), (40000, 250), (40000, 1000)):
    for a, b in ((32e3, 300e3), (100e3, 200e3), (200e3, 300e3)):
        (r1, k1), (r2, k2) = esc(f"H2_U1_z{z}_dr{dr}", a, b), esc(f"H2_U2_z{z}_dr{dr}", a, b)
        q = r2 / r1
        print(f"z{z} dr{dr} {a/1e3:.0f}-{b/1e3:.0f}ks: reflect {r1/2.85e26:.3f} kepler {r2/2.85e26:.3f} ratio {q:.3f} +- {q*np.sqrt(1/k1+1/k2):.3f}")
