import sys, numpy as np, time
sys.path.insert(0, "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
from dsmclib import exosphere
AMU = 1.66053892e-27; m = 2.02 * AMU; r0 = 3489.5e3; n0 = 2.3e13; T0 = 270.
for z in (2627.3e3, 40127.3e3):
    rt = 3389.5e3 + z
    for nv in (900, 1200, 1500, 2500, 4000):
        t = time.time()
        b = exosphere.escape_rate(n0, T0, m, r0, rt, ubc="ballistic", n_v=nv)
        r = exosphere.escape_rate(n0, T0, m, r0, rt, ubc="reflect", n_v=nv)
        h = exosphere.held_count(n0, T0, m, r0, rt, n_v=nv)
        h3 = exosphere.held_count(n0, T0, m, r0, rt, n_v=nv, t=3e5)
        print(f"z={z/1e3:.0f} nv={nv} excess={100*(b/r-1):.3f}% held={h:.4e} frac3e5={h3/h:.4f} ({time.time()-t:.1f}s)", flush=True)
