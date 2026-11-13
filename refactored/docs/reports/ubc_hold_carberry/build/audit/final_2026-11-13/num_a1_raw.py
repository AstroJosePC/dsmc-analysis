"""Re-read raw BallisticHold files independently and compare with hold_* series in analysis files."""
import re
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
NAME = re.compile(r"Output(\d+)_Proc(\d+)_BallisticHold\.ascii$")
for f in sorted((ROOT / "outputs/h2_hold/analysis").glob("*_analysis.nc")):
    if f.name == "h2_hold_analysis.nc":
        continue
    ds = xr.open_dataset(f).load()
    run = Path(ds.attrs["source_files"])
    diag = run.parent / "Diagnostics"
    raw = {}
    for p in diag.glob("Output*_Proc*_BallisticHold.ascii"):
        m = NAME.search(p.name)
        o = int(m.group(1))
        a = np.atleast_2d(np.loadtxt(p, comments="#"))
        a = a[-1]  # last species row (H2 is single species)
        a = np.concatenate([a, np.zeros(max(0, 13 - a.size))])
        raw.setdefault(o, []).append(a)
    outs = sorted(raw)
    rows = []
    for o in outs:
        ps = raw[o]
        r = ps[0].copy()
        for q in ps[1:]:
            r[4:12] += q[4:12]
            r[12] = max(r[12], q[12])
        rows.append(r)
    R = np.array(rows)
    t = R[:, 0]
    nin, nret, ed, eo, held = R[:, 4], R[:, 5], R[:, 6], R[:, 7], R[:, 8]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(nret > 0, R[:, 10] / nret, 0)
        m2 = np.where(nret > 0, R[:, 11] / nret, 0)
    # pooled delay
    dm = (R[:, 10].sum() / nret.sum()) if nret.sum() else np.nan
    dsd = np.sqrt(R[:, 11].sum() / nret.sum() - dm**2) if nret.sum() else np.nan
    # invariant
    viol = 0
    if outs[0] == 1:
        viol = abs(held[0] - (nin[0] - nret[0]))
    d = np.diff(held) - (nin - nret)[1:]
    cont = np.diff(outs) == 1
    if d.size:
        viol = max(viol, np.abs(d[cont]).max() if cont.any() else 0)
    chk = {
        "n_time": (len(t), ds.sizes.get("hold_time")),
        "time": np.allclose(t, ds.hold_time.values),
        "sum_in": (nin.sum(), int(ds.hold_n_hold_in.sum())),
        "sum_ret": (nret.sum(), int(ds.hold_n_hold_ret.sum())),
        "esc_d": (ed.sum(), int(ds.hold_n_esc_direct.sum())),
        "esc_o": (eo.sum(), int(ds.hold_n_esc_orbit.sum())),
        "held_end": (held[-1], int(ds.hold_held_end.values[-1])),
        "viol": (viol, ds.attrs.get("hold_invariant_violation")),
        "rmax": (R[:, 12].max(), float(ds.hold_r_held_max.max())),
    }
    ok = all((v if isinstance(v, (bool, np.bool_)) else np.isclose(v[0], v[1])) for v in chk.values())
    print(f"{f.name:40s} files={sum(len(v) for v in raw.values())} {'OK' if ok else 'MISMATCH'} delay={dm:.4f}+-{dsd:.4f} "
          f"in/ret={nin.sum():.0f}/{nret.sum():.0f} esc={ed.sum():.0f}+{eo.sum():.0f} held_end={held[-1]:.0f} viol={viol} "
          f"rmax_alt_km={(R[:,12].max()-3389.5e3)/1e3:.4g}")
    if not ok:
        print("   ", chk)
