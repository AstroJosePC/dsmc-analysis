import sys, numpy as np
R = "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored/docs/reports/ubc_hold_carberry/scripts"
sys.path.insert(0, R)
import analysis as an
A = an.get()
SP = A.source_params()
AN = A.analytic()
R0 = SP["r0"]

print("== campaign Phi_ball/Phi_refl per top")
for z in (2500, 10000, 40000):
    r1 = A.get("campaign", 1, z, 250)
    d1 = A.reduce(r1)
    for u in (2, 5):
        rr = A.get("campaign", u, z, 250)
        dd = A.reduce(rr)
        t0 = max(dd.attrs["t0"], d1.attrs["t0"])
        q, qe, ka, kb = an.window_ratio(A.escape_series(rr, "run"), A.escape_series(r1, "run"), t0)
        print(z, u, f"t0={t0/1e3:.0f}ks ratio={q:.3f}+-{qe:.3f} pull={(q-1)/qe:.2f}")
    # reflect escape in phiJ0
print("== sweep pairs")
for z in (2500, 5000, 10000, 20000, 40000):
    for dr in (250, 1000):
        a2, a1 = A.get("sweep", 2, z, dr), A.get("sweep", 1, z, dr)
        d2, d1 = A.reduce(a2), A.reduce(a1)
        q, qe, _, _ = an.window_ratio(A.escape_series(a2), A.escape_series(a1), max(d2.attrs["t0"], d1.attrs["t0"]))
        print(z, dr, f"{q:.3f}+-{qe:.3f} pull {(q-1)/qe:.1f}")

print("== sweep profiles top 40000 dr250: ratio vs reflect below Kn0=1 exobase (1666 km)")
d1 = A.reduce(A.get("sweep", 1, 40000, 250))
for u in (2, 3):
    du = A.reduce(A.get("sweep", u, 40000, 250))
    for var in ("num_dens", "temp_trans_corr"):
        r, e, zz, ok = an.profile_compare(du, d1, var)
        alt = (du.radial.values - SP["R"]) / 1e3
        sel = alt < 1666
        print(u, var, "max|z| below 1666 km:", np.nanmax(np.abs(zz[sel])), "max|r-1|:", np.nanmax(np.abs(r[sel] - 1)))
        sel2 = alt < 588
        print(u, var, "  below 588 km:", np.nanmax(np.abs(zz[sel2])))
        # where does largest |z| occur
        j = np.nanargmax(np.abs(zz)); print("   largest |z|", zz[j], "at", alt[j], "ratio", r[j])

print("== held counts per campaign top (end and mean)")
for z in (2500, 10000, 40000):
    for u in (2, 5):
        d = A.reduce(A.get("campaign", u, z, 250))
        print(z, u, "held end", float(d.held_t.values[-1]), "held mean", d.attrs.get("held_mean"), "wall h", d.attrs["wall_s"] / 3600)
