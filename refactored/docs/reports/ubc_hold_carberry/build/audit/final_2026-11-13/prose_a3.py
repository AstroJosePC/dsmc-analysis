import sys, numpy as np
R = "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored/docs/reports/ubc_hold_carberry/scripts"
sys.path.insert(0, R)
import analysis as an
A = an.get()
SP = A.source_params()
print({k: SP[k] for k in SP if not hasattr(SP[k], '__len__')})
d1 = A.reduce(A.get("sweep", 1, 40000, 250))
print(d1.radial.values[:3], d1.radial.values[-3:], d1.radial.attrs)
RM = 3389.5
for u in (2, 3):
    du = A.reduce(A.get("sweep", u, 40000, 250))
    for var in ("num_dens", "temp_trans_corr"):
        r, e, zz, ok = an.profile_compare(du, d1, var)
        rad = du.radial.values
        alt = du.altitude.values; alt = alt/1e3 if alt.max() > 1e6 else alt
        for lim in (1666, 588):
            sel = alt < lim
            print(u, var, f"below {lim}: max|z|={np.nanmax(np.abs(zz[sel])):.2f} max|r-1|={np.nanmax(np.abs(r[sel]-1)):.3f} ncells={sel.sum()}")
        j = np.nanargmax(np.abs(zz)); print("   largest |z|", round(float(zz[j]), 2), "at alt", round(float(alt[j])), "ratio", round(float(r[j]), 3))
        # top cell
        jt = np.where(ok)[0][-1]; print("   top ok cell alt", round(float(alt[jt])), r[jt], e[jt], zz[jt])
# reflect top T
T = d1.p_temp_trans_corr.values if hasattr(d1, 'p_temp_trans_corr') else None
print([v for v in d1.data_vars][:40])
