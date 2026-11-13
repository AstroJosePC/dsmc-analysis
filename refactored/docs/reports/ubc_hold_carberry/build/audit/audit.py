"""Independent recomputation of selected report numbers (does NOT import scripts/analysis.py)."""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
sys.path.insert(0, "/Users/josepc/MarsAtmos-UBC-explicit/verification")
from dsmclib.ncrun import open_run_nc
from dsmclib import escape, exosphere, profiles
from hold_diagnostics import read_hold

G, KB, AMU = 6.674e-11, 1.3806503e-23, 1.66053892e-27
MM, RM = 6.4171e23, 3389.5e3
m = 2.02 * AMU
r0 = RM + 100e3; T0 = 270.0; n0 = 2.3e13; d = 2.92e-10
out = {}
lam = G * MM * m / (KB * T0 * r0); out["oLambdaZero"] = lam
H = KB * T0 * r0**2 / (G * MM * m); out["oScaleHZero_km"] = H / 1e3
l = 1 / (np.sqrt(2) * n0 * np.pi * d**2); out["oMfpZero_km"] = l / 1e3
out["oKnZero"] = l / r0; out["oKnH"] = l / H; out["oKnRatio"] = (l / r0) / 2.5e-3
cbar = np.sqrt(8 * KB * T0 / (np.pi * m))
out["oPhiJZero"] = np.pi * r0**2 * n0 * cbar * (1 + lam) * np.exp(-lam)
# Mars Hill radius (sun): a (m/3Ms)^(1/3)
rH = 1.52366231 * 1.495978707e11 * (MM / (3 * 1.98892e30)) ** (1 / 3)
out["oHill_r0"] = rH / r0; out["oHillKm"] = rH / 1e3
# Callisto
mc, r0c, T0c, n0c = 1.08e23, 2410e3, 155.0, 4e14
lc = G * mc * m / (KB * T0c * r0c); out["pLambdaZero"] = lc
out["pPhiJZero"] = np.pi * r0c**2 * n0c * np.sqrt(8 * KB * T0c / (np.pi * m)) * (1 + lc) * np.exp(-lc)
out["pHill"] = (mc / (3 * 1.898e27)) ** (1 / 3) * 26.3 * 71492e3 / r0c
# slowest return: radial orbit with apoapsis at r_Hill, from r_top(40000 km) and back
mu = G * MM
for z in (2627.3e3, 40134e3):
    rt = RM + z; a = rH / 2
    E = np.arccos(1 - rt / a)
    out[f"tReturnMax_from_{z/1e3:.0f}km"] = 2 * np.sqrt(a**3 / mu) * (np.pi - (E - np.sin(E)))
out["tReturnMax_days"] = out["tReturnMax_from_40134km"] / 86400
out["tReturnMax_orb"] = out["tReturnMax_from_40134km"] / 1.44e6

# sweep: runs at z 40000 dr250
SW = Path.home() / "MarsAtmos-H2UBC-sweep/runs"
def op(name):
    return open_run_nc(SW / name / "Results" / f"{name}_run.nc")
R = {u: op(f"H2_U{u}_z40000_dr250") for u in (1, 2, 3)}
rtop = None
for u, run in R.items():
    e = escape.dsmc_escape(run, 32e3)
    out[f"esc_U{u}_t32k"] = float(e.escape_rate.isel(species=0)); out[f"k_U{u}_t32k"] = int(e.escape_npart.isel(species=0))
    rtop = e.attrs["top_radius"]
out["rtop_40000_km_alt"] = (rtop - RM) / 1e3
phiJ = out["oPhiJZero"]
r12 = out["esc_U2_t32k"] / out["esc_U1_t32k"]
sig = r12 * np.sqrt(1 / out["k_U1_t32k"] + 1 / out["k_U2_t32k"])
out["sweepPhiRatioTop"] = (r12, sig)
out["phiOverJ_U1_40000"] = out["esc_U1_t32k"] / phiJ
out["phiOverJ_U3_40000"] = out["esc_U3_t32k"] / phiJ
# all finished reflect runs, Phi/phiJ0 from t0 = 32 ks (rough; report uses per-run steady windows)
vals = []
for p in sorted(SW.glob("H2_U1_z*_dr*")):
    meta = (p / "run_meta.txt").read_text()
    if "finished:" not in meta:
        continue
    f = p / "Results" / f"{p.name}_run.nc"
    if not f.exists():
        continue
    run = open_run_nc(f)
    vals.append((p.name, float(escape.dsmc_escape(run, 13e3).escape_rate.isel(species=0)) / phiJ))
out["phiOverJ_reflect_all_t13k"] = vals
# collisionless at r_top 40000
an = {u: exosphere.escape_rate(n0, T0, m, r0, rtop, ubc=u) for u in ("reflect", "ballistic", "free_escape")}
out["anBallisticExcess_pct"] = 100 * (an["ballistic"] / an["reflect"] - 1)
out["anFreeOverJTop"] = an["free_escape"] / phiJ
out["collEnhanceTop"] = out["esc_U1_t32k"] / an["reflect"]
# tau90 at 40000 top
hinf = exosphere.held_count(n0, T0, m, r0, rtop)
tt = np.logspace(4, 8, 400)
h = exosphere.held_count(n0, T0, m, r0, rtop, t=tt)
out["heldSteadyHi"] = hinf
out["tauFill_40000"] = float(np.interp(0.9, h / hinf, tt))
out["heldFracSweepEnd_pct"] = 100 * exosphere.held_count(n0, T0, m, r0, rtop, t=3e5) / hinf
out["heldFracCampEnd_pct"] = 100 * exosphere.held_count(n0, T0, m, r0, rtop, t=2e6) / hinf
# profiles + exobase
pr = {}
for u, run in R.items():
    run.macro["temp_trans_corr"] = profiles.temp_trans_corrected(run)
    pr[u] = profiles.time_mean_profiles(run, 32e3 if u == 1 else 26e3,
                                        fields=("num_dens", "temp_trans", "temp_trans_corr", "knudsen"))
p1 = pr[1].isel(species=0); p2 = pr[2].isel(species=0)
top = int(np.where(np.isfinite(p1.num_dens.values) & (p1.num_dens.values > 0))[0][-1])
out["profTopAltKm"] = float(pr[1].altitude[top]) / 1e3
out["nRatioKeplerTop"] = float(p2.num_dens[top] / p1.num_dens[top])
out["TRatioKeplerTop"] = float(p2.temp_trans_corr[top] / p1.temp_trans_corr[top])
out["TtopReflect"] = float(p1.temp_trans_corr[top]); out["TtopReflectRaw"] = float(p1.temp_trans[top])
zx = profiles.exobase(pr[1]); out["exoZReflectKn"] = float(zx.isel(species=0))
alt = pr[1].altitude.values / 1e3
out["exoTReflectKn"] = float(np.interp(out["exoZReflectKn"], alt, p1.temp_trans_corr.values)) / T0
out["exoRReflectKn"] = (RM + out["exoZReflectKn"] * 1e3) / r0
# verification hold: E7b delay; E8 returns
V = Path("/Users/josepc/MarsAtmos-UBC-explicit/verification/runs")
hb = read_hold(V / "E7b_explicit").isel(species=0)
n = hb.n_hold_ret.values.astype(float)
s1 = np.nansum(hb.ret_dt_mean.values * n); s2 = np.nansum((hb.ret_dt_std.values**2 + hb.ret_dt_mean.values**2) * n)
mean = s1 / n.sum(); out["verDelayMean"] = mean; out["verDelayStd"] = np.sqrt(s2 / n.sum() - mean**2)
out["verRheldMax_km_alt"] = (float(hb.r_held_max.max()) - RM) / 1e3
h8 = read_hold(V / "E8_main_kepler").isel(species=0)
out["eightReturns"] = int(h8.n_hold_ret.sum()); out["eightHeldEnd"] = int(h8.held_end[-1]); out["eightIn"] = int(h8.n_hold_in.sum())
# E7a escape vs analytic
e7a = open_run_nc(sorted((V / "E7a_kepler" / "Results").glob("*_L1.nc"))[0])
e = escape.dsmc_escape(e7a, 30e3)
rt7 = e.attrs["top_radius"]
out["verZtopKm"] = (rt7 - RM) / 1e3
out["verEscRatioKepler"] = float(e.escape_rate.isel(species=0)) / exosphere.escape_rate(n0, T0, m, r0, rt7, ubc="ballistic")
# campaign live: explicit/Kepler escape over common time
C = Path.home() / "MarsAtmos-H2Hold-campaign/runs"
hk = read_hold(C / "H2_U2_z40000_dr250").isel(species=0); he = read_hold(C / "H2_U5_z40000_dr250").isel(species=0)
tc = min(float(hk.time[-1]), float(he.time[-1]))
ek = int((hk.n_esc_direct + hk.n_esc_orbit).where(hk.time <= tc, 0).sum())
ee = int((he.n_esc_direct + he.n_esc_orbit).where(he.time <= tc, 0).sum())
out["camp_tc"] = tc; out["prodEscRatioTop"] = (ee / ek, ee / ek * np.sqrt(1 / ee + 1 / ek))
out["camp_returns_K_E"] = (int(hk.n_hold_ret.sum()), int(he.n_hold_ret.sum()))
out["camp_weight"] = float(hk.part_weight)
for k, v in out.items():
    print(f"{k:28s} {v}")
