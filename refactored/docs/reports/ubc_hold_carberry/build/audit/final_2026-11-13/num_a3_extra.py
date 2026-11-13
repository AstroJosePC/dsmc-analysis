"""Sweep profile macros, timescales table, restart check."""
import sys
from pathlib import Path
import numpy as np
sys.argv = ["x"]
exec(open(Path(__file__).parent / "a2_main.py").read().split("# ================================================================ campaign")[0])

print("=== sweep profiles at z_top 40000 dr250")
S = {u: load(SA / f"H2_U{u}_z40000_dr250_analysis.nc").sel(species="H2") for u in (1, 2, 3)}
ref = S[1]
for u, nm in ((2, "Kepler"), (3, "Free")):
    rn, en, zn, ok = pcomp(S[u], ref, "num_dens")
    rT, eT, zT, _ = pcomp(S[u], ref, "temp_trans_corr")
    j = np.where(ok)[0][-1]
    chk(f"nRatio{nm}Top", [rn[j], en[j]])
    chk(f"TRatio{nm}Top", [rT[j], eT[j]])
    chk(f"nZmax{nm}", [np.nanmax(np.abs(zn))])
    chk(f"TZmax{nm}", [np.nanmax(np.abs(zT))])
    chk(f"nRatio{nm}MaxDev", [100 * np.nanmax(np.abs(rn - 1))])
    chk(f"TRatio{nm}MaxDev", [100 * np.nanmax(np.abs(rT - 1))])
    print(f"   {nm}: last ok cell {j}/{len(ok)} alt {float(ref.altitude[j])/1e3:.0f} km")
chk("profTopAltKm", [float(ref.altitude[j]) / 1e3])
chk("profWindowReflect", [ref.attrs["profile_window_t0"] / 1e3])
chk("TtopReflect", [float(ref.profile_temp_trans_corr[j])])
chk("TtopReflectRaw", [float(ref.profile_temp_trans[j])])

print("=== timescales")
chk("tauFillLo", [TAU[i_low]]); chk("tauFillHi", [TAU[i_top]])
chk("tauFillLoOrb", [TAU[i_low] / 1.44e6]); chk("tauFillHiOrb", [TAU[i_top] / 1.44e6])
tr = AN.t_return_max.values.max()
chk("tReturnMax", [tr]); chk("tReturnMaxOrb", [tr / 1.44e6]); chk("tReturnMaxDays", [tr / 86400])
chk("heldSteadyLo", [float(AN.held_steady[i_top])]); chk("heldSteadyHi", [float(AN.held_steady[i_low])])
chk("sweepTendOverTau", [3e5 / TAU[i_top]])
# independent tau90 from held_t
for i in range(len(ZT)):
    fr = AN.held_t.values[i] / AN.held_steady.values[i]
    k = int(np.argmax(fr >= 0.9))
    t90 = np.exp(np.interp(0.9, fr[k - 1:k + 1], np.log(AN.t.values[k - 1:k + 1])))
    # longest return independently: radial orbit with apoapsis at r_Hill
    G = 6.6743e-11
    rt = Rp + ZT[i] * 1e3
    a = AN.attrs["hill_m"] / 2
    mu = G * SP["M"]
    E = np.arccos(1 - rt / a)
    tret = 2 * np.sqrt(a**3 / mu) * (np.pi - (E - np.sin(E)))
    print(f"   ztop {ZT[i]:.1f}: tau90 file {TAU[i]:.4g} recomputed {t90:.4g}; t_return file {AN.t_return_max.values[i]:.4g} recomputed {tret:.4g}")
tabt = [l for l in (REP / "tables/tab_timescales.tex").read_text().splitlines() if not l.startswith("%")]
for i, line in enumerate(tabt):
    rin = float(AN.escz_free_escape[i] - AN.escz_ballistic[i])
    chk(f"tab_timescales {ZT[i]:.0f}", [ZT[i], (Rp + ZT[i] * 1e3) / R0, float(AN.held_steady[i]), rin, TAU[i], TAU[i] / 1.44e6,
                                        float(AN.t_return_max[i]), 100 * (float(AN.escz_ballistic[i] / AN.escz_reflect[i]) - 1)], reported=line)

print("=== restart provenance")
b = load(HA / "E7b_explicit_analysis.nc"); d = load(HA / "E7d_explicit_restart_analysis.nc")
n = 15
print("   E7b hold_in first 16:", b.hold_n_hold_in.values[:16])
print("   E7d hold_in first 16:", d.hold_n_hold_in.values[:16])
print("   E7d time[:3]", d.time.values[:3], "hold_time[:3]", d.hold_time.values[:3], "outputs", d.hold_output.values[:3])
print("   identical first 15 hold_in:", np.array_equal(b.hold_n_hold_in.values[:15], d.hold_n_hold_in.values[:15]),
      " N_ball_phys first 15 equal:", np.allclose(b.N_ball_phys.values[:15], d.N_ball_phys.values[:15]))
print("==== non-OK")
for nme, s, v, r in RES:
    if s != "OK":
        print(nme, s, v, r)
