"""Independent recomputation of campaign, verification and sweep macros/tables of ubc_hold_carberry."""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
REP = ROOT / "docs/reports/ubc_hold_carberry"
HA = ROOT / "outputs/h2_hold/analysis"
SA = ROOT / "outputs/h2_ubc_rtop/analysis"

# ---------------------------------------------------------------- reported macros
MAC = {}
for line in (REP / "numbers.tex").read_text().splitlines():
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
    if m:
        MAC[m.group(1)] = m.group(2)


def nums(s):
    """Extract numeric values from a LaTeX macro string."""
    s = s.replace("\u2212", "-").replace("\\,", "")
    out = []
    for m in re.finditer(r"\\sci\{([-\d.]+)\}\{([-\d]+)\}|\\num\{([-\d.]+)\}", s):
        if m.group(1):
            out.append(float(m.group(1)) * 10 ** int(m.group(2)))
        else:
            out.append(float(m.group(3)))
    return out


RES = []


def chk(name, vals, reported=None, tol_digits=None):
    """Compare recomputed values (list) with numbers parsed from macro (or provided reported string)."""
    rep = MAC.get(name) if reported is None else reported
    if rep is None:
        RES.append((name, "MISSING", vals, None))
        print(f"  ?? {name}: macro missing; computed {vals}")
        return
    rv = nums(rep)
    vals = list(np.atleast_1d(vals))
    status = "OK"
    if len(rv) != len(vals):
        status = "SHAPE"
    else:
        for a, b in zip(vals, rv):
            # tolerance: half unit of the last printed digit of b (from string) + tiny
            sb = None
            status_i = None
            if not np.isfinite(a):
                status_i = "NAN"
            else:
                # decimal places in reported
                pass
            if status_i:
                status = status_i
    # digit-aware tolerance
    if status == "OK":
        tok = re.findall(r"\\sci\{([-\d.]+)\}\{([-\d]+)\}|\\num\{([-\d.\\,\u2212]+)\}", rep)
        for (a, b, t) in zip(vals, rv, tok):
            if t[0]:
                dec = len(t[0].split(".")[1]) if "." in t[0] else 0
                unit = 10 ** (int(t[1]) - dec)
            else:
                st = t[2].replace("\\,", "")
                dec = len(st.split(".")[1]) if "." in st else 0
                unit = 10 ** (-dec)
            d = abs(a - b)
            if d > 0.5 * unit * (1 + 1e-9) + 1e-12:
                status = "ROUND" if d <= 1.0 * unit else "MISMATCH"
                if status == "MISMATCH":
                    break
    RES.append((name, status, vals, rep))
    flag = "  " if status == "OK" else "!!"
    print(f"{flag} {name:28s} {status:8s} computed={[float(f'{v:.6g}') for v in vals]} reported={rep}")


def load(p):
    with xr.open_dataset(p) as d:
        return d.load()


AN = load(HA / "h2_hold_analysis.nc")
SP = {k[7:]: v for k, v in AN.attrs.items() if k.startswith("source_") and k not in ("source_files", "source_sha256")}
R0, Rp, phiJ0 = SP["r0"], SP["R"], SP["phiJ0"]
TAU = AN.tau90.values
ZT = AN.ztop.values
i_top, i_low = int(np.argmax(ZT)), int(np.argmin(ZT))
logt = np.log(AN.t.values)


def zidx(rtop_m):
    return int(np.argmin(np.abs(ZT - (rtop_m - Rp) / 1e3)))


def held_an(i, t):
    return float(np.interp(np.log(t), logt, AN.held_t.values[i]))


def pcomp(a, b, field, minp=10):
    ya, yb = a[f"profile_{field}"].values, b[f"profile_{field}"].values
    ea, eb = a[f"profile_{field}_sem"].values, b[f"profile_{field}_sem"].values
    pa = a.profile_num_part.values
    pb = b.profile_num_part.values
    ok = (pa >= minp) & (pb >= minp) & (yb > 0)
    r = ya / yb
    e = r * np.sqrt((ea / ya) ** 2 + (eb / yb) ** 2)
    z = (ya - yb) / np.sqrt(ea**2 + eb**2)
    r[~ok] = e[~ok] = z[~ok] = np.nan
    return r, e, z, ok


def wratio(ta, na, ra, da, tb, nb, rb, db, t0):
    """Ratio of dt-weighted mean rates over common output times > t0."""
    sa, sb = ta > t0, tb > t0
    common = np.intersect1d(ta[sa], tb[sb])
    ia = np.isin(ta, common)
    ib = np.isin(tb, common)
    Ra = (ra[ia] * da[ia]).sum() / da[ia].sum()
    Rb = (rb[ib] * db[ib]).sum() / db[ib].sum()
    ka, kb = na[ia].sum(), nb[ib].sum()
    q = Ra / Rb
    return q, q * np.sqrt(1 / max(ka, 1) + 1 / kb), ka, kb


def an_inret(i):
    r_in = float(AN.escz_free_escape[i] - AN.escz_ballistic[i])
    t = AN.t.values
    H = AN.held_t.values[i]
    dH = np.gradient(H, t)
    ret = r_in - dH
    ratio = np.where((H > 0) & (ret > 0), r_in / ret, np.nan)
    return t, ratio, r_in


# ================================================================ campaign
print("=== campaign / production")
C = {(u, z): load(HA / f"Hold_H2_U{u}_z{z}_dr250_analysis.nc") for u in (1, 2, 5) for z in (2500, 10000, 40000)}
TN = {2500: "Low", 10000: "Mid", 40000: "Top"}
prod_tab = []
agg = dict(zn=[], zT=[], hr=[], wr=[], ha=[], irc=[], ira=[], phi=[])
for z in (2500, 10000, 40000):
    d1, d2, d5 = C[(1, z)], C[(2, z)], C[(5, z)]
    tn = TN[z]
    rn, en, zn, ok = pcomp(d5, d2, "num_dens")
    rT, eT, zT, okT = pcomp(d5, d2, "temp_trans_corr")
    zmn, zmT = np.nanmax(np.abs(zn)), np.nanmax(np.abs(zT))
    e2 = float(d2.hold_n_esc_direct.sum() + d2.hold_n_esc_orbit.sum())
    e5 = float(d5.hold_n_esc_direct.sum() + d5.hold_n_esc_orbit.sum())
    in2, in5 = float(d2.hold_n_hold_in.sum()), float(d5.hold_n_hold_in.sum())
    # held mean over each run's own window t > window_t0
    hm = lambda d: float(np.nanmean(d.N_ball_phys.values[d.time.values > d.attrs["window_t0"]]))
    hr = hm(d5) / hm(d2)
    # cross-check the attribute
    assert np.isclose(hm(d5), d5.attrs["held_mean_window"]) and np.isclose(hm(d2), d2.attrs["held_mean_window"])
    nr = d5.hold_n_hold_ret.values.astype(float)
    # pooled delay from per-interval mean/std (sum and sumsq reconstruction)
    mu = np.nan_to_num(d5.hold_ret_dt_mean.values)
    sd = np.nan_to_num(d5.hold_ret_dt_std.values)
    S1 = (nr * mu).sum()
    S2 = (nr * (sd**2 + mu**2)).sum()
    dm = S1 / nr.sum()
    dsd = np.sqrt(S2 / nr.sum() - dm**2)
    # wall: from run_meta.txt independently
    def wall(d):
        meta = Path(d.attrs["source_files"]).parent.parent / "run_meta.txt"
        kv = dict(re.findall(r"^(\w+):\s+(.*)$", meta.read_text(), re.M))
        return (datetime.fromisoformat(kv["finished"].strip()) - datetime.fromisoformat(kv["started"].strip())).total_seconds() / 3600
    w2, w5 = wall(d2), wall(d5)
    i = zidx(d2.attrs["r_top_m"])
    t_end = float(d2.time.values[-1])
    h_an = held_an(i, t_end)
    ha2, ha5 = float(d2.N_ball_phys.values[-1]) / h_an, float(d5.N_ball_phys.values[-1]) / h_an
    t_a, rat_a, _ = an_inret(i)
    sel = (t_a > 1e6) & (t_a <= t_end) & np.isfinite(rat_a)
    ra_late = float(np.mean(rat_a[sel]))
    # alternative: time-weighted mean of rates (R_in * T / (R_in*T - dH)) over (1e6, t_end]
    r_in = float(AN.escz_free_escape[i] - AN.escz_ballistic[i])
    dH = held_an(i, t_end) - held_an(i, 1e6)
    ra_alt = r_in * (t_end - 1e6) / (r_in * (t_end - 1e6) - dH)
    ir = []
    for d in (d2, d5):
        late = d.hold_time.values > 1e6
        ir.append(float(d.hold_n_hold_in.values[late].sum() / d.hold_n_hold_ret.values[late].sum()))
    # escape ratio vs reflect from run escape series over common steady window
    for d, u in ((d2, 2), (d5, 5)):
        q, qe, _, _ = wratio(d.time.values, d.escape_npart.values, d.escape_rate.values, d.avg_time_interval.values,
                             d1.time.values, d1.escape_npart.values, d1.escape_rate.values, d1.avg_time_interval.values,
                             max(d.attrs["window_t0"], d1.attrs["window_t0"]))
        agg["phi"].append((z, u, q, qe))
    print(f" z={z}: i_ztop={i} ({ZT[i]:.1f} km) t_end={t_end:g}  anInRet(log-grid mean)={ra_late:.4f}  time-avg alt={ra_alt:.4f}  "
          f"n_grid_pts={sel.sum()}")
    chk(f"prodZmaxN{tn}", [zmn])
    chk(f"prodZmaxT{tn}", [zmT])
    chk(f"prodWallRatio{tn}", [w5 / w2])
    chk(f"prodWallKepler{tn}", [w2])
    chk(f"prodWallExplicit{tn}", [w5])
    chk(f"campHeldOverAnalytic{tn}", [ha2, ha5])
    chk(f"campInRet{tn}", ir)
    chk(f"anInRetCamp{tn}", [ra_late])
    chk(f"campHeldFracAn{tn}", [100 * h_an / float(AN.held_steady[i])])
    er = e5 / e2
    row = [z, er, er * np.sqrt(1 / e5 + 1 / e2), hr, in5 / in2, zmn, zmT, dm, dsd,
           float((d5.hold_r_held_max.max() - Rp) / 1e3), w2, w5, w5 / w2]
    prod_tab.append(row)
    agg["zn"].append(zmn); agg["zT"].append(zmT); agg["hr"].append(hr); agg["wr"].append(w5 / w2)
    agg["ha"] += [ha2, ha5]; agg["irc"] += ir; agg["ira"].append(ra_late)
    if z == 40000:
        j = np.where(ok)[0][-1]
        chk("prodEscRatioTop", [er, er * np.sqrt(1 / e5 + 1 / e2)])
        chk("nRatioExplicitTop", [rn[j], en[j]])
        chk("TRatioExplicitTop", [rT[j], eT[j]])
        chk("nZmaxExplicit", [zmn])
        chk("TZmaxExplicit", [zmT])
        chk("prodSimReachedTop", [t_end])
        for dd, nm in ((d2, "Kepler"), (d5, "Explicit")):
            cn, cen, _, okc = pcomp(dd, d1, "num_dens")
            cT, ceT, _, _ = pcomp(dd, d1, "temp_trans_corr")
            jc = np.where(okc)[0][-1]
            print(f"   top cell for {nm} vs reflect: idx {jc} of {len(okc)}, alt {float(dd.altitude[jc])/1e3:.0f} km; "
                  f"top-cell npart K/E/R={float(dd.profile_num_part[jc]):.0f}/{float(d1.profile_num_part[jc]):.0f}")
            chk(f"nRatioCamp{nm}Top", [cn[jc], cen[jc]])
            chk(f"TRatioCamp{nm}Top", [cT[jc], ceT[jc]])
chk("prodZmaxN", [max(agg["zn"])])
chk("prodZmaxT", [max(agg["zT"])])
chk("prodHeldRatioRange", [min(agg["hr"]), max(agg["hr"])])
chk("prodWallRatioRange", [min(agg["wr"]), max(agg["wr"])])
chk("campHeldOverAnalyticRange", [min(agg["ha"]), max(agg["ha"])])
chk("campInRetRange", [min(agg["irc"]), max(agg["irc"])])
chk("anInRetCampRange", [min(agg["ira"]), max(agg["ira"])])
chk("prodNTops", [3])
chk("campPhiRatioRange", [min(x[2] for x in agg["phi"]), max(x[2] for x in agg["phi"])])
chk("campPhiRatioPullMax", [max(abs(x[2] - 1) / x[3] for x in agg["phi"])])
print("   phi_c:", [(z, u, round(q, 4), round(e, 4)) for z, u, q, e in agg["phi"]])
chk("campViolMax", [max(float(C[k].attrs["hold_invariant_violation"]) for k in C)])
chk("campReflectInEqRet", [], reported="")  # placeholder; checked below
refl_ok = all(bool((C[(1, z)].hold_n_hold_in == C[(1, z)].hold_n_hold_ret).all()) for z in (2500, 10000, 40000))
print(f"   campReflectInEqRet computed={'yes' if refl_ok else 'no'} reported={MAC['campReflectInEqRet']}")
chk("campTlast", [max(float(C[k].hold_time[-1]) for k in C)])
chk("campTend", [2e6])
chk("campTendOverTau", [2e6 / TAU[i_top]])
chk("heldFracCampEnd", [100 * held_an(i_top, 2e6) / float(AN.held_steady[i_top])])
chk("heldFracSweepEnd", [100 * held_an(i_top, 3e5) / float(AN.held_steady[i_top])])
chk("heldFracVerEnd", [100 * held_an(i_low, 6e4) / float(AN.held_steady[i_low])])
chk("campLateKs", [1000])

# production table
print("--- tab_production.tex")
tab = [l for l in (REP / "tables/tab_production.tex").read_text().splitlines() if not l.startswith("%")]
for row, line in zip(prod_tab, tab):
    z, er, ee, hr, irr, zmn, zmT, dm, dsd, rmax, w2, w5, wr = row
    chk(f"tab_production z={z}", [z, er, ee, hr, irr, zmn, zmT, dm, dsd, rmax, w2, w5, wr], reported=line)

# ================================================================ verification
print("=== verification")
V = json.loads(AN.attrs["verification_json"])
E = {n: load(HA / f"{n}_analysis.nc") for n in ("E7a_kepler", "E7b_explicit", "E7c_explicit_dbg", "E7d_explicit_restart",
                                                 "E8_main_kepler", "E9_reflect")}
T0V = 3.0e4
import importlib
ex = importlib.import_module("dsmclib.exosphere")  # analytic escape: library function (not report helper)
for n, nm, ubc in (("E7a_kepler", "Kepler", "ballistic"), ("E7b_explicit", "Explicit", "ballistic"), ("E9_reflect", "Reflect", "reflect")):
    d = E[n]
    nn, nse, T = d.profile_num_dens.values, d.profile_num_dens_sem.values, d.profile_temp_trans_corr.values
    na, Ta = d.analytic_num_dens.values, d.analytic_temp_trans.values
    g = d.profile_num_part.values > 5
    esc = d.attrs["dsmc_escape_rate"]
    k = d.attrs["dsmc_escape_npart"]
    # recompute esc from series over t > T0V
    t = d.time.values
    w = t > T0V
    esc_s = (d.escape_rate.values[w] * d.avg_time_interval.values[w]).sum() / d.avg_time_interval.values[w].sum()
    k_s = d.escape_npart.values[w].sum()
    esc_an = ex.escape_rate(SP["n0"], SP["T0"], SP["m"], d.attrs["r0_m"], d.attrs["r_top_m"], ubc=ubc,
                            planet_mass=d.attrs["planet_mass"])
    print(f" {n}: esc attr {esc:.5g} vs series {esc_s:.5g}; k attr {k} vs series {k_s:.1f}; esc_an {esc_an:.5g} (json {V['analytic'][n]['esc_an']:.5g})")
    chk(f"verDnMax{nm}", [100 * np.nanmax(np.abs((nn - na) / na)[g])])
    chk(f"verZnMax{nm}", [np.nanmax(np.abs((nn - na) / nse)[g])])
    chk(f"verDTMax{nm}", [100 * np.nanmax(np.abs((T - Ta) / Ta)[g])])
    chk(f"verEscZ{nm}", [(esc - esc_an) / (esc_an / np.sqrt(k))])
    chk(f"verEscRatio{nm}", [esc / esc_an])
    chk(f"verEscK{nm}", [k])
    if ubc == "ballistic":
        held = float(d.N_ball_phys.values[w].mean())
        han = float(np.mean(ex.held_count(SP["n0"], SP["T0"], SP["m"], d.attrs["r0_m"], d.attrs["r_top_m"],
                                          planet_mass=d.attrs["planet_mass"], t=t[w])))
        chk(f"verHeldRatio{nm}", [held / han])
        late = d.hold_time.values > T0V
        chk(f"verInRet{nm}", [float(d.hold_n_hold_in.values[late].sum() / d.hold_n_hold_ret.values[late].sum())])
pa, pb = E["E7a_kepler"], E["E7b_explicit"]
zz = (pa.profile_num_dens - pb.profile_num_dens) / np.hypot(pa.profile_num_dens_sem, pb.profile_num_dens_sem)
chk("verKvsEZ", [float(np.abs(zz).max())])
hk = float(pa.N_ball_phys.values[pa.time.values > T0V].mean())
he = float(pb.N_ball_phys.values[pb.time.values > T0V].mean())
chk("verKvsEHeld", [100 * (he / hk - 1)])
d = E["E7b_explicit"]
nr = d.hold_n_hold_ret.values.astype(float)
mu, sd = np.nan_to_num(d.hold_ret_dt_mean.values), np.nan_to_num(d.hold_ret_dt_std.values)
dm = (nr * mu).sum() / nr.sum()
chk("verDelayMean", [dm])
chk("verDelayStd", [np.sqrt((nr * (sd**2 + mu**2)).sum() / nr.sum() - dm**2)])
chk("verDelayUniformStd", [1 / np.sqrt(12)])
chk("verRheldMax", [float((d.hold_r_held_max.max() - Rp) / 1e3)])
chk("verViolMax", [max(float(x.attrs["hold_invariant_violation"]) for x in E.values())])
chk("verNHoldRuns", [sum("hold_time" in x.dims for x in E.values())])
chk("verZtopKm", [(E["E7a_kepler"].attrs["r_top_m"] - Rp) / 1e3])
chk("verTend", [float(E["E7a_kepler"].time[-1])])
d8 = E["E8_main_kepler"]
th = d8.hold_time.values
chk("eightGrowth", [np.polyfit(th, d8.hold_held_end.values.astype(float), 1)[0]])
chk("eightEntry", [float(d8.hold_n_hold_in.sum()) / th[-1]])
chk("eightReturns", [int(d8.hold_n_hold_ret.sum())])
chk("eightHeldEnd", [int(d8.hold_held_end.values[-1])])
# restart: only from json (needs run files) -> check macro vs json
chk("verRestartZ", [V["restart"]["zmax"]])
chk("verRestartHeld", [100 * (V["restart"]["held_d"] / V["restart"]["held_b"] - 1)])
# restart independent: from analysis-tier N_ball_phys in window (15,30] ks
db, dd = E["E7b_explicit"], E["E7d_explicit_restart"]
wb = (db.time.values > 1.5e4) & (db.time.values <= 3e4)
wd = (dd.time.values > 1.5e4) & (dd.time.values <= 3e4)
print(f"   restart held from analysis series: b={db.N_ball_phys.values[wb].mean():.5g} d={dd.N_ball_phys.values[wd].mean():.5g} "
      f"-> {100*(dd.N_ball_phys.values[wd].mean()/db.N_ball_phys.values[wb].mean()-1):.3f}%  (json b={V['restart']['held_b']:.5g} d={V['restart']['held_d']:.5g}); "
      f"E7d time[0]={dd.time.values[0]}, t_restart json={V['restart']['t_restart']}")
# phi ratio E7a/E7b vs E9 with hold escape series
s9 = E["E9_reflect"]


def hold_series(d):
    n = (d.hold_n_esc_direct + d.hold_n_esc_orbit).values.astype(float)
    dt = d.hold_avg_time_interval.values
    return d.hold_time.values, n, n * d.attrs["hold_part_weight"] / dt, dt


t9, n9, r9, dt9 = hold_series(s9)
for n, nm in (("E7a_kepler", "Kepler"), ("E7b_explicit", "Explicit")):
    ta, na_, ra, dta = hold_series(E[n])
    q, qe, _, _ = wratio(ta, na_, ra, dta, t9, n9, r9, dt9, T0V)
    chk(f"verPhiRatio{nm}", [q, qe])
# table
print("--- tab_verification.tex")
tabv = [l for l in (REP / "tables/tab_verification.tex").read_text().splitlines() if not l.startswith("%")]
for n, line in zip(E, tabv):
    d = E[n]
    vals = [d.sizes["hold_time"], float(d.attrs["hold_invariant_violation"]), int(d.hold_n_hold_in.sum()), int(d.hold_n_hold_ret.sum()),
            int(d.hold_n_esc_direct.sum()), int(d.hold_n_esc_orbit.sum()), int(d.hold_held_end.values[-1])]
    nr = d.hold_n_hold_ret.values.astype(float)
    mu, sd = np.nan_to_num(d.hold_ret_dt_mean.values), np.nan_to_num(d.hold_ret_dt_std.values)
    if nr.sum() > 0 and np.any(mu != 0):
        dm = (nr * mu).sum() / nr.sum()
        vals += [dm, np.sqrt((nr * (sd**2 + mu**2)).sum() / nr.sum() - dm**2)]
    if n in V["analytic"]:
        vals.append(V["analytic"][n]["esc_z"])
    chk(f"tab_verification {n}", vals, reported=line)
# analytic Fig-3 numbers
i_v = zidx(E["E7a_kepler"].attrs["r_top_m"])
t_a, rat_a, rin = an_inret(i_v)
chk("anInRetVerEnd", [float(np.interp(np.log(6e4), np.log(t_a), rat_a))])
chk("anRinVer", [rin])
chk("anPhiRatioVer", [float(AN.escz_ballistic[i_v] / AN.escz_reflect[i_v])])

# ================================================================ sweep
print("=== sweep")
S = {}
for u in (1, 2, 3):
    for z in (2500, 5000, 10000, 20000, 40000):
        for dr in (250, 1000):
            d = load(SA / f"H2_U{u}_z{z}_dr{dr}_analysis.nc").sel(species="H2")
            S[(u, z, dr)] = d
phi = {}
rows = []
for (u, z, dr), d in S.items():
    t = d.time.values
    dt = np.diff(t, prepend=0.0)
    t0, t1 = d.attrs["profile_window_t0"], d.attrs["profile_window_t1"]
    w = (t > t0) & (t <= t1)
    Phi_s = (d.escape_rate.values[w] * dt[w]).sum() / dt[w].sum()
    k_s = d.escape_npart.values[w].sum()
    Phi = float(d.dsmc_escape_rate)
    k = int(d.dsmc_escape_npart)
    if not np.isclose(Phi, Phi_s, rtol=1e-6) or abs(k - k_s) > 0.5:
        print(f"   !! {u},{z},{dr}: attr Phi {Phi:.6g} k {k} vs series {Phi_s:.6g} k {k_s}")
    rt = float(d.attrs["r_top_m"])
    phi[(u, z, dr)] = (Phi, k, rt, d, t, dt)
    rows.append((u, z, dr, rt / R0, t0 / 1e3, k, Phi, Phi / phiJ0, Phi / phiJ0 / np.sqrt(k)))
tabs = [l for l in (REP / "tables/tab_escape_sweep.tex").read_text().splitlines() if not l.startswith("%") and not l.startswith("\\addlinespace")]
print("--- tab_escape_sweep.tex (cols 2-8; analytic column checked separately)")
for r, line in zip(rows, tabs):
    rep = "&".join(line.split("&")[1:8])
    chk(f"tab_escape_sweep U{r[0]} z{r[1]} dr{r[2]}", [r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]], reported=rep)
# analytic column via dsmclib, n_v=900
for r, line in zip(rows, tabs):
    u = r[0]
    ana = ex.escape_rate(SP["n0"], SP["T0"], SP["m"], R0, phi[(u, r[1], r[2])][2],
                         ubc={1: "reflect", 2: "ballistic", 3: "free_escape"}[u], planet_mass=SP["M"], n_v=900)
    rep = line.split("&")[8]
    chk(f"tab_escape_sweep ana U{u} z{r[1]} dr{r[2]}", [ana / phiJ0], reported=rep)
for u, nm in ((1, "Reflect"), (2, "Kepler"), (3, "Free")):
    v = [p[0] / phiJ0 for k_, p in phi.items() if k_[0] == u]
    chk(f"phiOverJ{nm}", [min(v), max(v)])
chk("phiOverJFreeTop", [phi[(3, 40000, 250)][0] / phiJ0])
chk("phiOverJFreeLowZ", [phi[(3, 2500, 250)][0] / phiJ0])
chk("collEnhanceTop", [phi[(1, 40000, 250)][0] / float(AN.escz_reflect[i_top])])
print(f"   collEnhanceTop: Phi_U1(40000,250)={phi[(1,40000,250)][0]:.4g}, escz_reflect[top]={float(AN.escz_reflect[i_top]):.4g}, "
      f"ztop[i_top]={ZT[i_top]:.1f} km vs run r_top alt {(phi[(1,40000,250)][2]-Rp)/1e3:.1f} km")
chk("anReflectOverJ", [float(AN.escz_reflect[i_top]) / phiJ0])
chk("anBallisticOverJ", [float(AN.escz_ballistic[i_top]) / phiJ0])
chk("anFreeOverJTop", [float(AN.escz_free_escape[i_top]) / phiJ0])
chk("anBallisticExcess", [100 * (float(AN.escz_ballistic[i_top] / AN.escz_reflect[i_top]) - 1)])
kr = [phi[(2, z, dr)][0] / phi[(1, z, dr)][0] for (u, z, dr) in phi if u == 2]
chk("keplerOverReflectRange", [min(kr), max(kr)])
# dr effect, matched-top rule
dres, skip, allres = [], set(), []
for (u, z, dr) in phi:
    if dr != 250:
        continue
    a, b = phi[(u, z, 250)], phi[(u, z, 1000)]
    x = a[0] / b[0] - 1
    allres.append((u, z, round(100 * x, 2), round((a[2] - b[2]) / R0, 4)))
    if abs(a[2] - b[2]) < 0.01 * R0:
        dres.append(x)
    else:
        skip.add(z)
print("   dr pairs (u,z,%diff,dRtop/R0):", allres)
chk("drEffectMax", [100 * max(abs(x) for x in dres)])
chk("drSkipZtop", sorted(skip))
# reflect/kepler-only dr effect (excluding free escape) for context
print("   drEffectMax U1/U2 only:", round(100 * max(abs(x[2]) / 100 for x in allres if x[0] in (1, 2) and abs(x[3]) < 0.01), 2))
# esc drift
drift, late, late_t = [], {}, []
for key, (Phi, k, rt, d, t, dt) in phi.items():
    tl = t[-1] * (1 - 1 / 3)
    w = t > tl
    rl = (d.escape_rate.values[w] * dt[w]).sum() / dt[w].sum()
    late[key] = rl
    late_t.append(tl)
    drift.append((key, 100 * (rl / Phi - 1)))
chk("escDriftLo", [min(x[1] for x in drift)])
chk("escDriftHi", [max(x[1] for x in drift)])
print("   drift extremes:", min(drift, key=lambda x: x[1]), max(drift, key=lambda x: x[1]))
chk("escLateStartKs", [min(late_t) / 1e3])
chk("escLateReflectRange", [min(v / phiJ0 for k_, v in late.items() if k_[0] == 1), max(v / phiJ0 for k_, v in late.items() if k_[0] == 1)])
chk("escLateCollEnhanceTop", [late[(1, 40000, 250)] / float(AN.escz_reflect[i_top])])
lr = []
for (u, z, dr) in late:
    if u == 2:
        a, b = phi[(2, z, dr)], phi[(1, z, dr)]
        lr.append(wratio(a[4], a[3].escape_npart.values, a[3].escape_rate.values, a[5],
                         b[4], b[3].escape_npart.values, b[3].escape_rate.values, b[5], min(late_t))[0])
chk("sweepPhiRatioLateRange", [min(lr), max(lr)])
swr, swh = [], []
for z in (2500, 5000, 10000, 20000, 40000):
    for dr in (250, 1000):
        a, b = phi[(2, z, dr)], phi[(1, z, dr)]
        t0 = max(a[3].attrs["profile_window_t0"], b[3].attrs["profile_window_t0"])
        q, qe, _, _ = wratio(a[4], a[3].escape_npart.values, a[3].escape_rate.values, a[5],
                             b[4], b[3].escape_npart.values, b[3].escape_rate.values, b[5], t0)
        swr.append((z, dr, q, qe))
        if dr == 250:
            i = zidx(a[2])
            swh.append((z, float(a[3].N_ball_phys.values[-1]) / held_an(i, a[4][-1]), i))
chk("sweepPhiRatioRange", [min(x[2] for x in swr), max(x[2] for x in swr)])
top = [x for x in swr if x[0] == 40000 and x[1] == 250][0]
chk("sweepPhiRatioTop", [top[2], top[3]])
chk("sweepPhiRatioPullMax", [max(abs(x[2] - 1) / x[3] for x in swr)])
chk("nSweepPairs", [len(swr)])
chk("sweepHeldOverAnalytic", [min(x[1] for x in swh), max(x[1] for x in swh)])
print("   sweep held/analytic:", [(z, round(v, 4), ZT[i]) for z, v, i in swh])
# time-window note: sweep windowed (t0) Phi vs window ratio check
chk("nSweepTotal", [len(S)])
chk("nSweepDone", [len(S)])

print("\n==== SUMMARY of non-OK")
for n, s, v, r in RES:
    if s != "OK":
        print(n, s, v, r)
