"""Automated consistency checks on the drift_mb analysis outputs (run after compute_analysis.py).

    python -B checks/validate_drift_mb.py

Writes outputs/drift_mb/tables/validation.csv (one row per check and run/species) and exits 1 if any check FAILs.
WARN rows are physics expectations that are reported, not enforced.

Checks
  completeness   22 per-run analysis files + the cross-run file; tier attributes; expected dimensions
  provenance     source_sha256 of every analysis file equals the current run files' SHA-256
  nan            required variables finite (profiles: on resolved cells)
  schedule       320 forcing + 80 ring-down outputs, 25 outputs per phase bin (phase locking)
  restart        restart_rate_factor = spin-up interval / forced interval exactly; corrected inject_rate * dt / W
                 equals the drift-diagnostic injection count (independent groups of the run file)
  bookkeeping    sum of folded VDF counts = sum of injection counts; escape-speed histogram total = escape counts
  mass_balance   N_phys(end) - N_phys(start) = sum (inject - return - escape) dt over the run, relative to the
                 number injected (< 2e-3; interval-mean inventories limit this to shot noise)
  injection      window-mean injected number / momentum / energy rates vs drifting-Maxwellian theory (5 sigma)
  vdf            KS D / D_crit(0.1 %) < 1 per group; chi2 p > 0.01 / n_groups (Bonferroni); |z| of the six
                 moments < 4.5
  controls       s = 0 controls vs spin-up: escape enhancement and exobase dT consistent with 0 (3 sigma)
  null           forced u_r amplitude at 150 km > 5 x the control noise floor and > 5 x its standard error
  linearity      [WARN] u_r(150 km) amplitude ratio s=0.2/s=0.1 and s=0.1/s=0.05 within 20 % of 2
  common         [WARN] common-window baseline rule time_matched for every forced run
  crossrun       cross-run file holds all 22 runs and the key scalars of the forced runs are finite
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import xarray as xr

from dsmclib import registry, tiers
from dsmclib.ncrun import RUN_SUFFIX, open_run_nc
from dsmclib.paths import output_dir

CAMPAIGN = "drift_mb"
rows: list[dict] = []


def rec(check, run, ok, value="", detail="", species="", level="FAIL"):
    status = "PASS" if ok else level
    rows.append({"check": check, "run": run, "species": species, "status": status, "value": value, "detail": detail})


def load(path):
    with xr.open_dataset(path) as ds:
        return ds.load()


def main():
    runs = registry.campaign(CAMPAIGN, include_alternates=True)
    aliases = [r.alias for r in runs]
    rec("completeness", "campaign", len(aliases) == 22, len(aliases), "registered runs")
    meta = pd.read_csv(output_dir(CAMPAIGN, "tables") / "run_summary.csv").set_index("run")
    cr_path = tiers.analysis_path(CAMPAIGN, CAMPAIGN)
    rec("completeness", "cross-run", cr_path.exists(), str(cr_path.name))
    an = {}
    for r in runs:
        p = tiers.analysis_path(CAMPAIGN, r.alias)
        ok = p.exists()
        rec("completeness", r.alias, ok, p.name)
        if not ok:
            continue
        ds = load(p)
        an[r.alias] = ds
        a = ds.attrs
        rec("completeness", r.alias, a.get("data_tier") == "analysis" and a.get("convention_version") == tiers.CONVENTION_VERSION,
            a.get("data_tier"), "tier attributes")
        srcs = a["source_files"].split(",")
        shas = a["source_sha256"].split(",")
        good = all(tiers.sha256(s) == h for s, h in zip(srcs, shas))
        rec("provenance", r.alias, good, len(srcs), "source_sha256 matches current run files")

    forced = [a for a in aliases if not a.startswith("SPIN")]
    for a in forced:
        ds = an[a]
        sch_ok = (int((ds.forcing == 1).sum()) == 320 and int((ds.forcing == 0).sum()) == 80
                  and all(int(((ds.phase_index == j) & (ds.forcing == 1)).sum()) == 20 for j in range(16))
                  and ds.sizes["period"] == 25 and ds.sizes["vdf_group"] == 17)
        rec("schedule", a, sch_ok, f"{int((ds.forcing == 1).sum())}+{int((ds.forcing == 0).sum())}")
        # NaN checks
        for v in ("escape_rate", "escape_npart", "Ndot_inj", "Pdot_inj", "Edot_inj", "Ndot_ret", "inj_mean_y", "wdrift",
                  "vdf_count", "vdf_model_prob", "vdf_ks_D", "vdf_mean_unb", "vdf_var_hist", "escape_rate_period"):
            rec("nan", a, bool(np.all(np.isfinite(ds[v].values))), v)
        res = ds["resolved"].values.astype(bool)
        for v in ("ur_amp", "ur_phase", "dT_amp", "dn_amp", "dT_cyclemean", "baseline_num_dens"):
            vals = ds[v].values
            rec("nan", a, bool(np.all(np.isfinite(vals[res]))), v, "resolved cells")

        # restart correction and cross-group identity, from the run file itself
        run = open_run_nc(registry.run(a).path / f"{a}{RUN_SUFFIX}")
        spin = "SPIN_O" if a.startswith("O_") else "SPIN_OH2"
        sp_avg = int(open_run_nc(registry.run(spin).path / f"{spin}{RUN_SUFFIX}").attrs["avg_steps"])
        expect = sp_avg / int(run.attrs["avg_steps"])
        fac = float(ds.attrs["restart_rate_factor"])
        rec("restart", a, abs(fac - expect) < 1e-9, f"{fac:.6f}", f"expected {expect:.6f} (spin avg / forced avg)")
        dtrue = int(run.attrs["avg_steps"]) * run.dt
        inj_rate = run.surface["inject_rate"].sum("boundary_cell").transpose("species", "time").values * fac
        cnt = run.diagnostics["diagnostics_drift"]["inj_count"].isel(boundary_cell=0).transpose("species", "time").values
        W = run.weight.values[:, None]
        rec("restart", a, bool(np.allclose(inj_rate * dtrue / W, cnt, rtol=1e-9, atol=1e-6)),
            f"{np.max(np.abs(inj_rate * dtrue / W - cnt)):.2e}", "corrected inject_rate*dt/W == drift inj_count")
        # bookkeeping
        for i, sp in enumerate(ds.species.values):
            sp = str(sp)
            vsum = int(ds["vdf_count"].sel(species=sp).sum())
            isum = int(ds["inj_npart"].sel(species=sp).sum())
            rec("bookkeeping", a, vsum == isum, f"{vsum}", f"folded VDF counts vs injections {isum}", sp)
            esum = float(ds["escape_vdf_forcing"].sel(species=sp).sum() + ds["escape_vdf_ringdown"].sel(species=sp).sum())
            nsum = float(ds["escape_npart"].sel(species=sp).sum())
            rec("bookkeeping", a, esum == nsum, f"{esum:.0f}", f"escape-speed histogram vs escape counts {nsum:.0f}", sp)
            # mass balance from the (corrected) rates and the inventory
            N = (run.macro["num_dens"] * run.cell_volume()).sum("radial").sel(species=sp).values
            surf = run.surface.sum("boundary_cell").sel(species=sp)
            esc = run.escape["escape_rate"].sum("boundary_cell").sel(species=sp).values * fac
            net = (surf["inject_rate"].values - surf["return_rate"].values) * fac - esc
            dN = N[-1] - N[0]
            flow = float(np.sum(net[1:]) * dtrue)
            inj_tot = float(np.sum(surf["inject_rate"].values[1:] * fac) * dtrue)
            rel = abs(dN - flow) / inj_tot
            rec("mass_balance", a, rel < 2e-3, f"{rel:.2e}", f"|dN - sum(net dt)| / injected (dN={dN:.3e}, flow={flow:.3e})", sp)
    # injection vs theory (tables)
    b = pd.read_csv(output_dir(CAMPAIGN, "tables") / "flux_budget.csv")
    for _, r in b.iterrows():
        ds = an[r.run]
        n = float(ds["inj_npart"].sel(species=r.species).values[:].mean()) * r.n_outputs
        tol = 5 / math.sqrt(n)
        for k in ("inj_over_theory", "mom_over_theory", "energy_over_theory"):
            rec("injection", r.run, abs(r[k] - 1) < tol + 1e-6, f"{r[k]:.6f}", f"{k}, window {r.window}, tol {tol:.1e}", r.species)
    # VDF statistics
    v = pd.read_csv(output_dir(CAMPAIGN, "tables") / "vdf_moments.csv")
    ngroups = len(v)
    for _, r in v.iterrows():
        rec("vdf", r.run, r.ks_ratio < 1.0, f"{r.ks_ratio:.3f}", f"KS D/Dcrit {r.group}", r.species)
        rec("vdf", r.run, r.chi2_p > 0.01 / ngroups, f"{r.chi2_p:.2e}", f"chi2 p {r.group} (Bonferroni {0.01 / ngroups:.1e})", r.species)
        zmax = max(abs(r[k]) for k in ("mean_z", "var_z", "skew_z", "exkurt_z", "y2_z", "vt2_z"))
        rec("vdf", r.run, zmax < 4.5, f"{zmax:.2f}", f"max |z| of moments {r.group}", r.species)
    # controls vs spin-up
    e = pd.read_csv(output_dir(CAMPAIGN, "tables") / "escape_summary.csv")
    h = pd.read_csv(output_dir(CAMPAIGN, "tables") / "heating.csv")
    for c in ("O_s000_P330", "OH2_s000_P330"):
        for _, r in e[(e.run == c) & e.enhancement.notna()].iterrows():
            rec("controls", c, abs(r.enhancement) < 3 * r.enhancement_se, f"{r.enhancement:+.3f}",
                f"escape vs spin-up, se {r.enhancement_se:.3f}", r.species)
        for _, r in h[h.run == c].iterrows():
            rec("controls", c, abs(r.dT_exo) < 3 * r.dT_exo_se, f"{r.dT_exo:+.2f} K", f"exobase dT vs spin-up, se {r.dT_exo_se:.2f}", r.species)
        ds = an[c]
        z = (ds["ur_amp"] / ds["ur_amp_se"]).values[ds["resolved"].values.astype(bool)]
        frac = float(np.mean(z > 3.5))
        rec("controls", c, frac < 0.05, f"{frac:.3f}", "fraction of resolved cells with u_r amp/se > 3.5 at omega (null)")
    # forced signal above the noise floor and linearity
    w = pd.read_csv(output_dir(CAMPAIGN, "tables") / "wave_response.csv")
    w = w.join(meta[["s_O", "period_s", "atmosphere", "is_control"]], on="run")
    for _, r in w[w.is_control == 0].iterrows():
        rec("null", r.run, r.ur_amp_150 > 5 * r.ur_null_150 and r.ur_amp_150 > 5 * r.ur_amp_se_150, f"{r.ur_amp_150 / r.ur_null_150:.1f}",
            "u_r(150 km) / control noise floor", r.species)
    for (atm, sp, P), g in w[w.is_control == 0].groupby(["atmosphere", "species", "period_s"]):
        a = g.set_index("s_O")["ur_amp_150"]
        if {0.05, 0.1, 0.2} <= set(a.index):
            for hi, lo in ((0.2, 0.1), (0.1, 0.05)):
                ratio = a[hi] / a[lo]
                rec("linearity", f"{atm}_P{P:.0f}", abs(ratio / 2 - 1) < 0.2, f"{ratio:.3f}",
                    f"u_r(150 km) s={hi}/s={lo}", sp, level="WARN")
    for a in forced:
        if a.startswith(("O_s000", "OH2_s000")):
            continue
        rule = an[a].attrs.get("common_baseline_rule", "")
        rec("common", a, rule == "time_matched", rule, "common-window baseline", level="WARN")
    # cross-run
    if cr_path.exists():
        cr = load(cr_path)
        rec("crossrun", "cross-run", sorted(map(str, cr.run.values)) == sorted(aliases), cr.sizes["run"], "runs")
        fr = [x for x in forced if "s000" not in x]
        for k in ("wave_c_phase", "heat_dT_exo", "budget_fit_inj_over_theory", "vdf_max_ks_ratio"):
            vals = cr[k].sel(run=fr, species="O").values
            rec("crossrun", "cross-run", bool(np.all(np.isfinite(vals))), k, "finite for forced runs (O)")
        vals = cr["esc_enhancement"].sel(run=[x for x in fr if x.startswith("OH2")], species="H2").values
        rec("crossrun", "cross-run", bool(np.all(np.isfinite(vals))), "esc_enhancement", "finite for OH2 forced runs (H2)")

    df = pd.DataFrame(rows)
    out = output_dir(CAMPAIGN, "tables") / "validation.csv"
    df.to_csv(out, index=False)
    summ = df.groupby(["check", "status"]).size().unstack(fill_value=0)
    print(summ.to_string())
    bad = df[df.status != "PASS"]
    if len(bad):
        print("\nNon-PASS rows:")
        print(bad.to_string(index=False))
    n_fail = int((df.status == "FAIL").sum())
    print(f"\n{len(df)} checks: {int((df.status == 'PASS').sum())} PASS, {int((df.status == 'WARN').sum())} WARN, {n_fail} FAIL -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
