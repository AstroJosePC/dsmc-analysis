"""Cache the inputs of slides 7 and 8 into small tracked files in deck/data/.

    cd refactored && conda run -n dsmc python deck/cache_slides78.py

The sources live in other workspaces (read-only here):
  * truncation-baseline campaign, ~/MarsAtmos-TruncBase (geometry of the Stage 1B deck, wheel-fix
    report numbers) and its Stage 1B runs in ~/MarsAtmos-TruncBase-runs/1b;
  * MAGNUS-P isothermal-O baseline, ~/MAGNUSP-isoO-baseline (Plate 1 panel A profiles), and the
    summary sidecars of Plates 1 and 3 in ../advisor_update_plates (slide 8 callouts);
  * this repo's main checkout: outputs/h2_ubc_rtop/tables/escape_summary.csv (Plate 3 panel A) and
    exports/table_s1_manuscript.csv (Table S1, total particles of S6). Both are gitignored/untracked,
    hence read from the main checkout.
Every number that appears on a slide is asserted against the value quoted in the reports, so a
changed source stops this script instead of silently changing the deck. Each cache records its
source paths and the source repository's commit.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAIN = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer")
TB_REPO = Path.home() / "MarsAtmos-TruncBase"
TB = TB_REPO / "campaigns" / "truncation_baseline"
RUNS_1B = Path.home() / "MarsAtmos-TruncBase-runs" / "1b"
MAGNUSP = Path.home() / "MAGNUSP-isoO-baseline"
S01 = MAGNUSP / "output" / "analysis" / "s01"
PLATES = MAIN.parent / "advisor_update_plates"
R_MARS_KM = 3389.5


def repo_head(repo: Path) -> str:
    """Short commit of a checkout or worktree, read from its git files (no git call)."""
    dotgit = repo / ".git"
    gitdir = Path(dotgit.read_text().split("gitdir:")[1].strip()) if dotgit.is_file() else dotgit
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head[:8]
    ref = head.split()[1]
    common = gitdir / (gitdir / "commondir").read_text().strip() if (gitdir / "commondir").exists() else gitdir
    loose = common / ref
    if loose.exists():
        return loose.read_text().strip()[:8]
    for line in (common / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line[:8]
    raise RuntimeError(f"cannot resolve {ref} in {repo}")


def close(value, expected, tol, what):
    if abs(value - expected) > tol:
        raise SystemExit(f"{what}: {value} differs from the reported {expected} (tol {tol})")


# ================================================================= slide 7
def tex_numbers(path: Path) -> dict[str, float]:
    """\\newcommand{\\name}{\\num{value}} -> {name: float}; Unicode minus and thousands commas."""
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{\\num\{([^}]*)\}\}", path.read_text()):
        try:
            out[name] = float(val.replace("−", "-").replace(",", "").replace("+", ""))
        except ValueError:
            pass                                    # dates and other non-numeric entries
    return out


def macro_particle_total(run: Path) -> tuple[int, float, int]:
    """Sum of the block-averaged particle count (column 7) over all cells at the last output."""
    macro = run / "Results" / "Macro"
    last = max(int(m.group(1)) for p in macro.glob("Output*_Proc0_Macroscopic.ascii")
               if (m := re.match(r"Output(\d+)_", p.name)))
    total, cells = 0.0, 0
    for p in sorted(macro.glob(f"Output{last}_Proc*_Macroscopic.ascii")):
        a = np.loadtxt(p, ndmin=2)
        total += a[:, 6].sum()
        cells += a.shape[0]
    return last, total, cells


def run_costs(dirs: list[Path]) -> dict:
    """Wall time, ranks and span of the finished Stage 1B runs, from each run's provenance.txt and
    wall_seconds (written by jobs/run_1b_local.sh), plus whether the wave runs feed their stream
    through the fluid-coupling boundary (Input/Boundaries/FluidCoupling.input, enable = 1)."""
    if not dirs:
        return {"n": 0}
    prov = [dict(kv.split("=", 1) for kv in (d / "provenance.txt").read_text().split()) for d in dirs]
    wall = np.array([float((d / "wall_seconds").read_text()) for d in dirs])
    nps = {int(p["np"]) for p in prov}
    assert len(nps) == 1 and all(p["rc"] == "0" for p in prov), "1B runs: mixed np or nonzero rc"
    t = [dt.datetime.fromisoformat(p[k].replace("Z", "+00:00")) for p in prov for k in ("start", "end")]
    coupled = [bool(re.search(r"^\s*enable\s*=\s*1\b", (d / "Input" / "Boundaries" / "FluidCoupling.input")
                              .read_text(), re.M)) for d in dirs if d.name.startswith("wave")]
    np_ = nps.pop()
    out = {"n": len(dirs), "n_wave": sum(d.name.startswith("wave") for d in dirs),
           "n_nowave": sum(d.name.startswith("nowave") for d in dirs), "np": np_,
           "wall_h_min": wall.min() / 3600, "wall_h_max": wall.max() / 3600,
           "core_h_min": np_ * wall.min() / 3600, "core_h_max": np_ * wall.max() / 3600,
           "span_h": (max(t) - min(t)).total_seconds() / 3600, "wave_runs_coupled": all(coupled)}
    close(out["wall_h_min"], 7.2, 0.05, "1B wall time, shortest run (h)")
    close(out["wall_h_max"], 8.2, 0.05, "1B wall time, longest run (h)")
    return out


def slide07():
    deck = TB / "decks" / "wave1b" / "Input"
    grid = tomllib.loads((deck / "Grid" / "Grid.toml").read_text())
    bnd = tomllib.loads((deck / "Boundaries" / "Boundaries.toml").read_text())
    run = RUNS_1B / "wave1b_r1"
    radial = np.loadtxt(run / "Results" / "Info" / "Radial_Cells.ascii", ndmin=2)
    sector_deg = grid["dimensions"]["azimuth_range"]
    n_azi = grid["cells"]["azimuth"]
    base_km = bnd["lower"]["altitude"] / 1e3
    width_km = np.deg2rad(sector_deg) * (R_MARS_KM + base_km)
    band_deg = float(re.search(r"zenith band\s+([\d.]+) deg", (run / "run.log").open().read(20000)).group(1))
    top_edge_km = radial[-1, 2] / 1e3 - R_MARS_KM
    params = (run / "Results" / "Info" / "Simulation_Parameters.ascii").read_text()
    weight = float(re.search(r"Particle Weight:\s+(\S+)", params).group(1))
    dt_s = float(re.search(r"Time Step:\s+(\S+)", params).group(1))
    out_n, n_part, n_cells = macro_particle_total(run)

    close(sector_deg, 6.4934, 1e-3, "sector width (deg)")
    close(width_km, 400.0, 0.01, "sector width at the base (km)")
    assert n_azi == 50 and radial.shape[0] == 73 and n_cells == 50 * 73
    close(base_km, 140.0, 1e-9, "base altitude")
    close(top_edge_km, 504.6, 0.05, "top cell edge (km)")
    close(band_deg / 2, 7.2, 1e-6, "band half-width beta (deg)")
    close(n_part, 6.245e6, 0.02 * 6.245e6, "Stage 1B particle total")

    reps = {f"{case}_r{k}": (RUNS_1B / f"{case}_r{k}" / "COMPLETE.ascii").exists()
            for case, n in (("wave1b", 5), ("nowave1b", 3)) for k in range(1, n + 1)}
    runs = run_costs([RUNS_1B / name for name, done in reps.items() if done])

    num = tex_numbers(TB / "reports" / "wheel_fix" / "numbers.tex")
    card = {"chi2_density_12x1d_fix": num["fixXIINChi"], "chi2_density_12x1d_pre": num["preXIINChi"],
            "chi2_density_3x1d_fix": num["fixNChi"], "chi2_temperature_12x1d_fix": num["fixXIITChi"],
            "budget_R_fix": num["fixR"], "budget_R_fix_se": num["fixRse"],
            "escape_rel_pct_fix": num["fixEscRel"], "escape_rel_pct_pre": num["preEscRel"]}
    for k, v in (("chi2_density_12x1d_fix", 1.14), ("chi2_density_12x1d_pre", 2638.36),
                 ("budget_R_fix", -0.2), ("budget_R_fix_se", 0.7),
                 ("escape_rel_pct_fix", -1.1), ("escape_rel_pct_pre", -9.0)):
        close(card[k], v, 1e-9, k)

    s1 = pd.read_csv(MAIN / "exports" / "table_s1_manuscript.csv").set_index("Simulation")
    s6_total = float(s1.loc["S6", "Total particles"])
    s6_cell = float(s1.loc["S6", "dPhi r0 (km)"])
    close(s6_total, 1.08e9, 1e6, "S6 total particles (Table S1)")
    close(s6_cell, 21.4, 1e-9, "S6 azimuthal cell width (km)")

    facts = {
        "created": dt.date.today().isoformat(),
        "sources": {
            "deck": str(deck), "run": str(run), "report": str(TB / "reports" / "wheel_fix" / "numbers.tex"),
            "table_s1": str(MAIN / "exports" / "table_s1_manuscript.csv"),
            "marsatmos_commit": repo_head(TB_REPO),
            "run_solver_rev": (run / "provenance.txt").read_text().split("rev=")[1][:8],
        },
        "geometry": {"sector_deg": sector_deg, "sector_width_km": width_km, "n_azi": n_azi,
                     "azi_cell_km": width_km / n_azi, "base_km": base_km,
                     "top_input_km": bnd["upper"]["altitude"] / 1e3, "top_edge_km": top_edge_km,
                     "n_radial": int(radial.shape[0]), "beta_deg": band_deg / 2},
        "stage1b": {"particles": n_part, "macro_output": out_n, "cells": n_cells, "weight": weight,
                    "dt_s": dt_s, "replicates": reps, "finished": sum(reps.values()), "planned": len(reps)},
        "s6": {"particles": s6_total, "azi_cell_km": s6_cell},
        "runs_1b": runs,
        "null_test": card,
    }
    path = DATA / "slide07_facts.json"
    path.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"{path.name}: sector {sector_deg:.4f} deg = {width_km:.1f} km, top edge {top_edge_km:.1f} km, "
          f"1B {n_part:.4g} particles (S6/1B = {s6_total / n_part:.0f}), "
          f"{facts['stage1b']['finished']}/{len(reps)} replicates finished")


# ================================================================= slide 8
def fig08a():
    p = np.load(S01 / "profiles" / "profile_S01B8.npz")
    z = p["z_m"] / 1e3
    keep = z <= 200.5
    w_model, w_lin = 2 * np.abs(p["c_w"][keep]), 2 * np.abs(p["c_w_linear"][keep])
    z = z[keep]
    peak_z = float(z[np.argmax(np.where(z <= 180, w_model, 0))])
    levels = json.loads((S01 / "isoo_profile.json").read_text())["runs"]["S01B8"]["levels"]
    lv = ("100.25", "120.25", "140.25")
    misfit_pct = np.array([100 * levels[k]["linear_rel_diff"] for k in lv])
    for got, want in zip(misfit_pct, (0.1312, 0.2150, 0.2219)):
        close(got, want, 5e-5, "Plate 1A model-vs-theory misfit (%)")
    close(peak_z, 109.25, 1e-9, "Plate 1A amplitude peak (km)")
    assert int(p["tw_s"]) == 3600 and int(p["lambda_x_m"]) == 400_000 and float(p["amp_m_s"]) == 0.1
    meta = {"source": str(S01 / "profiles" / "profile_S01B8.npz"), "misfits": str(S01 / "isoo_profile.json"),
            "magnusp_commit": repo_head(MAGNUSP), "run": "S01B8 (wave 8: 60 min, 400 km, 0.1 m/s)",
            "created": dt.date.today().isoformat()}
    path = DATA / "fig08a_plate1A.npz"
    np.savez(path, z_km=z, w_model=w_model, w_linear=w_lin, level_km=np.array([100.25, 120.25, 140.25]),
             misfit_pct=misfit_pct, peak_z_km=peak_z, period_s=int(p["tw_s"]),
             lambda_x_km=int(p["lambda_x_m"]) / 1e3, amp_m_s=float(p["amp_m_s"]), meta=json.dumps(meta))
    print(f"{path.name}: {z.size} levels, misfits {np.round(misfit_pct, 4)} %, peak {peak_z} km")


def fig08b():
    src = MAIN / "refactored" / "outputs" / "h2_ubc_rtop" / "tables" / "escape_summary.csv"
    e = pd.read_csv(src)
    h = e[e["species"] == "H2"][["atmosphere", "ubc_num", "ubc", "z_top_nominal_km", "dr_max_km",
                                 "z_top_actual_km", "dsmc_escape_rate_s", "dsmc_escape_rate_se_s"]]
    assert len(h) == 60
    # Plate 3 definition (docs/reports/h2_ubc_rtop/scripts/derived.py, escape_multipliers):
    # free-escape rate / mean reflection+hold rate over all tops and both cell widths
    base = h[h["ubc_num"].isin([1, 2])].groupby("atmosphere")["dsmc_escape_rate_s"].mean()
    u3 = h[h["ubc_num"] == 3]
    mult = u3["dsmc_escape_rate_s"].to_numpy() / base.loc[u3["atmosphere"]].to_numpy()
    at2500 = (u3["z_top_nominal_km"] == 2500).to_numpy()
    close(mult[at2500].min(), 42.40, 0.01, "free-escape overcount at 2,500 km, minimum")
    close(mult[at2500].max(), 89.34, 0.01, "free-escape overcount at 2,500 km, maximum")
    fine = (u3["dr_max_km"] == 250).to_numpy() & at2500
    fine_mult = dict(zip(u3["atmosphere"][fine], mult[fine]))
    close(fine_mult["H2"], 89.34, 0.01, "overcount at 2,500 km, pure H2, 250 km cells")
    close(fine_mult["H2O"], 49.5, 0.05, "overcount at 2,500 km, O+H2, 250 km cells")
    path = DATA / "fig08b_escape.csv"
    with path.open("w") as f:
        f.write(f"# {src} (H2 rows; atmosphere H2O = O+H2) @ dsmc_analyzer {repo_head(MAIN)}, "
                f"{dt.date.today().isoformat()}\n")
        f.write(f"# free-escape overcount at 2500 km, 250 km cells: H2 {fine_mult['H2']:.1f}x, "
                f"O+H2 {fine_mult['H2O']:.1f}x (both cell widths: {mult[at2500].min():.1f}-"
                f"{mult[at2500].max():.1f}x; at 40000 km {mult[~at2500].min():.1f}x and up)\n")
        h.to_csv(f, index=False)
    print(f"{path.name}: 60 rows; overcount at 2,500 km (250 km cells) "
          f"H2 {fine_mult['H2']:.1f}x, O+H2 {fine_mult['H2O']:.1f}x")


def slide08():
    """Callout numbers from the Plate 1 and Plate 3 sidecars (advisor_update_plates/*.json)."""
    p1 = json.loads((PLATES / "plate_01_magnusp_isoo.json").read_text())
    p3 = json.loads((PLATES / "plate_03_dsmc_ubc_sweep.json").read_text())
    facts = {
        "created": dt.date.today().isoformat(),
        "sources": {"plate1": str(PLATES / "plate_01_magnusp_isoo.json"),
                    "plate3": str(PLATES / "plate_03_dsmc_ubc_sweep.json"),
                    "magnusp_commit": repo_head(MAGNUSP)},
        "n_streams": p1["n_streams"], "n_weak_streams": p1["n_B_streams"],
        "linear_ok_waves": p1["mis_ok_n"], "linear_waves": p1["mis_ok_n"] + len(p1["mis_bad"]),
        "linear_ok_max_pct": 100 * p1["mis_ok_max"],
        "strong_wind_m_s": p1["u_peak"], "strong_wind_z_km": p1["u_peak_z_km"], "strong_wind_t_h": p1["u_peak_t_h"],
        "z_top_min_km": p3["z_top_min_km"], "dr_fine_km": p3["dr_fine_km"],
    }
    assert facts["n_streams"] == 57 and facts["n_weak_streams"] == 24
    assert (facts["linear_ok_waves"], facts["linear_waves"]) == (7, 8)
    close(facts["linear_ok_max_pct"], 0.71, 0.005, "weak-forcing misfit, 7 of 8 waves (%)")
    close(facts["strong_wind_m_s"], -6.08, 0.005, "strong-forcing wind (m/s)")
    close(facts["strong_wind_z_km"], 97.25, 1e-9, "strong-forcing wind altitude")
    assert facts["z_top_min_km"] == 2500 and facts["dr_fine_km"] == 250
    path = DATA / "slide08_facts.json"
    path.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"{path.name}: {facts['n_streams']} streams, linear within {facts['linear_ok_max_pct']:.2f}% "
          f"for {facts['linear_ok_waves']}/{facts['linear_waves']} waves, wind {facts['strong_wind_m_s']:.2f} m/s")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    slide07()
    fig08a()
    fig08b()
    slide08()
