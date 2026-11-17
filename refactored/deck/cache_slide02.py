"""Cache the inputs of slide 2 (what MAVEN sees) into small tracked files in deck/data/.

    cd refactored && conda run -n dsmc python deck/cache_slide02.py

Sources (read-only here):
  FIG 2a  orbit 2521 inbound pass (16 Jan 2016) from our reduction of the public NGIMS L2 csn-abund v08_r01
          files: williamson-replication/outputs/tables/profiles.parquet (delta rho / rho_bar per species
          against a 5th-order polynomial fit to log10 density over each species' valid range, as in
          Williamson et al. 2019) and orbits.parquet (exobase altitude at N_tot = 5e18 m^-2, peak amplitudes).
  FIG 2b  every inbound pass of the primary window (Feb 2015 - Nov 2017) from climatology_orbits.parquet:
          each pass's mean |delta rho / rho_bar| per species, binned by the solar zenith angle where the pass
          crosses N_tot = 2e18 m^-2. No event selection, so the trend does not depend on the >40% rule.
  Text    the parsed papers in MarsAtmos/docs/Literature/Markdown Parsed (Williamson et al. 2019:
          threshold, 252 of 4259 orbits, Feb 2015 - Nov 2017; England et al. 2017: amplitudes
          anticorrelated with background temperature; Yigit et al. 2021: up to 40% in the 2018 dust storm).
The replication is not a git repository, so the sources are pinned by SHA-256 in slide02_facts.json.
Every quoted number is asserted against its source, so a changed source stops this script.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DSMC = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs")
WREP = DSMC / "williamson-replication"
TABLES = WREP / "outputs" / "tables"
REPORT = WREP / "report" / "validation_report.md"
LIT = DSMC / "MarsAtmos" / "docs" / "Literature" / "Markdown Parsed"
PAPERS = {
    "williamson2019": "Williamson et al. 2019 - Large amplitude perturbations in the martian exosphere "
                      "seen in MAVEN NGIMS data",
    "england2017": "England et al. 2017 - MAVEN NGIMS observations of atmospheric gravity waves in the "
                   "Martian thermosphere",
    "yigit2021": "Yiğit et al. 2021 - Dust storm‐enhanced gravity wave activity in the martian thermosphere "
                 "observed by MAVEN and implication for atmospheric escape",
}
ORBIT = 2521
SPECIES = ("O", "Ar", "CO2")
SZA_EDGES = np.arange(0.0, 181.0, 15.0)
MIN_PASSES = 50                       # bins with fewer passes are not drawn (0-15 deg has 25)
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paper(key: str) -> Path:
    return LIT / PAPERS[key] / f"{PAPERS[key]}.md"


def text_has(path: Path, *phrases):
    t = path.read_text()
    for p in phrases:
        if p not in t:
            raise SystemExit(f"{path.name}: expected phrase not found: {p!r}")


def main():
    # ---------------------------------------------------------------- literature claims on the slide
    text_has(paper("williamson2019"),
             "Ar or CO<sub>2</sub> peak amplitude greater than 40% of the background density",
             "The O amplitude is significantly smaller than that of Ar or CO<sub>2</sub>",
             "we find 252 examples out of the 4259 orbits from February 2015 to November 2017",
             "slightly larger amplitudes on the nightside",
             "the inbound portion of orbit 2521, from January 16, 2016",
             "5<sup>th</sup> degree polynomial fit to the NGIMS data")
    text_has(paper("england2017"),
             "The observed wave density amplitudes are anticorrelated with the background atmospheric temperature")
    text_has(paper("yigit2021"), "density fluctuations in the thermosphere are up to 40%")

    # ---------------------------------------------------------------- FIG 2a: orbit 2521
    orb = pd.read_parquet(TABLES / "orbits.parquet").set_index("orbit").loc[ORBIT]
    prof = pd.read_parquet(TABLES / "profiles.parquet", filters=[("orbit", "==", ORBIT)])
    prof = prof.sort_values("alt")
    peaks = {sp: float(orb[f"maxA_{sp}"]) for sp in SPECIES}
    rep = REPORT.read_text()
    m = re.search(r"Orbit 2521 peak amplitude O / Ar / CO2 \(Fig\. 4\.1\) \| ~21 / ~65 / ~87 % \| "
                  r"(\d+) / (\d+) / (\d+) %", rep)
    if not m or [round(peaks[s]) for s in SPECIES] != [int(v) for v in m.groups()]:
        raise SystemExit(f"orbit 2521 peaks {peaks} do not match the validation report")
    m = re.search(r"Orbit 2521 exobase altitude \(N_tot = 5e18\) \| ~148-150 km \| ([\d.]+) km", rep)
    if not m or abs(float(orb["z_exo"]) - float(m.group(1))) > 0.05:
        raise SystemExit("orbit 2521 exobase altitude does not match the validation report")
    for sp in SPECIES:                 # the profile's own maximum is the catalogued peak
        a = prof[f"A_{sp}"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(prof["N_tot"].to_numpy(float))
        if abs(np.nanmax(a[ok]) - peaks[sp]) > 1e-6:
            raise SystemExit(f"orbit 2521 {sp}: profile maximum differs from orbits.parquet")
    i_peak = int(np.nanargmax(prof["A_CO2"].to_numpy(float)))
    z_peak = float(prof["alt"].iloc[i_peak])

    arrs = {"alt_km": prof["alt"].to_numpy(np.float32)}
    for sp in SPECIES:
        arrs[f"A_{sp}"] = prof[f"A_{sp}"].to_numpy(np.float32)

    # ---------------------------------------------------------------- FIG 2b: all passes vs SZA
    clim = pd.read_parquet(TABLES / "climatology_orbits.parquet")
    n_pass = len(clim)
    m = re.search(r"Inbound passes analysed: \d+ in [\d-]+ to [\d-]+; (\d+) in the primary window", rep)
    if not m or int(m.group(1)) != n_pass:
        raise SystemExit("pass count in climatology_orbits.parquet does not match the validation report")
    d0, d1 = pd.Timestamp(clim["date"].min()), pd.Timestamp(clim["date"].max())
    if not (d0.year == 2015 and d0.month == 2 and d1.year == 2017 and d1.month == 11):
        raise SystemExit(f"unexpected primary window {d0} - {d1}")
    sza = clim["ref_sza"].to_numpy(float)
    rng = np.random.default_rng(20261117)
    centres, counts, med, lo, hi = [], [], {s: [] for s in SPECIES}, {s: [] for s in SPECIES}, \
        {s: [] for s in SPECIES}
    for a, b in zip(SZA_EDGES[:-1], SZA_EDGES[1:]):
        sel = (sza >= a) & (sza < b)
        if sel.sum() < MIN_PASSES:
            continue
        centres.append(0.5 * (a + b))
        counts.append(int(sel.sum()))
        for sp in SPECIES:
            v = clim.loc[sel, f"meanabsA_{sp}"].dropna().to_numpy(float)
            boot = np.median(rng.choice(v, size=(N_BOOT, v.size)), axis=1)
            med[sp].append(np.median(v))
            lo[sp].append(np.percentile(boot, 16))
            hi[sp].append(np.percentile(boot, 84))
    arrs["sza_deg"] = np.array(centres)
    arrs["n_passes"] = np.array(counts)
    for sp in SPECIES:
        arrs[f"med_{sp}"], arrs[f"lo_{sp}"], arrs[f"hi_{sp}"] = (np.array(x[sp]) for x in (med, lo, hi))

    day, night = sza < 75.0, sza >= 105.0
    split = {sp: {"day_lt75": float(np.nanmedian(clim.loc[day, f"meanabsA_{sp}"])),
                  "night_ge105": float(np.nanmedian(clim.loc[night, f"meanabsA_{sp}"]))} for sp in SPECIES}
    for sp in SPECIES:                 # the slide says "larger at high SZA" and "CO2, Ar > O"
        if not split[sp]["night_ge105"] > 1.4 * split[sp]["day_lt75"]:
            raise SystemExit(f"{sp}: nightside amplitude not clearly above dayside: {split[sp]}")
    if not all(np.all(arrs[f"med_{sp}"] > arrs["med_O"]) for sp in ("Ar", "CO2")):
        raise SystemExit("O is not the smallest amplitude in every SZA bin")

    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / "fig02_ngims.npz", **arrs)
    srcs = [TABLES / "orbits.parquet", TABLES / "profiles.parquet", TABLES / "climatology_orbits.parquet",
            REPORT, WREP / "config.toml"]
    facts = {
        "generated": dt.date.today().isoformat(),
        "orbit": ORBIT, "orbit_date": "2016-01-16",
        "orbit_lst_h": round(float(orb["ref_lst"]), 1), "orbit_sza_deg": round(float(orb["ref_sza"])),
        "orbit_z_exo_km": round(float(orb["z_exo"]), 1),
        "orbit_peak_pct": {sp: round(peaks[sp]) for sp in SPECIES},
        "orbit_peak_alt_km": round(z_peak, 1),
        "threshold_pct": 40, "williamson_orbits": 4259, "williamson_events": 252,
        "williamson_window": "Feb 2015 - Nov 2017",
        "replication_passes": n_pass,
        "sza_bins_deg": [float(c) for c in centres], "sza_bin_passes": counts,
        "median_meanabsA_pct_day_vs_night": {sp: {k: round(v, 1) for k, v in split[sp].items()}
                                             for sp in SPECIES},
        "background": "5th-order polynomial in log10 density over each species' valid range "
                      "(Williamson et al. 2019; wrep.detrend)",
        "sources": {str(p): sha256(p) for p in srcs},
        "papers": {k: str(paper(k)) for k in PAPERS},
    }
    (DATA / "slide02_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in facts.items() if k not in ("sources", "papers")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
