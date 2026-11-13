"""Simulation cards (dsmclib.card) vs independent sources.

    python checks/regress_sim_card.py            # every registered run that has a run file

For each run the card is re-extracted from its run file (nothing rendered) and compared with:
  1. the raw solver Info/Simulation_Parameters.ascii, parsed here with its own regexes (not the
     builder's): time step, steps, sampling/averaging, boundaries, cell counts, species masses and
     weights, collision reduced masses, wave start/end/frequency/amplitude;
  2. dsmclib.ncrun (an independent reader of the run file): output count, end time, averaging window,
     domain inventory of computational particles per species at the last output;
  3. the raw Rates ASCII is not re-read here (regress_oh2_netcdf_vs_ascii covers run file == ASCII).
For the OH2 runs the card must also reproduce the values the O+H2 report prints in Table 1
(docs/reports/oh2_escape_rerun/numbers.tex).
Every existing card PDF must be exactly one page.
Report: outputs/checks/tables/regress_sim_card.json; exits 1 on any failure.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from dsmclib import registry
from dsmclib.card import CARD_SUFFIX, _page_count, extract
from dsmclib.ncrun import RUN_SUFFIX, open_run_nc
from dsmclib.paths import REFACTORED_ROOT, output_dir

warnings.simplefilter("ignore", RuntimeWarning)
NUMBERS = REFACTORED_ROOT / "docs" / "reports" / "oh2_escape_rerun" / "numbers.tex"
FLOAT = r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)"


def info_params(results: Path) -> dict:
    t = (results / "Info" / "Simulation_Parameters.ascii").read_text()

    def one(key):
        m = re.search(re.escape(key) + r"\s*:?\s*" + FLOAT, t)
        return float(m.group(1).replace("D", "E")) if m else None

    def many(key):
        return [float(v.replace("D", "E")) for v in re.findall(r"^\s*" + re.escape(key) + r":\s*" + FLOAT, t, re.M)]

    masses = many("Mass")  # planet (kg) first, then species (amu)
    return {
        "dt": one("Time Step"), "n_steps": one("Number of Total Simulations Steps"),
        "samp": one("Samples Taken Every"), "avg": one("Averages Taken Every"),
        "r_lo_km": one("Lower Radial Boundary of Grid"), "r_hi_km": one("Upper Radial Boundary of Grid"),
        "planet_km": one("Radius"), "n_rad": one("Number of Radial Cells"), "n_zen": one("Number of Zenith Cells"),
        "n_azi": one("Number of Azimuth Cells"), "species_mass_amu": masses[1:], "weights": many("Particle Weight"),
        "reduced": many("Reduced Mass"), "wave_start": one("Start Step of Wave Injection"),
        "wave_end": one("End Step of Wave Injection"), "wave_omega": one("Wave Injection Frequency"),
        "wave_amp": one("Wave Injection Amplitude"),
    }


def report_macros() -> dict:
    if not NUMBERS.is_file():
        return {}
    out = {}
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", NUMBERS.read_text(), re.M):
        nums = re.findall(r"\\num\{([^}]*)\}|\\sci\{([^}]*)\}\{([^}]*)\}", v)
        vals = []
        for plain, mant, exp in nums:
            if plain:
                vals.append(float(plain.replace(r"\,", "").replace("\u2212", "-")))
            else:
                vals.append(float(mant) * 10 ** int(exp.replace("\u2212", "-")))
        out[k] = vals
    return out


def close(a, b, rtol=1e-9, atol=0.0):
    return a is not None and b is not None and bool(np.isclose(a, b, rtol=rtol, atol=atol))


def main():
    macros = report_macros()
    report, failures = {}, []
    runs = [r for r in registry.all_runs() if (r.path / f"{r.alias}{RUN_SUFFIX}").is_file()]
    for r in runs:
        c = extract(r.alias)
        ip = info_params(r.path)
        d, tm, f = c["domain"], c["time"], c["forcing"]
        checks = {
            "dt": close(tm["dt_s"], ip["dt"]),
            "n_steps": tm["n_steps"] == int(ip["n_steps"]),
            "samp/avg steps": tm["samp_steps"] == int(ip["samp"]) and tm["avg_steps"] == int(ip["avg"]),
            "lower boundary": close(d["lower_alt_km"], ip["r_lo_km"] - ip["planet_km"], rtol=1e-12, atol=1e-6),
            "upper boundary": close(d["upper_alt_km"], ip["r_hi_km"] - ip["planet_km"], rtol=1e-12, atol=1e-6),
            "cells": (d["n_radial"], d["n_zenith"], d["n_azimuthal"]) == tuple(int(x) for x in (ip["n_rad"], ip["n_zen"], ip["n_azi"])),
            "species masses": np.allclose([s["mass_amu"] for s in c["species"]], ip["species_mass_amu"], rtol=1e-6),
            "weights": np.allclose([s["weight"] for s in c["species"]], ip["weights"], rtol=1e-12),
            "reduced masses": np.allclose([x["reduced_mass_amu"] for x in c["collisions"]], ip["reduced"], rtol=1e-12),
            "wave start/end": f["start_step"] == int(ip["wave_start"] or 0) and f["end_step"] == int(ip["wave_end"] or 0),
            "wave omega/amplitude": close(f["omega_rad_s"], ip["wave_omega"] or 0.0, atol=1e-15)
            and close(f["amplitude"], ip["wave_amp"] or 0.0, atol=1e-15),
        }
        run = open_run_nc(r.alias)
        npart_last = run.macro.num_part.isel(time=-1).sum("radial").values
        checks.update({
            "outputs (ncrun)": tm["n_outputs"] == run.macro.sizes["time"],
            "end time (ncrun)": close(tm["t_end_s"], run.t_end),
            "avg window (ncrun)": close(tm["avg_window_s"], float(np.median(run.macro.avg_time_interval.values))),
            "N_part last (ncrun)": np.allclose([s["npart_final"] for s in c["species"]], npart_last, rtol=1e-9),
        })
        del run
        if r.campaign == "oh2_escape_rerun" and macros:   # the O+H2 report's runs (not drift_mb's OH2_s*)
            ab = "A" if r.alias == "OH2_A" else "B"
            sp = {s["name"]: s for s in c["species"]}
            exp = {f"topAlt{ab}": round(d["upper_alt_km"]), f"nOut{ab}": tm["n_outputs"],
                   f"avg{ab}": round(tm["avg_window_s"], 2), f"tEnd{ab}": round(tm["t_end_s"]),
                   "weightO": float(f"{sp['O']['weight']:.2e}"), "weightH": float(f"{sp['H2']['weight']:.2e}"),
                   "nzeroO": float(f"{sp['O']['n0_m3'][0]:.2e}"), "nzeroH": float(f"{sp['H2']['n0_m3'][0]:.2e}"),
                   "Tzero": round(sp["O"]["T0_K"][0]), "dtStep": round(tm["dt_s"], 4), "nSteps": tm["n_steps"]}
            if f["in_run"]:
                exp.update({"tWave": round(f["start_time_s"]), "period": round(f["period_s"], 1),
                            "periodsInRun": round(f["periods_in_run"], 2)})
            for k, v in exp.items():
                checks[f"report {k}"] = bool(macros.get(k)) and np.isclose(macros[k][0], v, rtol=1e-9)
        pdf = r.path / f"{r.alias}{CARD_SUFFIX}.pdf"
        if pdf.is_file():
            checks["card is one page"] = _page_count(pdf) == 1
        bad = [k for k, ok in checks.items() if not ok]
        failures += [f"{r.alias}: {k}" for k in bad]
        report[r.alias] = {"checks": len(checks), "failed": bad,
                           "status": [f"{s['level']}: {s['label']}" for s in c["status"]]}
        print(f"{r.alias:24s} {len(checks) - len(bad):2d}/{len(checks)} ok" + (f"  FAILED {bad}" if bad else ""), flush=True)
    out = {"runs": report, "pass": not failures, "failures": failures}
    path = output_dir("checks", "tables") / "regress_sim_card.json"
    path.write_text(json.dumps(out, indent=1))
    print(("PASS" if not failures else "FAIL") + f" ({len(runs)} runs) -> {path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
