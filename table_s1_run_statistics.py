#!/usr/bin/env python3
"""Measured particle statistics for Table S1 / Text S1 (runs S1-S6, plus the 2025 S4 re-run).

Every number comes from the run's own files:

* particle weight, time step, wave-injection width, band and N_inj from ``Info/``
  (``compare_s4_band_thickness_reductions.read_metadata``);
* particles per cell from ``macro Number Particles`` averaged over the pre-wave background
  window (``processor.bg_window``) and over azimuth, reported at the native radial cells
  nearest 150 / 230 / 320 km;
* the total simulated particle count as the pre-wave sum of particles over all cells;
* the steady-state shot-noise level: relative temporal std of the cell density over the last
  ``NOISE_SAMPLES`` pre-wave samples, azimuth-averaged, and std*sqrt(ppc) (pure Poisson = 1).

S1-S3 are loaded from ``combined_10runs_macro_grids_data.pkl``, a 10-member ensemble MEAN: its
``Number Particles`` is the per-member count, but its density noise is lower than one member's.
For those runs the same noise measurement is repeated on the single-member
``macro_grids_data.pkl``, so the ensemble's real noise reduction is measured rather than assumed
(it is not sqrt(10)).  ``NOISE_SAMPLES = 17`` because that is all the pre-wave data the
ensemble files keep.

Outputs::

    exports/table_s1_run_statistics.csv    all measurements, including the 2025 S4 re-run and noise
    exports/table_s1_run_statistics.json
    exports/table_s1_manuscript.csv         Table S1 as printed: S1-S6 only, no noise column

Usage::

    python table_s1_run_statistics.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pickle

import numpy as np

from compare_s4_band_thickness_reductions import RUNS as S4_RUNS
from compare_s4_band_thickness_reductions import read_metadata, resolve_path
from optimized_loader import OptimizedDSMCDataProcessor
from reconciled_attenuation_analysis import AZI_CELLS, LABELS_REMAP, RESULTS_PATHS

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exports"

PPC_ALTITUDES_KM = (150.0, 230.0, 320.0)
LOW_ALTITUDE_CHECK_KM = 400.0   # minimum-ppc check below this altitude
NOISE_SAMPLES = 17              # pre-wave samples available in the S1-S3 ensemble files


def run_inventory() -> list[tuple[str, int, Path]]:
    runs = [(LABELS_REMAP[n], n, Path(p)) for n, p in zip(AZI_CELLS, RESULTS_PATHS)]
    runs.append(("S4 (2025 re-run)", 512, resolve_path(S4_RUNS["2025"])))
    return runs


def shot_noise(time_s, density, n_particles, wave_start_s, alt_idx) -> tuple[float, float]:
    """Relative temporal density std over the last NOISE_SAMPLES pre-wave samples, and std*sqrt(ppc)."""
    pre = np.where(np.asarray(time_s) < wave_start_s)[0][-NOISE_SAMPLES:]
    x = density[pre, alt_idx, :]
    rel_std = float((x / x.mean(axis=0) - 1.0).std(axis=0, ddof=1).mean())
    ppc = float(n_particles[pre, alt_idx, :].mean())
    return rel_std, rel_std * np.sqrt(ppc)


def measure(label: str, n_azi: int, data_dir: Path) -> dict:
    print(f"\n{'=' * 78}\n{label}: {data_dir}\n{'=' * 78}")
    meta = read_metadata(data_dir)
    info = (data_dir / "Info" / "Simulation_Parameters.ascii").read_text(errors="replace")
    width_line = next(line for line in info.splitlines() if "Wave Horizontal Injection Width" in line)
    n_wedge = int(width_line.split()[-2])

    proc = OptimizedDSMCDataProcessor(data_dir)
    proc.load_parameters()
    proc.load_data()
    proc.get_background_profiles()
    w = proc.bg_window
    radial = np.asarray(proc.params.radial_cells, float)
    ppc = proc.number_particles_data[w["start"]:w["end"]].mean(axis=0)   # (alt, azi)
    ppc_profile = ppc.mean(axis=1)

    r_lower_km = 3389.5 + 100.0   # lower radial boundary, Info/Simulation_Parameters.ascii
    dphi_deg = 360.0 / n_azi
    row = {
        "run": label,
        "n_azi": n_azi,
        "data_dir": str(data_dir),
        "macro_file": proc._macro_data_path.name if proc._macro_data_path else "npy",
        "total_cells": int(ppc.size),
        "dphi_deg": dphi_deg,
        "dphi_km_lower_boundary": np.deg2rad(dphi_deg) * r_lower_km,
        "wedge_cells": n_wedge,
        "forcing_width_km_lower_boundary": n_wedge * np.deg2rad(dphi_deg) * r_lower_km,
        "particle_weight": meta["particle_weight"],
        "band_half_width_deg": meta["band_half_width_deg"],
        "n_inj_per_step": meta["n_inj_per_step"],
        "has_free_escape_ubc_line": meta["has_free_escape_ubc"],
        "end_step": meta["end_step"],
        "bg_window_idx": [int(w["start"]), int(w["end"])],
        "total_particles_prewave": float(ppc.sum()),
        "min_ppc_below_400km": float(ppc[radial <= LOW_ALTITUDE_CHECK_KM].min()),
        "min_ppc_domain": float(ppc.min()),
        # Table S1 caption claims >= 20 particles per cell.
        "highest_alt_km_min_ppc_ge_20": float(radial[np.all(ppc >= 20.0, axis=1)].max()),
    }
    for z in PPC_ALTITUDES_KM:
        i = int(np.argmin(np.abs(radial - z)))
        row[f"ppc_{int(z)}km"] = float(ppc_profile[i])
        row[f"ppc_{int(z)}km_native_alt"] = float(radial[i])
        row[f"ppc_{int(z)}km_azi_std"] = float(ppc[i].std())
    wave_start_s = proc.params.start_step_wave * proc.params.timestep
    single_member = data_dir / "macro_grids_data.pkl"
    is_ensemble = proc._macro_data_path is not None and proc._macro_data_path.name.startswith("combined_")
    if is_ensemble:
        with single_member.open("rb") as fh:
            member = pickle.load(fh)
    for z in PPC_ALTITUDES_KM:
        i = int(np.argmin(np.abs(radial - z)))
        rel_std, poisson_ratio = shot_noise(proc.params.time, proc.density_data, proc.number_particles_data,
                                            wave_start_s, i)
        row[f"noise_rel_std_{int(z)}km"], row[f"noise_std_sqrt_ppc_{int(z)}km"] = rel_std, poisson_ratio
        if is_ensemble:
            rel_std, poisson_ratio = shot_noise(member["Time Stamps"], member["Number Density"],
                                                member["Number Particles"], wave_start_s, i)
        row[f"single_member_noise_rel_std_{int(z)}km"] = rel_std
        row[f"single_member_noise_std_sqrt_ppc_{int(z)}km"] = poisson_ratio
    row["is_10_member_ensemble"] = is_ensemble
    for key, val in row.items():
        print(f"  {key:34s} {val}")
    return row


MANUSCRIPT_RUNS = ("S1", "S2", "S3", "S4", "S5", "S6")


def manuscript_table(rows: list[dict]) -> list[dict]:
    """Table S1 columns. Noise floors stay out of the table (they are quoted in Text S1 with their
    window definition); the forcing width is given in km at the lower boundary and in degrees."""
    table = []
    for r in (r for r in rows if r["run"] in MANUSCRIPT_RUNS):
        table.append({
            "Simulation": r["run"],
            "Total azimuth cells": r["n_azi"],
            "Total cells": r["total_cells"],
            "dPhi (deg)": round(r["dphi_deg"], 4),
            "dPhi r0 (km)": round(r["dphi_km_lower_boundary"], 1),
            "Forced cells": r["wedge_cells"],
            "Forcing region L (km)": round(r["forcing_width_km_lower_boundary"], 1),
            "Forcing region (deg)": round(r["wedge_cells"] * r["dphi_deg"], 2),
            "Particle weight": f"{r['particle_weight']:.3e}",
            "N_inj per step": int(round(r["n_inj_per_step"])),
            "Particles per cell 150 km": round(r["ppc_150km"]),
            "Particles per cell 230 km": round(r["ppc_230km"]),
            "Particles per cell 320 km": round(r["ppc_320km"]),
            "Total particles": f"{r['total_particles_prewave']:.2e}",
            "Highest altitude with >=20 ppc in every cell (km)": round(r["highest_alt_km_min_ppc_ge_20"], 1),
        })
    return table


def write_manuscript_table(rows: list[dict]) -> Path:
    table = manuscript_table(rows)
    path = EXPORT_DIR / "table_s1_manuscript.csv"
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    return path


def main():
    rows = [measure(*run) for run in run_inventory()]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXPORT_DIR / "table_s1_run_statistics.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    json_path = EXPORT_DIR / "table_s1_run_statistics.json"
    json_path.write_text(json.dumps(rows, indent=2))
    table_path = write_manuscript_table(rows)

    print(f"\n{'run':>18s} {'W':>10s} {'N_inj':>6s} {'band':>6s} {'ppc150':>8s} {'ppc230':>8s} "
          f"{'ppc320':>8s} {'N_total':>12s} {'min<400':>8s} {'noise230':>9s} {'1-member':>9s}")
    for r in rows:
        print(f"{r['run']:>18s} {r['particle_weight']:10.3e} {r['n_inj_per_step']:6.0f} "
              f"{r['band_half_width_deg']:6.3f} {r['ppc_150km']:8.0f} {r['ppc_230km']:8.0f} "
              f"{r['ppc_320km']:8.0f} {r['total_particles_prewave']:12.4e} {r['min_ppc_below_400km']:8.1f} "
              f"{r['noise_rel_std_230km']:9.4f} {r['single_member_noise_rel_std_230km']:9.4f}")
    print(f"\nWrote {csv_path}\nWrote {json_path}\nWrote {table_path}")


if __name__ == "__main__":
    main()
