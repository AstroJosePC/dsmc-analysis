"""Regression: reproduce the 2024 run-folder growth_peak_data_<col>azi_*.pkl files from raw data.

    MPLBACKEND=Agg python checks/regress_legacy_2024_growth.py             # S1 S2 S3 S6 (S4/S5 have no pkls)
    MPLBACKEND=Agg python checks/regress_legacy_2024_growth.py --runs S1

These pkls fed the published S1-S6 growth (zenodo 2d_growth_peak_data.parquet). Their generator
is lost; the recipe reconstructed here is
  * background "azimean_snapshot": azimuthal mean of the snapshot 2 outputs before forcing starts,
  * time window: forcing start .. forcing end + 600 s,
  * altitudes: the cells NEAREST to 150 km and 320 km (148.3 and 315.4 km),
  * the unchanged dsmclib peak detection (dynamic, prominence 0.1, distance 12).
Each column's 'peaks' table and 'baseline_peak_amps' should match bit for bit; matches within
ROUNDING (summation order of the azimuthal mean; S1) are reported separately. Known exception:
S6 column 512 (0.07 off even from its own stored series: the 2024 peaks were picked by hand).
Report: outputs/checks/tables/regress_legacy_2024_growth.json
"""

import argparse
import contextlib
import gc
import io
import json
import pickle
from pathlib import Path

import numpy as np

from dsmclib import registry
from dsmclib.growth import GROWTH_PEAK_KWARGS
from dsmclib.paths import output_dir
from dsmclib.peaks import process_growth_num_amp
from dsmclib.run_data import DSMCRun

LEGACY_PADDING = 600.0
LEGACY_ALT_RANGE = (150.0, 320.0)
ROUNDING = 1e-12


def check_run(alias: str) -> dict:
    entry = registry.run(alias)
    run = DSMCRun(entry.path, alias=alias).load()
    source = run.macro_path.name if run.macro_path else "macro_grids_data.pkl"
    run.release_macro()
    files = sorted(Path(entry.path).glob(f"growth_peak_data_*azi_{source}"))
    if not files:
        return {"n_files": 0}

    pert = run.density_perturbation(background="azimean_snapshot")
    mask = run.forcing_time_mask(LEGACY_PADDING)
    c0, c1 = (run.altitude_index(a, "nearest") for a in LEGACY_ALT_RANGE)

    identical, rounding, differing = [], [], {}
    for f in files:
        col = int(f.name.split("_")[3].replace("azi", ""))
        with open(f, "rb") as fh:
            legacy = pickle.load(fh)
        with contextlib.redirect_stdout(io.StringIO()):
            growth, baseline = process_growth_num_amp(
                pert[..., col], mask, run.params.radial_cells, c0, c1, run.params.num_periods,
                use_dynamic=True, **GROWTH_PEAK_KWARGS)
        ours = np.ma.filled(growth.astype(float), np.nan)
        ref = np.ma.filled(np.ma.asarray(legacy["peaks"], dtype=float), np.nan)
        same = (ours.shape == ref.shape and np.array_equal(ours, ref, equal_nan=True)
                and np.array_equal(np.asarray(baseline, float), np.asarray(legacy["baseline_peak_amps"], float)))
        diff = float(np.nanmax(np.abs(ours - ref))) if ours.shape == ref.shape else float("nan")
        if same:
            identical.append(col)
        elif diff <= ROUNDING and np.array_equal(np.isnan(ours), np.isnan(ref)):
            rounding.append(col)
        else:
            differing[col] = {"max_abs_diff": diff, "shape": [list(ours.shape), list(ref.shape)]}
        del legacy
    result = {"n_files": len(files), "source": source, "cells": [c0, c1],
              "altitudes_km": [float(run.params.radial_cells[c0]), float(run.params.radial_cells[c1])],
              "identical": identical, "within_rounding": rounding, "differing": differing}
    del run, pert
    gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["S1", "S2", "S3", "S6"])
    args = parser.parse_args()

    report = {}
    for alias in args.runs:
        report[alias] = check_run(alias)
        r = report[alias]
        print(f"{alias}: {r['n_files']} legacy pkls: {len(r.get('identical', []))} bit-identical, "
              f"{len(r.get('within_rounding', []))} within {ROUNDING:g}"
              + (f", differing: { {c: round(v['max_abs_diff'], 4) for c, v in r['differing'].items()} }"
                 if r.get("differing") else ""), flush=True)

    out = output_dir("checks", "tables") / "regress_legacy_2024_growth.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")
