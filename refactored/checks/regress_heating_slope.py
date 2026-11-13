"""Regression: staging dsmclib.heating vs the frozen root heating-slope code (Fig 3b / Table 2).

    MPLBACKEND=Agg python checks/regress_heating_slope.py            # S6
    MPLBACKEND=Agg python checks/regress_heating_slope.py --runs S6 S4

Reference: root optimized_loader.OptimizedDSMCDataProcessor.compute_temperature_heating_rate
(constant T0 = 270 K) on the two centre and two edge forced columns, then the
zenodo_release/Figure_2_left_right_panel.py slope: np.polyfit of the column-averaged
last-cycle <dT> over radial_cells[19:36], times H0. Checks that the per-column profiles are
bit-identical and that the staged slope equals the root slope. The staged result is also
written to the cache (outputs/<campaign>/cache/temperature_last_cycle_<run>.pkl).
Report: outputs/checks/tables/regress_heating_slope.json
"""

import argparse
import json
import pickle
import sys

import numpy as np

sys.dont_write_bytecode = True  # never write .pyc into the root __pycache__

from dsmclib import heating, registry  # noqa: E402
from dsmclib.paths import REFACTORED_ROOT, mirror_legacy_cache, output_dir  # noqa: E402
from dsmclib.run_data import DSMCRun  # noqa: E402

sys.path.append(str(REFACTORED_ROOT.parent))  # appended, so it cannot shadow dsmclib


def check_run(alias: str) -> dict:
    from optimized_loader import OptimizedDSMCDataProcessor

    entry = registry.run(alias)
    run = DSMCRun(entry.path, alias=alias).load()
    run.release_macro()
    temp = heating.compute_temperature_last_cycle(run)
    with open(heating.cache_path(entry.campaign, alias), "wb") as f:
        pickle.dump(temp, f)
    mirror_legacy_cache(heating.cache_path(entry.campaign, alias))
    staged = heating.group_heating_slopes(temp)
    groups = heating.centre_and_edge_columns(temp["azimuthal_indices"])
    del run

    root = OptimizedDSMCDataProcessor(str(entry.path))
    root.load_parameters()
    root.load_data()
    root._macro_cache = None
    cols = groups["center"] + groups["edge"]
    ref = root.compute_temperature_heating_rate(azimuthal_indices=cols, H=heating.H0_KM)["heating_rate_results"]
    radial = root.params.radial_cells

    result = {"columns": groups, "cells": [temp["cell_min_idx"], temp["cell_max_idx"]],
              "altitudes_km": [float(temp["altitudes"][0]), float(temp["altitudes"][-1])], "per_column": {}, "slopes": {}}
    for c in cols:
        s, r = temp["temp_last_cycle"][c], ref[c]["temp_last_cycle"]
        result["per_column"][c] = bool(np.array_equal(np.ma.filled(s, np.nan), np.ma.filled(r, np.nan), equal_nan=True))
    for name, pair in groups.items():
        profile = 0.5 * (ref[pair[0]]["temp_last_cycle"] + ref[pair[1]]["temp_last_cycle"])
        root_slope = np.polyfit(radial[19:36], profile, 1)[0] * heating.H0_KM
        result["slopes"][name] = {"root_K_H": float(root_slope), **{k: (float(v) if not isinstance(v, list) else v)
                                                                    for k, v in staged[name].items()}}
        result["slopes"][name]["match"] = bool(np.isclose(root_slope, staged[name]["slope_K_H"], rtol=1e-10, atol=0))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["S6"])
    args = parser.parse_args()
    report = {}
    for alias in args.runs:
        r = report[alias] = check_run(alias)
        print(f"{alias}: cells {r['cells']} ({r['altitudes_km'][0]:.1f}-{r['altitudes_km'][1]:.1f} km), "
              f"profiles identical: {r['per_column']}", flush=True)
        for name, s in r["slopes"].items():
            print(f"  {name} {s['columns']}: root {s['root_K_H']:.4f} K/H0, staged {s['slope_K_H']:.4f} "
                  f"± {s['se_K_H']:.4f} (SE), 95% t half-width {s['ci_half_K_H']:.4f}, n={s['n']}, "
                  f"R²={s['r2']:.3f}  {'PASS' if s['match'] else 'FAIL'}", flush=True)
    out = output_dir("checks", "tables") / "regress_heating_slope.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report: {out}")
