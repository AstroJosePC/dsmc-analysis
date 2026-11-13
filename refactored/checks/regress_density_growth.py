"""Regression: staging dsmclib density growth vs the frozen root code, run by run.

    MPLBACKEND=Agg python checks/regress_density_growth.py                 # all S and 1D runs
    MPLBACKEND=Agg python checks/regress_density_growth.py --runs S1 1D_5p

Reference per campaign:
  * S1-S6:  root optimized_loader.OptimizedDSMCDataProcessor.compute_density_growth
            (+ the cached growth_peak_data_<col>azi_*.pkl files in the run folder, 2024 code)
  * 1D:     root loader.DSMCDataProcessor + helper_funcs.process_growth_num_amp, i.e. the
            compare_density_growth.py recipe with the standard background.

The root modules are imported read-only (no bytecode written, no script main() run).
Report: outputs/checks/regress_density_growth.json (+ a text summary on stdout).
"""

import argparse
import gc
import json
import sys
import time

import numpy as np

sys.dont_write_bytecode = True  # never write .pyc into the root __pycache__

from dsmclib import registry  # noqa: E402
from dsmclib.constants import PADDING  # noqa: E402
from dsmclib.growth import compute_density_growth  # noqa: E402
from dsmclib.paths import REFACTORED_ROOT, output_dir  # noqa: E402
from dsmclib.run_data import DSMCRun  # noqa: E402

ROOT_DIR = REFACTORED_ROOT.parent
sys.path.append(str(ROOT_DIR))  # appended, so it cannot shadow dsmclib or the stdlib

ALT_RANGE = (150, 320)


def same_masked(a, b) -> bool:
    """Equal values (NaN == NaN) and equal masks."""
    a_mask, b_mask = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    if a_mask.shape != b_mask.shape or not np.array_equal(a_mask, b_mask):
        return False
    return np.array_equal(np.ma.getdata(a)[~a_mask], np.ma.getdata(b)[~b_mask], equal_nan=True)


def max_abs_diff(a, b) -> float:
    a = np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)
    b = np.ma.filled(np.ma.asarray(b, dtype=float), np.nan)
    if a.shape != b.shape:
        return float("nan")
    d = np.abs(a - b)
    return float(np.nanmax(d)) if np.any(np.isfinite(d)) else 0.0


def compare_columns(staged, reference) -> dict:
    """Column-by-column comparison of two {azi: {growth_peaks, baseline_peaks} or None} dicts."""
    cols = sorted(set(staged) | set(reference))
    identical, differing = [], {}
    for c in cols:
        s, r = staged.get(c), reference.get(c)
        if s is None or r is None:
            if (s is None) != (r is None):
                differing[c] = "failed in one pipeline only"
            else:
                identical.append(c)
            continue
        if same_masked(s["growth_peaks"], r["growth_peaks"]) and same_masked(s["baseline_peaks"], r["baseline_peaks"]):
            identical.append(c)
        else:
            differing[c] = max_abs_diff(s["growth_peaks"], r["growth_peaks"])
    return {"n_columns": len(cols), "n_identical": len(identical), "differing": differing}


def params_diff(p_staged, p_root, names) -> dict:
    out = {}
    for name in names:
        a, b = getattr(p_staged, name), getattr(p_root, name)
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            out[name] = f"staged={np.asarray(a).ravel()[:4]} root={np.asarray(b).ravel()[:4]}"
    return out


def check_2d(alias: str) -> dict:
    from optimized_loader import OptimizedDSMCDataProcessor

    path = registry.run(alias).path
    t0 = time.time()
    staged_run = DSMCRun(path, alias=alias).load()
    staged_run.release_macro()
    staged = compute_density_growth(staged_run, alt_range=ALT_RANGE)
    t_staged = time.time() - t0

    t0 = time.time()
    root = OptimizedDSMCDataProcessor(str(path))
    root.load_parameters()
    root.load_data()
    root._macro_cache = None
    reference = root.compute_density_growth(alt_range=ALT_RANGE)
    t_root = time.time() - t0

    result = {
        "reference": "root optimized_loader.compute_density_growth",
        "params_diff": params_diff(staged_run.params, root.params,
                                   ["radial_cells", "time", "start_step_wave", "final_step_wave", "timestep",
                                    "omega_freq", "num_periods", "n_azi_cell", "mid_azi", "azi_edges"]),
        "density_identical": bool(np.array_equal(staged_run.density_data, root.density_data, equal_nan=True)),
        "bg_window": {"staged": {k: int(v) for k, v in staged_run.background()["window"].items()},
                      "root": {k: int(v) for k, v in root.bg_window.items()}},
        "forced_columns_identical": bool(np.array_equal(staged["azimuthal_indices"], reference["azimuthal_indices"])),
        "forced_columns": [int(staged["azimuthal_indices"][0]), int(staged["azimuthal_indices"][-1])],
        "growth": compare_columns(staged["growth_results"],
                                  {int(k): v for k, v in reference["growth_results"].items()}),
        "cached_pkls": compare_cached_pkls(staged_run, staged),
        "seconds": {"staged": round(t_staged, 1), "root": round(t_root, 1)},
    }
    return result


def compare_cached_pkls(run: DSMCRun, staged: dict) -> dict:
    """Staged growth vs the growth_peak_data_<col>azi_<macro>.pkl caches in the run folder (informational).

    Only caches built from the same macro file the loader uses are compared: for S1-S3 that is
    the combined_10runs ensemble; the few single-member caches next to them are skipped.
    """
    from dsmclib.macro import load_pickle

    source = run.macro_path.name if run.macro_path is not None else "macro_grids_data.pkl"
    files = sorted(run.data_dir.glob(f"growth_peak_data_*azi_{source}"))
    if not files:
        return {"source": source, "n_files": 0}
    identical, differing, shape_mismatch = [], {}, {}
    for f in files:
        col = int(f.name.split("_")[3].removesuffix("azi"))
        cached = load_pickle(f)
        entry = staged["growth_results"].get(col)
        if entry is None:
            differing[col] = "no staged result for this column"
            continue
        s = np.ma.filled(np.ma.asarray(entry["growth_peaks"], dtype=float), np.nan)
        c = np.asarray(cached["peaks"], dtype=float)
        if s.shape != c.shape:
            shape_mismatch[col] = f"staged {s.shape} vs cached {c.shape}"
        elif np.array_equal(s, c, equal_nan=True):
            identical.append(col)
        else:
            differing[col] = max_abs_diff(s, c)
    return {"source": source, "n_files": len(files), "n_identical": len(identical),
            "differing": differing, "shape_mismatch": shape_mismatch}


def check_cached_only(alias: str) -> dict:
    """Staged growth from outputs/<campaign>/cache/ vs the run-folder caches, without re-running root."""
    from dsmclib.growth import cached_growth

    run = DSMCRun(registry.run(alias).path, alias=alias)
    return compare_cached_pkls(run, cached_growth(alias))


def check_1d(alias: str) -> dict:
    from loader import DSMCDataProcessor
    from helper_funcs import process_growth_num_amp as root_process_growth_num_amp

    path = registry.run(alias).path
    staged_run = DSMCRun(path, alias=alias).load()
    staged = compute_density_growth(staged_run, alt_range=ALT_RANGE)

    # compare_density_growth.py recipe (standard background, dynamic peaks, column 0)
    root = DSMCDataProcessor(str(path))
    root.load_parameters()
    root.load_data()
    background = root.get_background_profiles(None)
    dn = (root.density_data - background["density"][None, :, None]) / background["density"][None, :, None]
    t = root.params.time
    cell_min = np.where(root.params.radial_cells > ALT_RANGE[0])[0][0]
    cell_max = np.where(root.params.radial_cells > ALT_RANGE[1])[0][0]
    growth_peaks, baseline_peaks = root_process_growth_num_amp(
        dn[..., 0],
        (root.params.start_step_wave * root.params.timestep <= t)
        & (t <= root.params.final_step_wave * root.params.timestep + PADDING),
        root.params.radial_cells, cell_min, cell_max, root.params.num_periods,
        prominence=1e-1, distance=12,
        sampling_rate=1 / np.mean(np.diff(t)), cutoff_frequency=root.params.omega_freq * 1.2, filter_order=2,
        use_dynamic=True,
    )
    return {
        "reference": "root loader.DSMCDataProcessor + helper_funcs.process_growth_num_amp",
        "params_diff": params_diff(staged_run.params, root.params,
                                   ["radial_cells", "time", "start_step_wave", "final_step_wave", "timestep",
                                    "omega_freq", "num_periods"]),
        "density_identical": bool(np.array_equal(staged_run.density_data, root.density_data, equal_nan=True)),
        "bg_window": {"staged": {k: int(v) for k, v in staged_run.background()["window"].items()},
                      "root": {k: int(v) for k, v in root.bg_window.items()}},
        "forced_columns": [int(c) for c in staged["azimuthal_indices"]],
        "growth": compare_columns(staged["growth_results"],
                                  {0: {"growth_peaks": growth_peaks, "baseline_peaks": baseline_peaks}}),
    }


def main():
    default = [r.alias for r in registry.campaign("s1_s6_azimuthal")] + [r.alias for r in registry.campaign("radial_1d")]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=default, help="registry aliases (default: all S and 1D runs)")
    parser.add_argument("--cached-only", action="store_true",
                        help="only compare staging caches with the run-folder growth_peak_data pkls")
    args = parser.parse_args()

    if args.cached_only:
        path = output_dir("checks", "tables") / "regress_density_growth_cached_pkls.json"
        out = {alias: check_cached_only(alias) for alias in args.runs if registry.run(alias).campaign != "radial_1d"}
        path.write_text(json.dumps(out, indent=2, default=str))
        for alias, res in out.items():
            print(f"[{alias}] {res}")
        return

    report_path = output_dir("checks", "tables") / "regress_density_growth.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for alias in args.runs:
        print(f"\n===== {alias} =====", flush=True)
        run = registry.run(alias)
        report[alias] = check_1d(alias) if run.campaign == "radial_1d" else check_2d(alias)
        report_path.write_text(json.dumps(report, indent=2, default=str))
        g = report[alias]["growth"]
        verdict = "IDENTICAL" if g["n_identical"] == g["n_columns"] and not report[alias]["params_diff"] else "DIFFERS"
        print(f"[{alias}] {verdict}: {g['n_identical']}/{g['n_columns']} columns identical; "
              f"params_diff={report[alias]['params_diff']}; bg={report[alias]['bg_window']}", flush=True)
        if "cached_pkls" in report[alias]:
            print(f"[{alias}] cached pkls: {report[alias]['cached_pkls']}", flush=True)
        gc.collect()
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
